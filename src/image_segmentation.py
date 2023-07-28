import argparse
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Optional

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from skimage.segmentation import slic, mark_boundaries
from skimage.color import lab2rgb, label2rgb
from skimage.measure import regionprops_table


from .image_loading import ImageLoaded, ImageFilepathAdapter
from .layout import LayoutDetector, Layout


logger = logging.getLogger(__name__)


class Segments:

    def __init__(self, image: ImageLoaded, layout: Optional[Layout]):
        self.logger = ImageFilepathAdapter(logger, {"image_filepath": str(image.filepath)})
        self.logger.debug(f"Segment")
        self.image = image
        self.args: argparse.Namespace = image.args
        self.layout = layout
        self.mask = None
        self.boundaries = None
        self.regions = None

    def segment(self):
        self.logger.debug(f"Perform SLIC for superpixel identification")
        self.mask: np.ndarray = slic(
            self.image.rgb,
            mask=self.layout.mask if self.layout is not None else None,
            n_segments=self.args.num_superpixels,
            compactness=self.args.superpixel_compactness,
            sigma=self.args.sigma,
            max_num_iter=10,  # todo do we want to consider passing this up as an arg?
            start_label=1,  # segments start at 1 so that we can differentiate these from mask background
            # enforce_connectivity=True,  # todo investigate this - doesn't seem to prevent segments spanning circles
            # this slic implementation has broken behaviour when using the existing lab image
            # just allowing this function to redo conversion from rgb
            # todo work out what this is doing differently when using the already converted lab image
            # could it be scaling - some rgb is 1 to 255 but here i think is 0-1?
            convert2lab=True  # see above
        )
        self.logger.debug(f"SLIC complete: {len(np.unique(self.mask)) -1} segments")
        self.fix_duplicate_ids()
        self.logger.debug("Calculate region properties for each segment")
        self.regions = pd.DataFrame(regionprops_table(
                self.mask,
                intensity_image=self.image.lab,
                properties=('label', 'centroid'),
                extra_properties=(self.median_intensity,)
        ))
        self.logger.debug("Properties calculated")
        self.regions.set_index('label', inplace=True)
        self.regions.rename(
            columns={
                'centroid-0': 'y',
                'centroid-1': 'x',
                'median_intensity-0': 'L',
                'median_intensity-1': 'a',
                'median_intensity-2': 'b'
            },
            inplace=True
        )
        self.regions[['R', 'G', 'B']] = lab2rgb(self.lab)
        if self.args.image_debug <= 0:
            self.logger.debug("Draw average colour image")
            fig = self.image.figures.new_figure("SLIC segmentation")
            fig.plot_image(label2rgb(self.mask, self.image.rgb, kind='avg'), "Labels (average)")
            self.logger.debug("Add segment ID labels to average colour image")
            for i, r in self.centroids.iterrows():
                fig.add_label(str(i), (r['x'], r['y']), 1-self.rgb.loc[i], 2)
            fig.print(large=True)

    def fix_duplicate_ids(self):
        if self.layout is not None:
            # The skimage slic output includes segment labels that can span circles,
            # despite the mask separating them, even if enforce_connectivity is set.
            # Clean these annotations up by iterating through circles
            # and relabel segments if found in another circle - ensuring they are then unique.
            self.logger.debug('Relabel segments spanning multiple circles')
            circles_per_segment = defaultdict(int)
            segment_counter = np.max(self.mask)

            for circle in self.layout.circles:
                circle_mask = self.layout.get_circle_mask(circle)
                circle_segments = self.mask.copy()
                circle_segments[~circle_mask] = -1  # to differentiate the background in circle from background outside
                circle_segment_ids = set(np.unique(circle_segments))
                circle_segment_ids.remove(-1)
                for i in list(circle_segment_ids):
                    circles_per_segment[i] += 1
                    if circles_per_segment[i] > 1 or i == 0:  #
                        # add a new segment for this ID in this circle
                        segment_counter += 1
                        self.mask[circle_segments == i] = segment_counter
            self.logger.debug(f"Relabelling complete: {len(np.unique(self.mask)) - 1} distinct segments")

    def get_segments(self):
        self.segment()
        return self

    def mark_boundaries(self):
        if self.mask is None:
            self.logger.debug("Cannot mark boundaries without first segmenting, running segmentation first")
            self.segment()
        self.logger.debug("Create boundary image")
        self.boundaries = mark_boundaries(self.image.rgb, self.mask, background_label=0, color=(1, 0, 1))

    @staticmethod
    def median_intensity(mask, array):
        return np.median(array[mask], axis=0)

    @property
    def lab(self):
        return self.regions[['L', 'a', 'b']]

    @property
    def rgb(self):
        return self.regions[['R', 'G', 'B']]

    # add text label for each segment at the centroid for that superpixel
    @property
    def centroids(self):
        return self.regions[['x', 'y']]

    #def distances(self, reference_colours):
    #    reference_colours = np.array(reference_colours)
    #    distances = pd.DataFrame(
    #        data=np.array([deltaE_cie76(self.image.lab, r) for r in reference_colours]).transpose(),
    #        # todo consider replacing deltaE_cie76 with np.linalg.norm, both are just Euclidean distance in Lab?
    #        index=self.image.lab.index,
    #        columns=[str(r) for r in reference_colours]
    #    )
    #    return distances


# The Segmentor handles  multiprocessing of segmentation which is used during calibration from multiple images.
class Segmentor:
    def __init__(self, images: List[ImageLoaded]):
        self.images = images
        self.args = self.images[0].args
        self.image_to_segments = dict()
        self.rgb, self.lab = None, None

    def run(self):
        if self.args.processes > 1:
            self._multiprocess()
        else:
            self._process()
        self._summarise()

    def get_segments(self, image: ImageLoaded):
        logger.info(f"Segment image: {image.filepath}")
        if self.args.whole_image:
            layout = None
        else:
            layout: Layout = LayoutDetector(image).get_layout()
        segments: Segments = Segments(image, layout).get_segments()
        segments.mark_boundaries()
        return segments

    def _multiprocess(self):
        with ProcessPoolExecutor(max_workers=self.args.processes, mp_context=get_context('spawn')) as executor:
            future_to_image = {executor.submit(self.get_segments, image): image for image in self.images}
            for future in as_completed(future_to_image):
                image = future_to_image[future]
                try:
                    segments = future.result()
                    self.image_to_segments[image] = segments
                except Exception as exc:
                    image.logger.info(f'Exception occurred: {exc}')
                else:
                    image.logger.info(f'Successfully segmented')
        logger.debug(f"images processed: {len(self.image_to_segments.keys())}")

    def _process(self):
        for image in self.images:
            try:
                segments = self.get_segments(image)
                self.image_to_segments[image] = segments
            except Exception as exc:
                image.logger.info('%r generated an exception: %s' % (image.filepath, exc))
            else:
                image.logger.info(f'Successfully segmented')
        logger.debug(f"images processed: {len(self.image_to_segments.keys())}")

    def _summarise(self):
        if len(list(self.image_to_segments.keys())) != len(self.images):
            self.images = list(self.image_to_segments.keys())  # in case some did not segment properly
            logger.warning(f"Some images did not complete segmentation: reduced to {len(self.images)}")
        if len(self.images) == 0:
            logger.warning("No images were segmented, this is likely due to a failure to detect the layout")
            raise ValueError("Could not complete calibration")
        # Prepare some summary dataframes for the segment colours in rgb and lab colourspaces
        self.rgb = pd.concat(
            {image.filepath: seg.rgb for image, seg in self.image_to_segments.items()},
            axis=0,
            names=['filepath', 'sid']
        )
        self.lab = pd.concat(
            {image.filepath: seg.lab for image, seg in self.image_to_segments.items()},
            axis=0,
            names=['filepath', 'sid']
        )
