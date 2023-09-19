import argparse
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Optional, Callable

import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor

from skimage.segmentation import slic, felzenszwalb, find_boundaries, mark_boundaries
from skimage.color import lab2rgb, label2rgb
from skimage.measure import regionprops_table



from .image_loading import ImageLoaded, ImageFilepathAdapter
from .layout import LayoutDetector, Layout, LayoutLoader
from .logging import worker_log_configurer, logger_thread

logger = logging.getLogger(__name__)


class Segments:

    def __init__(self, image: ImageLoaded, layout: Optional[Layout] = None):
        self.logger = ImageFilepathAdapter(logger, {"image_filepath": str(image.filepath)})
        self.logger.debug(f"Segment")
        self.image = image
        self.args: argparse.Namespace = image.args
        self.layout = layout
        self.mask = None
        self.boundaries = None
        self.regions = None

    def segment(self):
        if self.args.superpixels:
            self.logger.debug(f"Perform SLIC for superpixel identification")
            self.mask: np.ndarray = slic(
                self.image.rgb,
                mask=self.layout.mask if self.layout is not None else None,
                n_segments=self.args.superpixels,
                compactness=self.args.compactness if self.args.compactness >= 0 else -self.args.compactness,
                sigma=self.args.sigma,
                slic_zero=True if self.args.compactness < 0 else False,
                max_num_iter=self.args.slic_iter,
                start_label=1,  # segments start at 1 so that we can differentiate these from mask background
                enforce_connectivity=True,
                # this slic implementation has broken behaviour when using the existing lab image
                # just allowing this function to redo conversion from rgb
                # todo work out what this is doing differently when using the already converted lab image
                # could it be scaling - some rgb is 1 to 255 but here i think is 0-1?
                convert2lab=True  # see above
            )
        else:
            self.logger.debug(f"Perform Felzenszwalb superpixel identification")
            masked_image = self.image.lab.copy()
            self.mask: np.ndarray = felzenszwalb(masked_image, scale=10000, sigma=self.args.sigma)
            # we do the felzenszwalb on the whole image as it is reasonably fast and behaves strangely on a masked image
            # then we can just

            self.mask += 1
            self.mask[~self.layout.mask] = 0

        self.logger.debug(f"Segmentation complete: {len(np.unique(self.mask)) -1} segments")
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
            fig = self.image.figures.new_figure("Unsupervised segmentation")
            fig.plot_image(label2rgb(self.mask, self.image.rgb, kind='avg'), "Labels (average)")
            #self.logger.debug("Add segment ID labels to average colour image")
            #for i, r in self.centroids.iterrows():
            #    fig.add_label(str(i), (r['x'], r['y']), 1-self.rgb.loc[i], 2)
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
        self.set_boundaries()
        return self

    def set_boundaries(self):
        if self.mask is None:
            self.logger.debug("Cannot mark boundaries without first segmenting, running segmentation first")
            self.segment()
        self.logger.debug("Create boundary image")
        self.boundaries = find_boundaries(self.mask)
        #self.boundaries = mark_boundaries(self.image.rgb, self.mask, background_label=0, color=(1, 1, 1))

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


# The Segmentor handles  multiprocessing of segmentation which is used during calibration from multiple images.
class Segmentor:
    def __init__(self, images: List[ImageLoaded]):
        self.images = images
        self.args = self.images[0].args
        self.image_to_segments = dict()
        self.rgb, self.lab = None, None

    def run(self, progress_callback: Callable):
        self._multiprocess(progress_callback)
        self._summarise()

    def get_segments(self, image: ImageLoaded, log_queue=None):
        if log_queue is not None:
            worker_log_configurer(log_queue)

        if self.args.whole_image:
            layout = None
        elif self.args.fixed_layout is not None:
            layout: Layout = LayoutLoader(image).get_layout()
        else:
            layout: Layout = LayoutDetector(image).get_layout()
        try:
            segments: Segments = Segments(image, layout).get_segments()
            return segments
        except Exception as exc:
            logger.info('%r generated an exception: %s' % (image.filepath, exc))

    def _multiprocess(self, progress_callback: Callable):
        with multiprocessing.Manager() as manager:
            log_queue = manager.Queue(-1)
            lp = threading.Thread(target=logger_thread, args=(log_queue,))
            lp.start()

            executor = ProcessPoolExecutor(
                    max_workers=self.args.processes,
                    mp_context=multiprocessing.get_context('spawn')
            )
            future_to_image = {
                executor.submit(
                    self.get_segments, image=image, log_queue=log_queue): image for image in self.images
            }

            while True:
                num_completed = sum([future.done() for future in future_to_image.keys()])
                num_total = len(future_to_image.keys())
                complete_percent = int(num_completed/num_total * 100)
                #logger.debug(f"Working {complete_percent}% complete")
                if complete_percent == 100:
                    progress_callback(complete_percent)
                    break
                progress_callback(complete_percent)
                #sleep(0.1)

            for future, image in future_to_image.items():
                try:
                    self.image_to_segments[image] = future.result()
                except Exception as exc:
                    image.logger.info(f"Exception occurred: {exc}")
                else:
                    image.logger.info(f"Successfully segmented")

            executor.shutdown()
            log_queue.put(None)
            lp.join()

            logger.debug(f"images processed: {len(self.image_to_segments.keys())}")

    def _summarise(self):
        if len(list(self.image_to_segments.keys())) != len(self.images):
            self.images = list(self.image_to_segments.keys())  # in case some did not segment properly
            logger.warning(f"Some images did not complete segmentation: reduced to {len(self.images)}")
        if len(self.images) == 0:
            logger.warning("No images were segmented, this is likely due to a failure to detect the layout")
            raise ValueError("Could not complete calibration")
        # Prepare some summary dataframes for the segment colours in rgb and lab colourspaces
        self.lab = pd.concat(
            {image.filepath: seg.lab for image, seg in self.image_to_segments.items()},
            axis=0,
            names=['filepath', 'sid']
        )
        self.rgb = pd.concat(
            {image.filepath: seg.rgb for image, seg in self.image_to_segments.items()},
            axis=0,
            names=['filepath', 'sid']
        )
