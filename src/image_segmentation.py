import argparse
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import List

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from skimage.segmentation import slic, mark_boundaries
from skimage.color import deltaE_cie76, lab2rgb, label2rgb
from skimage.measure import regionprops_table

from .figurebuilder import FigureBuilder
from .logging import CustomAdapter
from .image_loading import ImageLoaded
from .layout import LayoutDetector, Layout

logger = logging.getLogger(__name__)


class Segments:

    def __init__(self, image, layout):
        self.logger = CustomAdapter(logger, {'image_filepath': str(image.filepath)})
        self.logger.debug(f"Segment: {image.filepath}")
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
            mask=self.layout.mask,
            n_segments=self.args.num_superpixels,
            compactness=self.args.superpixel_compactness,
            sigma=self.args.sigma,
            max_num_iter=10,  # todo do we want to consider passing this up as an arg?
            start_label=1,  # segments start at 1 so that we can differentiate these from mask background
            enforce_connectivity=True,  # todo investigate this - doesn't seem to prevent segments spanning circles
            # this slic implementation has broken behaviour when using the existing lab image
            # just allowing this function to redo conversion from rgb
            # todo work out what this is doing differently when using the already converted lab image
            # could it be scaling - some rgb is 1 to 255 but here i think is 0-1?
            convert2lab=True  # see above
        )
        self.logger.debug(f"SLIC complete: {len(np.unique(self.mask)) -1} segments")
        # The slic output includes segment labels that can span circles, despite the mask
        # This breaks graph building but also seems to not reflect accurate clustering todo investigate this
        # Clean it up by iterating through circles and relabel segments if found in another circle
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
                if circles_per_segment[i] > 1 or i == 0: #
                    # add a new segment for this ID in this circle
                    segment_counter += 1
                    self.mask[circle_segments == i] = segment_counter
        self.logger.debug(f"Relabelling complete: {len(np.unique(self.mask)) - 1} segments")
        self.logger.debug("Calculate region properties")
        self.regions = pd.DataFrame(regionprops_table(
                self.mask,
                intensity_image=self.image.lab,
                properties=('label', 'centroid'),
                extra_properties=(self.median_intensity,)
        ))
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

        if self.args.debug:
            fig = FigureBuilder(self.image.filepath, self.args, "Superpixel labels")
            fig.add_image(label2rgb(self.mask, self.image.rgb, kind='avg'), "Labels (average)")
            for i, r in self.centroids.iterrows():
                fig.current_axis.text(
                    x=r['x'],
                    y=r['y'],
                    s=i,
                    size=2,
                    color=1-self.rgb.loc[i],  # invert of mean RGB colour to ensure label is visible
                    horizontalalignment='center',
                    verticalalignment='center'
                )
            fig.print(large=True)

    def get_segments(self):
        self.segment()
        return self

    def mark_boundaries(self):
        if self.mask is None:
            self.logger.debug("Cannot mark boundaries without first segmenting, running segmentation first")
            self.segment()
        self.logger.debug("Create boundary image")
        self.boundaries = mark_boundaries(self.image.rgb, self.mask, background_label=0)

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

    def distances(self, reference_colours):  #, reference_dist, reference_label):
        reference_colours = np.array(reference_colours)
        distances = pd.DataFrame(
            data=np.array([deltaE_cie76(self.lab, r) for r in reference_colours]).transpose(),
            # todo consider replacing deltaE_cie76 with np.linalg.norm, both are just Euclidean distance
            index=self.lab.index,
            columns=[str(r) for r in reference_colours]
        )
        return distances


# The Segmentor handles  multiprocessing of segmentation which is used during calibration from multiple images.
class Segmentor:   # todo consider not using layout for segmentation during calibration
    def __init__(self, image_filepaths: List[Path], args: argparse.Namespace):
        self.image_filepaths = image_filepaths
        self.args = args
        self.filepath_to_segments = dict()
        self.rgb, self.lab = None, None

    def run(self):
        if self.args.processes > 1:
            self._multiprocess()
        else:
            self._process()
        self._summarise()

    @staticmethod
    def get_segments(filepath: Path, args: argparse.Namespace):
        image = ImageLoaded(filepath, args)
        layout: Layout = LayoutDetector(image).get_layout()
        segments: Segments = Segments(image, layout).get_segments()
        segments.mark_boundaries()
        return segments

    def _multiprocess(self):
        with ProcessPoolExecutor(max_workers=self.args.processes, mp_context=get_context('spawn')) as executor:
            future_to_file = {executor.submit(self.get_segments, filepath, self.args): filepath for filepath in
                              self.image_filepaths}
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                adapted_logger = CustomAdapter(logger, {'image_filepath': str(filepath)})
                try:
                    segments = future.result()
                    self.filepath_to_segments[filepath] = segments
                except Exception as exc:
                    adapted_logger.info(f'Exception occurred: {exc}')

    def _process(self):
        for filepath in self.image_filepaths:
            try:
                segments = self.get_segments(filepath, self.args)
                self.filepath_to_segments[filepath] = segments
            except Exception as exc:
                print('%r generated an exception: %s' % (filepath, exc))
        logger.debug(f"images processed: {len(self.filepath_to_segments.keys())}")

    def _summarise(self):
        if len(list(self.filepath_to_segments.keys())) != len(self.image_filepaths):
            logger.warning("Some images selected for calibration did not complete segmentation")
            self.image_filepaths = list(self.filepath_to_segments.keys())  # in case some did not segment properly
        # Prepare some summary dataframes for the segment colours in rgb and lab colourspaces
        self.rgb = pd.concat(
            {fp: seg.rgb for fp, seg in self.filepath_to_segments.items()},
            axis=0,
            names=['filepath', 'sid']
        )
        self.lab = pd.concat(
            {fp: seg.lab for fp, seg in self.filepath_to_segments.items()},
            axis=0,
            names=['filepath', 'sid']
        )