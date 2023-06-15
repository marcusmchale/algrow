import argparse
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from skimage.segmentation import slic, mark_boundaries
from skimage.color import deltaE_cie76, lab2rgb, label2rgb
from skimage.measure import regionprops_table

from .figurebuilder import FigureBuilder
from .logging import CustomAdapter


logger = logging.getLogger(__name__)


class Segments:

    def __init__(self, image, layout):
        self.logger = CustomAdapter(logger, {'image_filepath': str(image.filepath)})
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
            # this slic implementation has broken behaviour when using the existing lab transformation and convert2lab=false
            # just allowing this function to redo conversion from rgb
            # todo work out what this is doing differently when using the already converted lab image
            # could it be scaling - some rgb is 1 to 255 but here i think is 0-1?
            mask=self.layout.mask,
            n_segments=self.args.num_superpixels,
            compactness=self.args.superpixel_compactness,
            sigma=self.args.sigma,
            max_num_iter=10,  # todo do we want to consider passing this up as an arg?
            start_label=1,  # segments start at 1 so that we can differentiate these from mask background
            enforce_connectivity=True,
            #slic_zero=True,  # tried this out - here compactness is the initial compactness,
            convert2lab=True  # see above
        )
        self.logger.debug(f"SLIC complete: {len(np.unique(self.mask)) -1} segments")
        # The slic output includes segment labels that span circles
        # This breaks graph building but also seems to not reflect accurate clustering todo investigate this
        # Clean it up by iterating through circles and relabel segments if found in another circle
        self.logger.debug('Clean up segments spanning multiple circles')
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
        self.logger.debug(f"Cleanup complete: {len(np.unique(self.mask)) - 1} segments")
        self.logger.debug("Calculate region properties table")
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
                    color=1-self.rgb.loc[i],  # invert of mean RGB colour for label to stand out
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

        #todo move this debugging to where the method is called
        #if debug:
        #    fig = FigureBuilder(
        #        self.image.filepath,
        #        f"Segment colours in CIELAB with {reference_label}"
        #    )
        #    # convert segment colours to rgb to colour scatter points
        #    segment_colours_rgb = pd.DataFrame(
        #        data=lab2rgb(segment_colours),
        #        index=segment_colours.index,
        #        columns=['R', 'G', 'B']
        #    )
        #    fig.add_subplot(projection='3d')
        #    fig.current_axis.scatter(
        #        xs=segment_colours['a'],
        #        ys=segment_colours['b'],
        #        zs=segment_colours['L'],
        #        s=10,
        #        c=segment_colours_rgb,
        #        lw=0
        #    )
        #    fig.current_axis.set_xlabel('a')
        #    fig.current_axis.set_ylabel('b')
        #    fig.current_axis.set_zlabel('L')
        #    for i, r in segment_colours.iterrows():
        #        fig.current_axis.text(x=r['a'], y=r['b'], z=r['L'], s=i, size=3)
        #
        #    # todo add sphere for each target
        #    # draw sphere
        #    def plt_spheres(ax, list_center, list_radius):
        #        for c, r in zip(list_center, list_radius):
        #            # draw sphere
        #            # adapted from https://stackoverflow.com/questions/64656951/plotting-spheres-of-radius-r
        #            u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
        #            x = r * np.cos(u) * np.sin(v)
        #            y = r * np.sin(u) * np.sin(v)
        #            z = r * np.cos(v)
        #            ax.plot_surface(x + c[1], y + c[2], z + c[0], color=lab2rgb(c), alpha=0.2)
        #
        #    plt_spheres(fig.current_axis, reference_colours, np.repeat(reference_dist, len(reference_colours)))
        #    fig.animate()
        #    fig.print()
        #
        #    fig = FigureBuilder(self.filepath, f"Segment ΔE from any {reference_label} colour")
        #    distances_copy = distances.copy()
        #    distances_copy.loc[0] = np.repeat(0, distances_copy.shape[1])  # add back a distance for background segments
        #    distances_copy.sort_index(inplace=True)
        #    dist_image = distances_copy.min(axis=1).to_numpy()[segments]
        #    fig.add_image(dist_image, f"Minimum ΔE any {reference_label} target colour", color_bar=True, diverging=True,
        #                  midpoint=reference_dist)
        #    fig.print()
        #
        #    # add debug for EACH reference colour in a single figure, to help debug reference colour selection
        #    fig = FigureBuilder(self.filepath, f"Segment ΔE from each {reference_label} colour",
        #                        nrows=len(reference_colours))
        #    for i, c in enumerate(reference_colours):
        #        dist_image = distances_copy[str(c)].to_numpy()[segments]
        #        fig.add_image(
        #            dist_image,
        #            str(c),
        #            color_bar=True,
        #            diverging=True,
        #            midpoint=reference_dist
        #        )
        #    fig.print()
        #return distances
