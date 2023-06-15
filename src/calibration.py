import logging
import argparse
from importlib import reload

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from alphashape import alphashape, optimizealpha
from trimesh import PointCloud, proximity


from typing import List

from .image_loading import ImageLoaded
from .layout import LayoutDetector, Layout
from .image_segmentation import Segments
from .figurebuilder import FigureBuilder

logger = logging.getLogger(__name__)

# todo make all logging start with the filename so can make sense of this when multithreading


def image_worker(filepath: Path, args: argparse.Namespace):
    logger.debug(f"Processing file for calibration: {filepath}")
    image = ImageLoaded(filepath, args)
    logger.debug(f"Detect layout for: {filepath}")
    layout: Layout = LayoutDetector(image).get_layout()
    logger.debug(f"Segment: {filepath}")
    segments: Segments = Segments(image, layout).get_segments()
    logger.debug(f"Prepare a segment boundary image for clicker")
    segments.mark_boundaries()
    logger.debug(f"Done preparing image: {filepath}")
    return segments


class Clicker:
    def __init__(self, image_filepaths: List[Path], args: argparse.Namespace):
        logger.debug("Start calibration window")
        self.image_filepaths = image_filepaths
        self.args = args

        # The below are defined by the selector windows and updated when we disconnect it
        self.selection = defaultdict(set)
        self.selected_lab = None
        self.alpha = None
        self.alpha_shape = None

        # Now prepare some images to open in the selector window
        self.filepath_segments = dict()
        with ProcessPoolExecutor(max_workers=args.processes, mp_context=get_context('spawn')) as executor:
            future_to_file = {executor.submit(image_worker, filepath, args): filepath for filepath in image_filepaths}
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    segments = future.result()
                    self.filepath_segments[filepath] = segments
                except Exception as exc:
                    print('%r generated an exception: %s' % (filepath, exc))
        logger.debug(f"images processed: {len(self.filepath_segments.keys())}")

        # Prepare some summary dataframes for the segments
        self.segments_lab = pd.concat(
            {fp: seg.lab for fp, seg in self.filepath_segments.items()},
            axis=0,
            names=['filepath', 'label']
        )
        self.segments_rgb = pd.concat(
            {fp: seg.rgb for fp, seg in self.filepath_segments.items()},
            axis=0,
            names=['filepath', 'label']
        )

        # Prepare the plots that will be used by the selector
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(f'Selection for target colours')
        self.ax.set_title("Click to select/deselect representative regions for target colour")

        self.ind = 0  # index for the images used when navigating prev/next

        axdelta = self.fig.add_axes([0.1, 0, 0.5, 0.025])
        self.delta_slider = Slider(ax=axdelta, label="Delta", valmin=0, valmax=50, valinit=self.args.delta)

        axprev = self.fig.add_axes([0.8, 0, 0.075, 0.05])
        self.bprev = Button(axprev, 'Prev')
        self.bprev.on_clicked(self.prev)

        axnext = self.fig.add_axes([0.9, 0, 0.075, 0.05])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next)

        # Load the first image into the figure
        first_image_segments = self.filepath_segments[image_filepaths[self.ind]]
        self.artist = self.ax.imshow(first_image_segments.boundaries, picker=True)

        # Launch the selector
        self.selector = ClickSelect(
            first_image_segments,
            self.fig,
            self.artist,
            self.delta_slider
        )
        plt.show()
        # when we are done we update values from the final selector window back to self
        self.disconnect_selector()


    def plot_all(self, path):
        fig = FigureBuilder(path, self.args, 'LAB colourspace (all sampled images) with alpha shape selection')
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            xs=self.segments_lab['a'],
            ys=self.segments_lab['b'],
            zs=self.segments_lab['L'],
            s=10,
            c=self.segments_rgb,
            lw=0
        )
        ax.plot_trisurf(
            *zip(
                *self.alpha_shape.vertices[:, [1, 2, 0]]
            ),
            triangles=self.alpha_shape.faces[:, [1, 2, 0]],
            color=(0, 1, 0, 0.5)
        )
        fig.animate()
        fig.print()

    def next(self, _):
        if self.ind < len(self.filepath_segments) - 1:
            self.ind += 1
            self._change_image()

    def prev(self, _):
        if self.ind > 0:
            self.ind -= 1
            self._change_image()

    def _change_image(self):
        logger.debug('load new image')
        # disconnect the old selector and update the stored values
        self.disconnect_selector()

        # prepare the set of existing selected colours
        selection_indices = [(k, i) for k, v in self.selection.items() for i in v]
        prior_lab: set = set(self.segments_lab.loc[selection_indices].apply(tuple, axis=1))

        # create the new selector
        self.selector = ClickSelect(
            self.filepath_segments[self.image_filepaths[self.ind]],
            self.fig,
            self.artist,
            self.delta_slider,
            existing_selection=self.selection[self.image_filepaths[self.ind]],
            prior_lab=prior_lab
        )
        self.fig.canvas.draw()
        logger.debug('new image loaded')

    def disconnect_selector(self):
        prior_image_path = self.selector.segments.image.filepath
        self.selection[prior_image_path] = self.selector.selection
        self.alpha = self.selector.alpha
        self.alpha_shape = self.selector.alpha_shape
        selection_indices = [(k, i) for k, v in self.selection.items() for i in v]
        self.selected_lab = self.segments_lab.loc[selection_indices].to_numpy().round(decimals=1)

        colours_string = f'{[",".join([str(j) for j in i]) for i in self.selected_lab]}'.replace("'", '"')
        logger.info(f'Colours selected: {colours_string}')
        self.selector.disconnect()



class ClickSelect:

    def __init__(self, segments, fig, artist, delta_slider, existing_selection=None, prior_lab=None):

        self.segments = segments
        self.displayed_img = self.segments.boundaries.copy()

        self.fig = fig
        self.artist = artist
        self.delta_slider = delta_slider

        if existing_selection is not None:
            self.selection = existing_selection
        else:
            self.selection = set()

        if prior_lab is not None:
            self.prior_lab = prior_lab
        else:
            self.prior_lab = set()

        self.lab_fig = plt.figure("Lab colourspace with selection")
        self.lab_ax = plt.axes(projection='3d')
        self.lab_ax.scatter(xs=self.segments.lab['a'], ys=self.segments.lab['b'], zs=self.segments.lab['L'], s=10,
                            c=self.segments.rgb, lw=0)
        #  todo consider plotting all points across all images,
        #   this would help to see clusters
        #   maybe color only those from current image to differentiate

        self.alpha_shape = None
        self.alpha = None
        self.trisurf = None

        self.click = self.fig.canvas.mpl_connect('pick_event', self.onclick)
        # self.shift_held = False
        # self.keypress = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        # self.keyrelease = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.delta_slider.on_changed(self.update_img)
        self.update_img()


    #def on_key_press(self, event):
    #    if event.key == 'shift':
    #        self.shift_held = True
    #
    #def on_key_release(self, event):
    #    if event.key == 'shift':
    #        self.shift_held = False

    def onclick(self, event):
        x = event.mouseevent.xdata.astype(int)
        y = event.mouseevent.ydata.astype(int)
        segment = self.segments.mask[y, x]
        logger.debug(f'segment: {segment}, x: {x}, y: {y}')
        if segment == 0:
            return
        #if self.shift_held:
        #    if segment in self.selection:
        #        self.selection.remove(segment)
        #    #if segment in self.background:
        #    #    self.background.remove(segment)
        #    else:
        #        self.background.add(segment)
        #else:
        #   if segment in self.background:
        #       self.background.remove(segment)
        if segment in self.selection:
            self.selection.remove(segment)
        else:
            self.selection.add(segment)
        logger.debug(f"selection: {self.selection}")
        #logger.debug(f"background {self.background}")
        self.update_img(points_changed=True)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.click)
        #self.fig.canvas.mpl_disconnect(self.keypress)
        #self.fig.canvas.mpl_disconnect(self.keyrelease)

    def update_img(self, _=None, points_changed=False):
        self.displayed_img = self.segments.boundaries.copy()

        if points_changed or self.alpha_shape is None:
            logger.debug('update points')
            if self.selection:
                selection_lab = self.segments.lab.loc[list(self.selection)].values
                points = list(self.prior_lab.union(set(map(tuple, selection_lab))))
            else:
                points = list(self.prior_lab)
            if len(points) >= 4:
                logger.debug('update alpha shape')
                logger.debug(f"optimising alpha")
                self.alpha = optimizealpha(points)
                logger.debug(f"optimised alpha: {self.alpha}")

                if self.alpha == 0:
                    # the api for alphashape is a bit strange,
                    # it returns a shapely polygon when alpha is 0
                    # rather than a trimesh object which is returned for other values of alpha
                    # so just calculate the convex hull with trimesh to ensure we get a consistent return value
                    self.alpha_shape = PointCloud(points).convex_hull
                else:
                    self.alpha_shape = alphashape(np.array(points), self.alpha)
            else:
                self.alpha_shape = None

        if self.alpha_shape:
            distance = proximity.signed_distance(self.alpha_shape, self.segments.lab)
            contained = list(self.segments.lab[distance >= -self.delta_slider.val].index)
            logger.debug(f"contained: {contained}")
            # below we reorder the vertices so L is the z axis
            if self.trisurf is not None:
                logger.debug("remove existing alpha shape from plot")
                self.trisurf.remove()
            logger.debug("Draw alpha shape on plot")
            self.trisurf = self.lab_ax.plot_trisurf(*zip(*self.alpha_shape.vertices[:,[1,2,0]]), triangles=self.alpha_shape.faces[:,[1,2,0]], color=(0,1,0,0.5))
            #self.lab_ax.scatter(xs=self.segments.lab['a'], ys=self.segments.lab['b'], zs=self.segments.lab['L'], s=10, c=self.segments.rgb, lw=0)
            self.lab_fig.canvas.draw()
            # todo consider update rather than recalculate alpha shape
        else:
            contained: list = list(self.selection)

        self.displayed_img[np.isin(self.segments.mask, contained)] = [0, 100, 0]
        self.displayed_img[np.isin(self.segments.mask, list(self.selection))] = [0, 0, 100]
        #self.displayed_img[np.isin(self.image.segments.mask, list(self.background))] = [100, 0, 0]
        #self.displayed_img[np.isin(self.image.segments.mask, list(self.background)) & np.isin(self.segments.mask, contained)] = [100, 0, 100]

        self.artist.set_data(self.displayed_img)
        self.fig.canvas.draw()
