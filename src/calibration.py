import logging
import argparse

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from alphashape import alphashape, optimizealpha
from trimesh import PointCloud, proximity


from typing import List

from .logging import CustomAdapter
from .image_loading import ImageLoaded
from .layout import LayoutDetector, Layout
from .image_segmentation import Segments
from .figurebuilder import FigureBuilder

logger = logging.getLogger(__name__)

# todo make all logging start with the filename so can make sense logs when multithreading
# consider getting getting a new logger per filename


def image_worker(filepath: Path, args: argparse.Namespace):
    adapter = CustomAdapter(logger, {'image_filepath': str(filepath)})
    adapter.debug(f"Loadin file for calibration: {filepath}")
    image = ImageLoaded(filepath, args)
    adapter.debug(f"Detect layout for: {filepath}")
    layout: Layout = LayoutDetector(image).get_layout()
    adapter.debug(f"Segment: {filepath}")
    segments: Segments = Segments(image, layout).get_segments()
    adapter.debug(f"Prepare a segment boundary image for clicker")
    segments.mark_boundaries()
    adapter.debug(f"Done preparing image: {filepath}")
    return segments


class Configurator:
    def __init__(self, image_filepaths: List[Path], args: argparse.Namespace):
        logger.debug("Prepare for calibration")
        self.image_filepaths = image_filepaths
        self.args = args

        # The below are defined by the selector windows and updated when we disconnect it
        self.selection = defaultdict(set)
        self.selected_lab = None
        self.alpha = self.args.alpha
        self.alpha_hull = None
        self.delta = self.args.delta

        # Now prepare some images to open in the selector window
        self.filepath_segments = dict()

        if self.args.processes > 1:
            with ProcessPoolExecutor(max_workers=args.processes, mp_context=get_context('spawn')) as executor:
                future_to_file = {executor.submit(image_worker, filepath, args): filepath for filepath in image_filepaths}
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        segments = future.result()
                        self.filepath_segments[filepath] = segments
                    except Exception as exc:
                        print('%r generated an exception: %s' % (filepath, exc))
        else:
            for filepath in image_filepaths:
                try:
                    segments = image_worker(filepath, args)
                    self.filepath_segments[filepath] = segments
                except Exception as exc:
                    print('%r generated an exception: %s' % (filepath, exc))
        logger.debug(f"images processed: {len(self.filepath_segments.keys())}")

        if len(list(self.filepath_segments.keys())) != len(self.image_filepaths):
            logger.warning("Some images selected for calibration did not complete segmentation")
            self.image_filepaths = list(self.filepath_segments.keys())  # in case some did not segment properly

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

        axialpha = self.fig.add_axes([0.1, 0.025, 0.075, 0.025])
        self.ialpha = TextBox(axialpha, 'Alpha', initial=self.alpha)

        axbalpha = self.fig.add_axes([0.2, 0.025, 0.075, 0.05])
        self.balpha = Button(axbalpha, 'Optimise\nAlpha')

        axidelta = self.fig.add_axes([0.4, 0.025, 0.075, 0.025])
        self.idelta = TextBox(axidelta, "Delta", initial=self.delta)

        axchecks = self.fig.add_axes([0.5, 0.025, 0.1, 0.05])
        self.bcheck = CheckButtons(
            axchecks,
            labels=['selection', 'within'],
            actives=[True, True]
        )

        axprev = self.fig.add_axes([0.6, 0.025, 0.075, 0.05])
        self.bprev = Button(axprev, 'Prev')
        self.bprev.on_clicked(self.prev)

        axnext = self.fig.add_axes([0.7, 0.025, 0.075, 0.05])
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
            self.idelta,
            self.ialpha,
            self.balpha,
            self.bcheck,
            alpha=self.alpha,
            delta=self.delta
        )
        plt.show()
        # when we are done we update values from the final selector window back to self
        self.disconnect_selector()

    def plot_all(self, path):
        if not self.alpha_hull:
            logger.warning("Configuration is incomplete, will not attempt to print summary figure")
            return
        fig = FigureBuilder(path, self.args, 'LAB colourspace (all sampled images) with alpha hull selection')
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
                *self.alpha_hull.vertices[:, [1, 2, 0]]
            ),
            triangles=self.alpha_hull.faces[:, [1, 2, 0]],
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
            self.idelta,
            self.ialpha,
            self.balpha,
            self.bcheck,
            existing_selection=self.selection[self.image_filepaths[self.ind]],
            prior_lab=prior_lab,
            alpha=self.alpha,
            delta=self.delta
        )
        self.fig.canvas.draw()
        logger.debug('new image loaded')

    def disconnect_selector(self):
        prior_image_path = self.selector.segments.image.filepath
        self.selection[prior_image_path] = self.selector.selection
        self.alpha = self.selector.alpha
        self.alpha_hull = self.selector.alpha_hull
        self.delta = self.selector.delta
        selection_indices = [(k, i) for k, v in self.selection.items() for i in v]
        self.selected_lab = self.segments_lab.loc[selection_indices].to_numpy().round(decimals=1)

        colours_string = f'{[",".join([str(j) for j in i]) for i in self.selected_lab]}'.replace("'", '"')
        logger.info(f'Colours selected: {colours_string}')
        self.selector.disconnect()

    def disconnect(self):
        raise NotImplementedError
        #todo should probably clean up the connections to checkbox and textboxes


class ClickSelect:

    def __init__(self, segments, fig, artist, delta_text, alpha_text, alpha_button, checkbuttons, existing_selection=None, prior_lab=None, alpha=None, delta=None):

        self.segments = segments
        self.displayed_img = self.segments.boundaries.copy()

        self.fig = fig
        self.artist = artist
        self.delta_text = delta_text
        self.alpha_text = alpha_text
        self.checkbuttons = checkbuttons

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

        self.alpha_hull = None
        self.alpha = alpha
        self.delta = delta
        self.trisurf = None

        self.click = self.fig.canvas.mpl_connect('pick_event', self.onclick)
        # self.shift_held = False
        # self.keypress = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        # self.keyrelease = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.delta_text.on_submit(self.set_delta)
        self.alpha_text.on_submit(self.set_alpha)
        alpha_button.on_clicked(self.optimise_alpha)

        self.highlight_selection, self.highlight_within = self.checkbuttons.get_status()
        self.checkbuttons.on_clicked(self.update_highlighting)

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
        self.update_img(recalculate_alpha_hull=True)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.click)
        #self.fig.canvas.mpl_disconnect(self.keypress)
        #self.fig.canvas.mpl_disconnect(self.keyrelease)

    def set_alpha(self, _):
        try:
            self.alpha = float(self.alpha_text.text)
        except ValueError:
            logger.debug("Value for alpha text input could not be coerced to float")
            self.alpha_text.set_val(self.alpha)
        self.update_img(recalculate_alpha_hull=True)

    def set_delta(self, _):
        try:
            self.delta = float(self.delta_text.text)
        except ValueError:
            logger.debug("Value for delta text input could not be coerced to float")
            self.alpha_text.set_val(self.delta)
        self.update_img()

    def optimise_alpha(self, _=None):
        points = self.get_points()
        if len(points) >= 4:
            logger.debug(f"optimising alpha")
            self.alpha = round(optimizealpha(points), ndigits=3)
            logger.debug(f"optimised alpha: {self.alpha}")
        else:
            self.alpha = None
            self.alpha_hull = None
            logger.debug(f"Insufficient points to construct polygon")
        self.alpha_text.set_val(self.alpha)
        self.update_img(recalculate_alpha_hull=True)

    def update_highlighting(self, _=None):
        self.highlight_selection, self.highlight_within = self.checkbuttons.get_status()
        self.update_img()

    def get_points(self):
        if self.selection:
            selection_lab = self.segments.lab.loc[list(self.selection)].values
            points = list(self.prior_lab.union(set(map(tuple, selection_lab))))
        else:
            points = list(self.prior_lab)
        return points

    def update_img(self, _=None, recalculate_alpha_hull=False):
        self.displayed_img = self.segments.boundaries.copy()
        points = self.get_points()
        if (recalculate_alpha_hull or self.alpha_hull is None) and len(points) >= 4:
            if self.alpha is None or self.alpha == 0:
                # the api for alphashape is a bit strange,
                # it returns a shapely polygon when alpha is 0
                # rather than a trimesh object which is returned for other values of alpha
                # so just calculate the convex hull with trimesh to ensure we get a consistent return value
                self.alpha_hull = PointCloud(points).convex_hull
            else:
                self.alpha_hull = alphashape(np.array(points), self.alpha)

        if self.alpha_hull is not None:
            distance = proximity.signed_distance(self.alpha_hull, self.segments.lab)
            contained = list(self.segments.lab[distance >= -self.delta].index)
            logger.debug(f"contained: {contained}")
            # below we reorder the vertices so L is the z axis
            if self.trisurf is not None:
                logger.debug("remove existing alpha hull from plot")
                self.trisurf.remove()
            logger.debug("Draw alpha hull on plot")
            self.trisurf = self.lab_ax.plot_trisurf(
                *zip(*self.alpha_hull.vertices[:, [1, 2, 0]]),
                triangles=self.alpha_hull.faces[:, [1, 2, 0]],
                color=(0, 1, 0, 0.5)
            )
            self.lab_fig.canvas.draw()
        else:
            contained: list = list(self.selection)
            if self.trisurf is not None:
                self.trisurf.remove()
            self.lab_fig.canvas.draw()

        if self.highlight_within:
            self.displayed_img[np.isin(self.segments.mask, list(set(contained)-self.selection))] = [0, 100, 0]
        if self.highlight_selection:
            self.displayed_img[np.isin(self.segments.mask, list(self.selection))] = [0, 0, 100]

        self.artist.set_data(self.displayed_img)
        self.fig.canvas.draw()
