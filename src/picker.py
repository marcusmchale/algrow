import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPLPath
from matplotlib.widgets import LassoSelector
from skimage.measure import regionprops

from .image_loading import ImageLoaded
from .figurebuilder import FigureBuilder
from .image_segmentation import Segments



logger = logging.getLogger(__name__)


class Picker:
    def __init__(self, image, activity: str):
        self.image: ImageLoaded = image
        self.args = image.args
        self.activity = activity

        self.selection_array = np.zeros_like(self.image.rgb[:, :, 1], dtype="int")
        xv, yv = np.meshgrid(np.arange(self.selection_array.shape[1]), np.arange(self.selection_array.shape[0]))
        self.pixel_coords = np.vstack((xv.flatten(), yv.flatten())).T

    def lasso_colours(self):
        if len(np.unique(self.selection_array)) == 1:
            self.pick_lasso()
        regions = regionprops(
                self.selection_array,
                intensity_image=self.image.lab,
                extra_properties=(Segments.median_intensity,)
        )
        colours = [np.around(r.median_intensity, decimals=1) for r in regions]
        colours_string = f'{[",".join([str(j) for j in i]) for i in colours]}'.replace("'", '"')
        logger.info(f'Colours selected: {colours_string}')
        return colours

    def pick_lasso(self):
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(f"Selection for {self.activity}")
        ax.set_title("Select representative regions (click and draw a lasso)")
        ax.imshow(self.image.rgb)
        selector = LassoSelect(ax, self.pixel_coords, self.selection_array)
        plt.show()
        selector.disconnect()
        if np.sum(self.selection_array) == 0:
            raise ValueError("No area selected")
        if self.args.debug:
            overlay = self.image.rgb.copy()
            overlay[self.selection_array != 0] = 255 - overlay[self.selection_array != 0]  # invert colour
            fig = FigureBuilder(self.image.filepath, self.args, f"Selection for {self.activity}")
            fig.add_image(overlay)
            fig.print()


class LassoSelect:
    def __init__(self, ax, pixel_coords, selection_array):
        self.counter = 0
        self.pixel_coords = pixel_coords
        self.selection_array = selection_array
        self.lasso = LassoSelector(ax, onselect=self.onselect)

    def update_array(self, indices):
        self.counter += 1
        lin = np.arange(self.selection_array.size)
        flat_array = self.selection_array.reshape(-1)  # this is a view so updates the selection_array
        flat_array[lin[indices]] = self.counter

    def onselect(self, verts):
        path = MPLPath(verts)
        ind = np.nonzero(path.contains_points(self.pixel_coords, radius=1))
        self.update_array(ind)

    def disconnect(self):
        self.lasso.disconnect_events()
