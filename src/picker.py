import logging
from pathlib import Path
from skimage.color import rgb2lab, lab2rgb
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import matplotlib.pyplot as plt
import numpy as np
from .debugger import Debugger

logger = logging.getLogger(__name__)


class Picker:
    def __init__(self, image_path):
        self.image_debugger = Debugger(image_path)
        logger.debug("Load selected image for colour picking")
        self.rgb = plt.imread(str(image_path)).copy()
        self.lab = rgb2lab(self.rgb / 255)
        self.selection_array = np.zeros_like(self.rgb[:, :, 1], dtype="bool")
        xv, yv = np.meshgrid(np.arange(self.selection_array.shape[1]), np.arange(self.selection_array.shape[0]))
        self.pixel_coords = np.vstack( (xv.flatten(), yv.flatten())).T


    @property
    def a(self):
        return self.lab[:, :, 1]

    @property
    def b(self):
        return self.lab[:, :, 2]

    def pick_regions(self):
        fig, ax = plt.subplots()
        ax.set_title("Select representative regions (draw circles around some)")
        ax.imshow(self.rgb)
        selector = SelectFromImage(ax, self.pixel_coords, self.selection_array)
        plt.show()
        if np.sum(self.selection_array) == 0:
            raise ValueError("No area selected")
        fig, ax = plt.subplots()
        ax.imshow(self.selection_array)
        plt.show()
        overlay = self.rgb.copy()
        overlay[:, :, 1][self.selection_array] = 0
        fig, ax = plt.subplots()
        ax.imshow(overlay)
        plt.show()
        return self.selection_array

    def get_mean_colour(self):
        mask = self.pick_regions()
        mean_colour = self.lab[mask].mean(0)
        picked_colour_array = np.full((10, 10, 3), lab2rgb(mean_colour) * 255, dtype="uint8")
        fig, ax = plt.subplots()
        ax.imshow(picked_colour_array)
        plt.show()
        mean_colour_dict = {
            "target_l": mean_colour[0],
            "target_a": mean_colour[1],
            "target_b": mean_colour[2]
        }
        logger.info(mean_colour_dict)
        return mean_colour_dict


class SelectFromImage:
    def __init__(self, ax, pixel_coords, selection_array):
        self.pixel_coords = pixel_coords
        self.selection_array = selection_array
        self.lasso = LassoSelector(ax, onselect=self.onselect)

    def update_array(self, indices):
        lin = np.arange(self.selection_array.size)
        flat_array = self.selection_array.reshape(-1) # this is a view so updates the selection_array
        flat_array[lin[indices]] = True

    def onselect(self, verts):
        path = Path(verts)
        ind = np.nonzero(path.contains_points(self.pixel_coords, radius=1))
        self.update_array(ind)

    def disconnect(self):
        self.lasso.disconnect_events()



