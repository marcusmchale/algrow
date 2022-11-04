import logging
from pathlib import Path
from skimage.color import rgb2lab, lab2rgb, rgb2hsv, hsv2rgb
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
        self.lab = rgb2lab(self.rgb)
        self.selection_array = np.zeros_like(self.rgb[:, :, 1], dtype="bool")
        xv, yv = np.meshgrid(np.arange(self.selection_array.shape[1]), np.arange(self.selection_array.shape[0]))
        self.pixel_coords = np.vstack( (xv.flatten(), yv.flatten())).T
        self.mask = None

    @property
    def l(self):
        return self.lab[:, :, 0]

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
        SelectFromImage(ax, self.pixel_coords, self.selection_array)
        plt.show()
        if np.sum(self.selection_array) == 0:
            raise ValueError("No area selected")
        overlay = self.rgb.copy()
        overlay[:, :, 1][self.selection_array] = 0
        self.image_debugger.render_image(overlay, "Overlay of selection")
        return self.selection_array

    def get_thresholds(self):
        if self.mask is None:
            self.mask = self.pick_regions()
        #todo consider using std and mean to create upper and lower bounds
        mean_l = self.l[self.mask].mean().round()
        thresholds = {
            "lower_a": self.a[self.mask].min().round(),
            "lower_b": self.b[self.mask].min().round(),
            "upper_a": self.a[self.mask].max().round(),
            "upper_b": self.b[self.mask].max().round()
        }
        npix = 10
        l = np.array(mean_l)  # constant L, we don't use for thresholding, just use so looks like selected levels
        a = np.linspace(thresholds["lower_a"], thresholds["upper_a"], npix)
        b = np.linspace(thresholds["lower_b"], thresholds["upper_b"], npix)
        colour_plot = np.array(np.meshgrid(l, a, b)).reshape((3, 10, 10))
        colour_plot = np.moveaxis(colour_plot, 0, 2)
        colour_plot_rgb = lab2rgb(colour_plot)
        fig, ax = plt.subplots()
        ax.set_title(f"Target regions colours {thresholds}")
        ax.imshow(colour_plot_rgb)
        plt.show()
        logger.info(f"Thresholds selected: {thresholds}")
        return thresholds


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



