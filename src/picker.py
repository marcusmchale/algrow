import logging
from pathlib import Path
from skimage.color import rgb2lab
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import matplotlib.pyplot as plt
import numpy as np
from .figurebuilder import FigureBuilder
from .options import options

logger = logging.getLogger(__name__)
args = options().parse_args()


class Picker:
    def __init__(self, image_path, activity: str):
        self.image_path = image_path
        logger.debug(f"Load selected image for {activity}")
        self.rgb = plt.imread(str(image_path)).copy()
        self.lab = rgb2lab(self.rgb)
        self.selection_array = np.zeros_like(self.rgb[:, :, 1], dtype="int")
        xv, yv = np.meshgrid(np.arange(self.selection_array.shape[1]), np.arange(self.selection_array.shape[0]))
        self.pixel_coords = np.vstack( (xv.flatten(), yv.flatten())).T
        self.mask = None
        self.activity = activity

    def pick_regions(self):
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(f"Selection for {self.activity}")
        ax.set_title("Select representative regions (click and draw)")
        ax.imshow(self.rgb)
        selector = SelectFromImage(ax, self.pixel_coords, self.selection_array)
        plt.show()
        selector.disconnect()
        if np.sum(self.selection_array) == 0:
            raise ValueError("No area selected")
        if args.debug:
            overlay = self.rgb.copy()
            overlay[self.selection_array != 0] = 255 - overlay[self.selection_array != 0] # invert colour
            fig = FigureBuilder(self.image_path, f"Selection for {self.activity}")
            fig.add_image(overlay)
            fig.print()

    def get_colours(self):
        if len(np.unique(self.selection_array)) == 1:
            self.pick_regions()
        colours = list()
        for c in np.unique(self.selection_array):
            if c == 0:
                continue
            lab = np.around(np.median(self.lab[self.selection_array == c], axis=0), decimals = 1)
            colours.append(lab)
        colours_string = f'{[",".join([str(j) for j in i]) for i in colours]}'.replace("'", '"')
        logger.info(f'Colours selected: {colours_string}')
        return colours


class SelectFromImage:
    def __init__(self, ax, pixel_coords, selection_array):
        self.counter = 0
        self.pixel_coords = pixel_coords
        self.selection_array = selection_array
        self.lasso = LassoSelector(ax, onselect=self.onselect)

    def update_array(self, indices):
        self.counter += 1
        lin = np.arange(self.selection_array.size)
        flat_array = self.selection_array.reshape(-1) # this is a view so updates the selection_array
        flat_array[lin[indices]] = self.counter

    def onselect(self, verts):
        path = Path(verts)
        ind = np.nonzero(path.contains_points(self.pixel_coords, radius=1))
        self.update_array(ind)

    def disconnect(self):
        self.lasso.disconnect_events()



