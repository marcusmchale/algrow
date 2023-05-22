import logging
from pathlib import Path
from skimage.segmentation import mark_boundaries
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .figurebuilder import FigureBuilder
from skimage.color import deltaE_cie76

logger = logging.getLogger(__name__)


class Picker:
    def __init__(self, rgb, lab, filepath, activity: str, args):
        self.args = args
        self.filepath = filepath
        logger.debug(f"Load selected image for {activity}")
        self.rgb = rgb
        self.lab = lab
        self.selection_array = np.zeros_like(self.rgb[:, :, 1], dtype="int")
        xv, yv = np.meshgrid(np.arange(self.selection_array.shape[1]), np.arange(self.selection_array.shape[0]))
        self.pixel_coords = np.vstack( (xv.flatten(), yv.flatten())).T
        self.activity = activity

    def pick_lasso(self):
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(f"Selection for {self.activity}")
        ax.set_title("Select representative regions (click and draw a lasso)")
        ax.imshow(self.rgb)
        selector = LassoSelectFromImage(ax, self.pixel_coords, self.selection_array)
        plt.show()
        selector.disconnect()
        if np.sum(self.selection_array) == 0:
            raise ValueError("No area selected")
        if self.args.debug:
            overlay = self.rgb.copy()
            overlay[self.selection_array != 0] = 255 - overlay[self.selection_array != 0] # invert colour
            fig = FigureBuilder(self.filepath, f"Selection for {self.activity}")
            fig.add_image(overlay)
            fig.print()

    def get_colours(self):
        if len(np.unique(self.selection_array)) == 1:
            self.pick_lasso()
        colours = list()
        for c in np.unique(self.selection_array):
            if c == 0:
                continue
            lab = np.around(np.median(self.lab[self.selection_array == c], axis=0), decimals = 1)
            colours.append(lab)
        colours_string = f'{[",".join([str(j) for j in i]) for i in colours]}'.replace("'", '"')
        logger.info(f'Colours selected: {colours_string}')
        return colours


    def get_reference_colours(self, segments, segment_colours, dist, reference_label):
        selection = set()
        fig, ax = plt.subplots()
        boundary_img = mark_boundaries(self.rgb, segments, color=(255, 0, 255))
        ax.set_title(label=f'Click to select {reference_label} regions')
        ax.imshow(boundary_img, picker=True)

        def get_segments_within_dist():
            reference_colours = np.array(segment_colours.loc[list(selection)])
            distances = pd.DataFrame(
                data=np.array([deltaE_cie76(segment_colours, r) for r in reference_colours]).transpose(),
                index=segment_colours.index,
                columns=[str(r) for r in reference_colours]
            )
            return distances.index[distances.min(axis=1) < dist].tolist()

        def update_img(img):
            similar = get_segments_within_dist()
            selection_mask = np.isin(segments, list(selection))
            similar_mask = np.isin(segments, similar)
            img[similar_mask] = img[similar_mask] * 0.8 + np.array([255, 0, 0]) * 0.2
            img[selection_mask] = np.array([255, 0, 0])

        def onclick(event):
            x = event.mouseevent.xdata.astype(int)
            y = event.mouseevent.ydata.astype(int)
            segment = segments[y,x]
            if segment == 0:
                return
            elif segment in selection:
                selection.remove(segment)
            else:
                selection.add(segment)
            if selection:
                img = boundary_img.copy()
                update_img(img)
            else:
                img = boundary_img.copy()
            artist = event.artist
            artist.set_data(img)
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', onclick)
        plt.show()

        if self.args.debug:
            fig = FigureBuilder(self.filepath, f'Pick {reference_label} regions')
            update_img(boundary_img)
            fig.add_image(boundary_img, label=f'Selected {reference_label} regions', picker=True)
            fig.print()

        return  np.around(np.array(segment_colours.loc[list(selection)]), decimals=1)



class LassoSelectFromImage:
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


class ClickSelectFromImage:

    def __init__(self, ax, pixel_coords, selection_array):
        self.counter = 0
        self.pixel_coords = pixel_coords
