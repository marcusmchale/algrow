import logging

import numpy as np

import wx
from pubsub import pub
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigureCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar
)

from typing import List

import matplotlib.pyplot as plt
from matplotlib.path import Path as MPLPath
from matplotlib.widgets import LassoSelector
from skimage.measure import regionprops
from skimage.color import lab2rgb

from ..options import update_arg

from ..image_loading import ImageLoaded
from ..figurebuilder import FigureBuilder
from ..image_segmentation import Segments


logger = logging.getLogger(__name__)


class LassoPanel(wx.Panel):  # todo: add a panel showing current selected circle colour
    def __init__(self, parent, images: List[ImageLoaded]):
        logger.debug("Launch lasso panel")
        super().__init__(parent)

        self.image: ImageLoaded = images[0]
        self.args = self.image.args

        self.fig, self.ax = plt.subplots()
        self.fig.set_dpi(150)
        self.ax.set_title("Click and drag to lasso colour")
        self.cv = FigureCanvas(self, -1, self.fig)

        self.ax.imshow(self.image.rgb)
        self.selector = LassoSelect(self.cv, self.ax, self.image.rgb.shape[0:2], self.update_colour)

        self.nav_toolbar = NavigationToolbar(self.cv)
        self.nav_toolbar.Realize()
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.nav_toolbar, 0, wx.ALIGN_CENTER)
        self.sizer.Add(self.cv, 1, wx.EXPAND)
        self.nav_toolbar.update()
        self.SetSizer(self.sizer)

        # add a clear selection button
        self.toolbar = wx.ToolBar(self, id=-1, style=wx.TB_HORIZONTAL | wx.TB_TEXT)  # | wx.TB_TEXT)
        self.clear_btn = wx.Button(self.toolbar, 1, "Clear selection")
        self.toolbar.AddControl(self.clear_btn)
        self.clear_btn.Bind(wx.EVT_BUTTON, self.clear_selection)

        # add a close button
        self.close_btn = wx.Button(self.toolbar, 2, "Save and close")
        self.toolbar.AddControl(self.close_btn)
        self.close_btn.Bind(wx.EVT_BUTTON, self.on_exit)

        # add indication of selected target colour beside the save and close button
        self.selected_btn = wx.Button(self.toolbar, 3, "    ")
        logger.debug(f"circle colour: {self.args.circle_colour}")
        self.toolbar.AddControl(self.selected_btn)
        if self.args.circle_colour is not None:
            self.selected_btn.SetBackgroundColour(tuple(np.array(self.args.circle_colour).astype(int)))

        self.sizer.Add(self.toolbar, 0, wx.ALIGN_LEFT)
        self.toolbar.Realize()

        self.cv.draw_idle()
        self.cv.flush_events()

        self.Bind(wx.EVT_CLOSE, self.on_exit)
        logger.debug("Lasso panel loaded")

    def update_colour(self):  # tuple in rgb255:
        colour = self.get_colour()
        if colour is None:
            if self.args.circle_colour:
                self.selected_btn.SetBackgroundColour(tuple(np.array(self.args.circle_colour).astype(int)))
            else:
                self.selected_btn.SetBackgroundColour(wx.NullColour)
        else:
            colour_rgb = tuple((lab2rgb(colour) * 255).astype(int))
            logger.debug(colour_rgb)
            self.selected_btn.SetBackgroundColour(colour_rgb)

    def get_colour(self):
        logger.debug(np.sum(self.selector.selection_array))
        if np.sum(self.selector.selection_array) > 0:
            regions = regionprops(
                self.selector.selection_array,
                intensity_image=self.image.lab,
                extra_properties=(Segments.median_intensity,)
            )
            colours = [np.around(r.median_intensity, decimals=1) for r in regions]
            colours_string = f'{[",".join([str(j) for j in i]) for i in colours]}'.replace("'", '"')
            logger.info(f'Colours selected: {colours_string}')
            return tuple(np.round(np.median(colours, axis=0), decimals=1))
        else:
            return None

    def on_exit(self, _=None):
        if np.sum(self.selector.selection_array) == 0:
            colour = self.args.circle_colour
            if self.args.circle_colour is None:
                raise ValueError("No area selected")
            else:
                logger.warning("No area selected - using preconfigured value")
        else:
            colour = self.get_colour()
        self.selector.disconnect()
        update_arg(self.args, "circle_colour", colour)

        logger.debug(f"Plotting debug image for circle colour: {self.args.circle_colour}")
        fig = FigureBuilder(".", self.args, "Circle colour")
        fig.plot_colours([self.args.circle_colour])
        fig.print()

        pub.sendMessage("enable_btns")
        self.Destroy()

    def clear_selection(self, _=None):
        self.selector.empty_array()
        self.update_colour()
        self.Update()


class LassoSelect:
    def __init__(self, cv, ax, shape, colour_callback):
        self.counter = 0
        self.cv = cv
        self.shape = shape

        self.selection_array = np.zeros(self.shape, dtype="int")
        xv, yv = np.meshgrid(np.arange(self.selection_array.shape[1]), np.arange(self.selection_array.shape[0]))
        self.pixel_coords = np.vstack((xv.flatten(), yv.flatten())).T

        self.lasso = LassoSelector(ax, onselect=self.onselect, useblit=False)
        self.colour_callback = colour_callback
        # todo work out how to fix blitting to speed up the above drawing -
        #   currently doesn't update the wx window unless done manually by resize etc.

    def empty_array(self):
        self.selection_array = np.zeros(self.shape, dtype="int")
        xv, yv = np.meshgrid(np.arange(self.selection_array.shape[1]), np.arange(self.selection_array.shape[0]))
        self.pixel_coords = np.vstack((xv.flatten(), yv.flatten())).T

    def update_array(self, indices):
        self.counter += 1
        lin = np.arange(self.selection_array.size)
        flat_array = self.selection_array.reshape(-1)  # this is a view so updates the selection_array
        flat_array[lin[indices]] = self.counter

    def onselect(self, verts):
        logger.debug("select done")
        path = MPLPath(verts)
        ind = np.nonzero(path.contains_points(self.pixel_coords, radius=1))
        self.update_array(ind)
        self.colour_callback()
        self.cv.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
