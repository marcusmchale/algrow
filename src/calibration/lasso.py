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

        self.selection_array = np.zeros_like(self.image.rgb[:, :, 1], dtype="int")
        xv, yv = np.meshgrid(np.arange(self.selection_array.shape[1]), np.arange(self.selection_array.shape[0]))
        self.pixel_coords = np.vstack((xv.flatten(), yv.flatten())).T

        self.fig, self.ax = plt.subplots()
        self.fig.set_dpi(150)
        self.ax.set_title("Click and drag to lasso colour")
        self.cv = FigureCanvas(self, -1, self.fig)

        self.ax.imshow(self.image.rgb)
        self.selector = LassoSelect(self.ax, self.pixel_coords, self.selection_array)

        self.nav_toolbar = NavigationToolbar(self.cv)
        self.nav_toolbar.Realize()
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.nav_toolbar, 0, wx.ALIGN_CENTER)
        self.sizer.Add(self.cv, 1, wx.EXPAND)
        self.nav_toolbar.update()
        self.SetSizer(self.sizer)

        # add a close button
        self.toolbar = wx.ToolBar(self, id=-1, style=wx.TB_HORIZONTAL | wx.TB_TEXT)  # | wx.TB_TEXT)
        self.close_btn = wx.Button(self.toolbar, 1, "Save and close")
        self.toolbar.AddControl(self.close_btn)
        self.close_btn.Bind(wx.EVT_BUTTON, self.on_exit)
        self.sizer.Add(self.toolbar, 0, wx.ALIGN_LEFT)
        self.toolbar.Realize()

        self.cv.draw_idle()
        self.cv.flush_events()

        self.Bind(wx.EVT_CLOSE, self.on_exit)
        logger.debug("Lasso panel loaded")

    def on_exit(self, _=None):
        if np.sum(self.selection_array) == 0:
            raise ValueError("No area selected")
        self.selector.disconnect()
        if self.args.debug:
            overlay = self.image.rgb.copy()
            overlay[self.selection_array != 0] = 255 - overlay[self.selection_array != 0]  # invert colour
            fig = FigureBuilder(self.image.filepath, self.args, f"Selection for circle colour")
            fig.add_image(overlay)
            fig.print()
        regions = regionprops(
                self.selection_array,
                intensity_image=self.image.lab,
                extra_properties=(Segments.median_intensity,)
        )
        colours = [np.around(r.median_intensity, decimals=1) for r in regions]
        colours_string = f'{[",".join([str(j) for j in i]) for i in colours]}'.replace("'", '"')
        logger.info(f'Colours selected: {colours_string}')
        colour = tuple(np.round(np.median(colours, axis=0), decimals=1))
        update_arg(self.args, "circle_colour", colour)
        pub.sendMessage("enable_btns")
        self.Destroy()


class LassoSelect:
    def __init__(self, ax, pixel_coords, selection_array):
        self.counter = 0
        self.pixel_coords = pixel_coords
        self.selection_array = selection_array
        self.lasso = LassoSelector(ax, onselect=self.onselect, useblit=False)
        # todo work out how to fix blitting to speed up the above drawing -
        #   currently doesn't update the wx window unless done manually by resize etc.

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
