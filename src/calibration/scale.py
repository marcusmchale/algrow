import logging
import numpy as np

from typing import List

import wx
from pubsub import pub

import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigureCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar
)
from matplotlib import get_data_path  # we are recycling some matplotlib icons

from ..image_loading import ImageLoaded
from ..options import update_arg

logger = logging.getLogger(__name__)


class ScalePanel(wx.Panel):
    def __init__(self, parent, images: List[ImageLoaded]):
        logger.debug("Launch scale panel")
        super().__init__(parent)

        self.image: ImageLoaded = images[0]
        self.args = self.image.args

        self.fig, self.ax = plt.subplots()
        self.fig.set_dpi(150)
        self.fig.suptitle("Define image scale")
        self.ax.set_title("Draw a line and enter the measured distance")

        self.cv = FigureCanvas(self, -1, self.fig)

        self.ax.imshow(self.image.rgb, picker=True, animated=True)

        self.nav_toolbar = NavigationToolbar(self.cv)
        self.nav_toolbar.Realize()
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.nav_toolbar, 0, wx.ALIGN_CENTER)
        self.sizer.Add(self.cv, 1, wx.EXPAND)
        self.nav_toolbar.update()
        self.SetSizer(self.sizer)

        self.toolbar = wx.ToolBar(self, id=-1, style=wx.TB_HORIZONTAL | wx.TB_TEXT)  # | wx.TB_TEXT)

        # start again button
        self.toolbar.AddTool(
            1,
            "clear",
            wx.Image(str(Path(get_data_path(), "images", "back.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            'Clear line',
            'Clear line and start again',
            None
        )
        self.Bind(wx.EVT_TOOL, self.clear_line, id=1)

        # Line length output
        self.toolbar.AddSeparator()
        self.len_text = wx.TextCtrl(self.toolbar, 2, "", style=wx.TE_READONLY)
        self.toolbar.AddControl(self.len_text)
        self.toolbar.AddControl(wx.StaticText(self.toolbar, label="px"))

        # Distance text input
        self.toolbar.AddSeparator()
        self.dist_text = wx.TextCtrl(self.toolbar, 3, "", style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_TEXT_ENTER, self.set_line_dist, id=3)
        self.toolbar.AddControl(self.dist_text)
        self.toolbar.AddControl(wx.StaticText(self.toolbar, label="mm"))

        # Scale text output
        self.toolbar.AddSeparator()
        self.scale_text = wx.TextCtrl(self.toolbar, 4, "", style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_TEXT_ENTER, self.set_scale, id=4)
        self.toolbar.AddControl(self.scale_text)
        self.toolbar.AddControl(wx.StaticText(self.toolbar, label="px/mm"))

        # add a close button
        self.toolbar.AddSeparator()
        self.close_btn = wx.Button(self.toolbar, 5, "Save and close")
        self.toolbar.AddControl(self.close_btn)
        self.close_btn.Bind(wx.EVT_BUTTON, self.on_exit)

        self.sizer.Add(self.toolbar, 0, wx.ALIGN_LEFT)
        self.toolbar.Realize()

        self.click_cid = self.cv.mpl_connect('button_press_event', self.on_click)
        self.move_cid = None

        self.line_start = None
        self.line_end = None
        self.line = None
        self.line_colour = 'b'

        self.line_px = None
        self.line_dist = None
        self.scale = None

        self.load_scale()

        self.cv.draw_idle()
        self.cv.flush_events()

        self.Bind(wx.EVT_CLOSE, self.on_exit)
        logger.debug("Scale panel loaded")

    def on_exit(self, event):
        logger.debug("Close scale window")
        update_arg(self.args, 'scale', self.scale)
        if self.args.scale is None:
            raise ValueError("Scale has not been set")
        if self.args.debug:
            # print an image with the line and an overlay of the defined measurement
            pass
        #event.Skip()
        pub.sendMessage("enable_btns")
        self.Destroy()

    def clear_line(self, _):
        if self.line is not None:
            self.line.remove()
            self.line = None
            self.cv.draw_idle()
        self.line_start = None
        self.line_end = None

    def set_line_px(self):
        if self.line_start and self.line_end:
            logger.debug(f"start: {self.line_start}, end: {self.line_end}")
            line_length = np.around(np.linalg.norm(np.array(self.line_start) - np.array(self.line_end)), decimals=1)
            self.line_px = line_length
            self.len_text.SetValue(str(self.line_px))
        self.calc_scale()

    def set_line_dist(self, _):
        value = self.dist_text.GetValue()
        try:
            value = float(value)
            self.line_dist = value
            self.dist_text.SetValue(str(value))
        except ValueError:
            logger.debug(f"Distance input could not be coerced to float: {value}")
            self.dist_text.SetValue("")
        self.calc_scale()

    def calc_scale(self):
        if self.line_px and self.line_dist:
            try:
                self.scale = np.around(self.line_px/self.line_dist, decimals=4)
                self.scale_text.SetValue(str(self.scale))
            except ValueError:
                logger.debug("Values for line_px and line_dist most be coerced to float")
            except ZeroDivisionError:
                logger.debug("Measured distance cannot be equal to 0")

    def set_scale(self, _=None):
        try:
            self.scale = float(self.scale_text.GetValue())
        except ValueError:
            logger.debug("Values for scale most be coerced to float")

    def load_scale(self):
        if self.args.scale is not None:
            self.scale = self.args.scale
            self.scale_text.SetValue(str(self.scale))

    def on_click(self, click_event):
        coords = (click_event.xdata, click_event.ydata)
        if self.line_start is None:
            self.line_start = coords
            self.move_cid = self.cv.mpl_connect('motion_notify_event', self.update_line)
        else:
            self.line_end = coords
            self.cv.mpl_disconnect(self.move_cid)
            xs = [self.line_start[0], self.line_end[0]]
            ys = [self.line_start[1], self.line_end[1]]
            if self.line is not None:
                self.line.remove()
            self.line = Line2D(xs, ys, color=self.line_colour)
            self.ax.add_line(self.line)
            self.set_line_px()
            self.cv.draw_idle()

    def update_line(self, mouse_event):
        if self.line_start and self.line_end:
            return
        if self.line is not None:
            self.line.remove()
            self.line = None
        mouse_coords = (mouse_event.xdata, mouse_event.ydata)
        if self.line_start and not self.line_end:
            xs = (self.line_start[0], mouse_coords[0])
            ys = (self.line_start[1], mouse_coords[1])
            self.line = Line2D(xs, ys, color=self.line_colour)
            self.ax.add_line(self.line)
            self.cv.draw_idle()

    def disconnect(self):
        self.cv.mpl_disconnect(self.click_cid)
        if self.move_cid is not None:
            self.cv.mpl_disconnect(self.move_cid)
