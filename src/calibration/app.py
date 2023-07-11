import logging

from typing import List
from pathlib import Path

import wx
from pubsub import pub

from ..image_loading import ImageLoaded
from ..image_segmentation import Segmentor

from .scale import ScalePanel
from .hull import HullPanel
from .lasso import LassoPanel
from .waiting import WaitPanel

logger = logging.getLogger(__name__)


## this snippet is useful to catch exceptions from wx.Frame
#import sys
#import traceback
#
#def excepthook(type, value, tb):
#    message = 'Uncaught exception:\n'
#    message += ''.join(traceback.format_exception(type, value, tb))
#    logger.debug(message)
#
#sys.excepthook = excepthook


class Calibrator(wx.App):
    # overriding init to pass in the arguments from CLI/configuration file(s)
    def __init__(self, images: List[ImageLoaded], **kwargs):
        logger.debug("Start Calibration App")
        self.images = images
        self.args = images[0].args
        self.frame = None

        super().__init__(self, **kwargs)

    def OnInit(self):
        self.frame = TopFrame(self.images)
        logger.debug("Start configuration GUI")
        self.frame.Show(True)
        return True

    def OnExit(self):
        # Output a file summarising the calibration values: selected colours, alpha and delta values
        logger.debug("Write out calibration parameters")
        with open(Path(self.args.out_dir, "calibration.conf"), 'w') as text_file:
            circle_colour_string = f"\"{','.join([str(i) for i in self.args.circle_colour])}\""
            hull_vertices_string = f'{[",".join([str(j) for j in i]) for i in self.args.hull_vertices]}'.replace(
                "'", '"')
            text_file.write(f"circle_colour = {circle_colour_string}\n")
            text_file.write(f"hull_vertices = {hull_vertices_string}\n")
            text_file.write(f"alpha = {self.args.alpha}\n")
            text_file.write(f"delta = {self.args.delta}\n")
            text_file.write(f"scale = {self.args.scale}\n")


class TopFrame(wx.Frame):
    def __init__(self, images: List[ImageLoaded]):
        super().__init__(None, title="AlGrow Calibration", size=(2000, 1000))
        self.images = images
        self.args = self.images[0].args

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.scaler_btn = wx.Button(self, 1, "Set scale", size=(100, 40))
        self.scaler_btn.Bind(wx.EVT_BUTTON, self.launch_scaler)
        self.btn_sizer.Add(self.scaler_btn, wx.ALIGN_CENTER)

        self.circle_colour_btn = wx.Button(self, 2, "Circle colour", size=(100, 40))
        self.circle_colour_btn.Bind(wx.EVT_BUTTON, self.launch_circle_colour)
        self.btn_sizer.Add(self.circle_colour_btn, wx.ALIGN_CENTER)

        self.hull_btn = wx.Button(self, 3, "Target hull", size=(100, 40))
        self.hull_btn.Bind(wx.EVT_BUTTON, self.launch_hull)
        self.btn_sizer.Add(self.hull_btn, wx.ALIGN_CENTER)

        self.cont_btn = wx.Button(self, 3, "Continue", size=(100, 40))
        self.cont_btn.Bind(wx.EVT_BUTTON, self.on_exit)
        self.btn_sizer.Add(self.cont_btn, wx.ALIGN_CENTER)

        self.sizer.Add(self.btn_sizer)

        pub.subscribe(self.enable_btns, 'enable_btns')

        self.disable_btns()
        self.enable_btns()
        self.update_btn_colour()

        self.SetSizer(self.sizer)
        self.Show()

        self.segmentor = None

    def enable_btns(self):
        buttons = [self.scaler_btn, self.circle_colour_btn]
        if self.args.circle_colour is not None:
            buttons.append(self.hull_btn)
        if self.done():
            buttons.append(self.cont_btn)
        for b in buttons:
            b.Enable()
        self.update_btn_colour()

    def disable_btns(self):
        for b in [self.scaler_btn, self.circle_colour_btn, self.hull_btn, self.cont_btn]:
            b.Disable()

    def update_btn_colour(self):
        if self.args.scale is not None:
            self.scaler_btn.SetBackgroundColour((0, 255, 0))
        else:
            self.scaler_btn.SetBackgroundColour(wx.NullColour)
        if self.args.circle_colour is not None:
            self.circle_colour_btn.SetBackgroundColour((0, 255, 0))
        else:
            self.circle_colour_btn.SetBackgroundColour(wx.NullColour)
        if self.args.hull_vertices is not None and len(self.args.hull_vertices >= 4):
            self.hull_btn.SetBackgroundColour((0, 255, 0))
        else:
            self.hull_btn.SetBackgroundColour(wx.NullColour)

        if self.done():
            self.cont_btn.SetBackgroundColour((0, 255, 0))
        else:
            self.cont_btn.SetBackgroundColour(wx.NullColour)

    def done(self):
        return all([
            self.args.scale is not None,
            self.args.circle_colour is not None,
            (self.args.hull_vertices is not None and len(self.args.hull_vertices >= 4))
        ])

    def display_panel(self, panel):
        self.disable_btns()
        self.sizer.Add(panel)
        self.SetSizer(self.sizer)
        self.Layout()
        self.Update()

    def launch_scaler(self, _=None):
        panel = ScalePanel(self, self.images)
        self.display_panel(panel)

    def launch_circle_colour(self, _=None):
        panel = LassoPanel(self, self.images)
        self.display_panel(panel)

    def launch_hull(self, _=None):
        logger.debug("launch wait panel")
        wait_panel = WaitPanel(self)
        self.display_panel(wait_panel)
        wx.Yield()
        if self.segmentor is None:
            self.segmentor = Segmentor(self.images)
            self.segmentor.run()
        logger.debug("launch hull panel")
        wait_panel.Destroy()
        hull_panel = HullPanel(self, self.segmentor)
        self.display_panel(hull_panel)

    def on_exit(self, _=None):
        if not self.done():
            raise ValueError("Not all required configuration values are set")
        self.Close()










