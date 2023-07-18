import logging

from typing import List
from pathlib import Path

import wx
from pubsub import pub

from ..image_loading import ImageLoaded
from ..image_segmentation import Segmentor

from .measure_layout import LayoutPanel
from .measure_scale import ScalePanel
from .hull import HullPanel
from .lasso import LassoPanel
from .waiting import WaitPanel

logger = logging.getLogger(__name__)


# this snippet is useful to catch exceptions from wx.Frame
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
        logger.info("Start configuration GUI")
        self.frame.Show(True)
        return True

    def OnExit(self):
        # Output a file summarising the calibration values: selected colours, alpha and delta values
        logger.info("Write out calibration parameters")
        with open(Path(self.args.out_dir, "calibration.conf"), 'w') as text_file:
            text_file.write(f"[Colour parameters]\n")
            circle_colour_string = f"\"{','.join([str(i) for i in self.args.circle_colour])}\""
            hull_vertices_string = f'{[",".join([str(j) for j in i]) for i in self.args.hull_vertices]}'.replace(
                "'", '"')
            text_file.write(f"circle_colour = {circle_colour_string}\n")
            text_file.write(f"hull_vertices = {hull_vertices_string}\n")
            text_file.write(f"alpha = {self.args.alpha}\n")
            text_file.write(f"delta = {self.args.delta}\n")
            text_file.write(f"scale = {self.args.scale}\n")
            text_file.write(f"[Layout parameters]\n")
            text_file.write(f"circle_diameter = {self.args.circle_diameter}\n")
            text_file.write(f"circle_expansion = {self.args.circle_expansion}\n")
            text_file.write(f"plate_circle_separation = {self.args.plate_circle_separation}\n")
            text_file.write(f"plate_width = {self.args.plate_width}\n")
            text_file.write(f"circles_per_plate = {self.args.circles_per_plate}\n")
            text_file.write(f"n_plates = {self.args.n_plates}\n")
            text_file.write(f"plates_cols_first = {self.args.plates_cols_first}\n")
            text_file.write(f"plates_bottom_top = {self.args.plates_bottom_top}\n")
            text_file.write(f"plates_right_left = {self.args.plates_right_left}\n")
            text_file.write(f"circles_cols_first = {self.args.circles_cols_first}\n")
            text_file.write(f"circles_bottom_top = {self.args.circles_bottom_top}\n")
            text_file.write(f"circles_right_left = {self.args.circles_right_left}\n")
            logger.debug("Finished writing to calibration file")
        return int(1)


class TopFrame(wx.Frame):
    def __init__(self, images: List[ImageLoaded]):
        super().__init__(None, title="AlGrow Calibration", size=(2000, 1000))
        self.images = images
        self.image = self.images[len(self.images)//2]
        self.args = self.image.args

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.scaler_btn = wx.Button(self, -1, "Set scale", size=(100, 40))
        self.scaler_btn.Bind(wx.EVT_BUTTON, self.launch_scaler)
        self.btn_sizer.Add(self.scaler_btn, wx.ALIGN_CENTER)

        self.circle_colour_btn = wx.Button(self, -1, "Circle colour", size=(100, 40))
        self.circle_colour_btn.Bind(wx.EVT_BUTTON, self.launch_circle_colour)
        self.btn_sizer.Add(self.circle_colour_btn, wx.ALIGN_CENTER)

        self.layout_btn = wx.Button(self, -1, "Define layout", size=(100, 40))
        self.layout_btn.Bind(wx.EVT_BUTTON, self.launch_layout)
        self.btn_sizer.Add(self.layout_btn, wx.ALIGN_CENTER)

        self.hull_btn = wx.Button(self, -1, "Target hull", size=(100, 40))
        self.hull_btn.Bind(wx.EVT_BUTTON, self.launch_hull)
        self.btn_sizer.Add(self.hull_btn, wx.ALIGN_CENTER)

        self.cont_btn = wx.Button(self, -1, "Continue", size=(100, 40))
        self.cont_btn.Bind(wx.EVT_BUTTON, self.on_exit)
        self.Bind(wx.EVT_CLOSE, self.on_exit)
        self.btn_sizer.Add(self.cont_btn, wx.ALIGN_CENTER)

        self.sizer.Add(self.btn_sizer)

        pub.subscribe(self.enable_btns, 'enable_btns')

        self.disable_btns()
        self.enable_btns()

        self.SetSizer(self.sizer)
        self.Show()

        self.segmentor = None

    def enable_btns(self):
        active_buttons = [self.scaler_btn, self.circle_colour_btn]
        if self.args.circle_colour is not None:
            active_buttons.append(self.layout_btn)
        if self.layout_done():
            active_buttons.append(self.hull_btn)
        if self.done():
            active_buttons.append(self.cont_btn)
        for b in active_buttons:
            b.Enable()
        self.set_btn_colours()

    def set_btn_colours(self):
        complete_buttons = []
        if self.args.scale is not None:
            complete_buttons.append(self.scaler_btn)
        if self.args.circle_colour is not None:
            complete_buttons.append(self.circle_colour_btn)
        if self.layout_done():
            complete_buttons.append(self.layout_btn)
        logger.debug(f"Hull vertices: {self.args.hull_vertices}")
        if self.args.hull_vertices is not None and len(self.args.hull_vertices) >= 4:
            complete_buttons.append(self.hull_btn)
        if self.done():
            complete_buttons.append(self.cont_btn)
        for b in complete_buttons:
            b.SetBackgroundColour((0, 255, 0))

    def disable_btns(self):
        for b in [self.scaler_btn, self.circle_colour_btn, self.layout_btn, self.hull_btn, self.cont_btn]:
            b.SetBackgroundColour(wx.NullColour)
            b.Disable()

    def done(self):
        return all([
            self.args.scale is not None,
            self.layout_done(),
            (self.args.hull_vertices is not None and len(self.args.hull_vertices) >= 4)
        ])

    def layout_done(self):
        return all([
            self.args.circle_colour is not None,
            self.args.circle_diameter is not None,
            self.args.plate_circle_separation is not None,
            self.args.plate_width is not None,
            self.args.circles_per_plate is not None,
            self.args.n_plates is not None,
            self.args.circles_cols_first is not None,
            self.args.circles_right_left is not None,
            self.args.circles_bottom_top is not None,
            self.args.plates_cols_first is not None,
            self.args.plates_bottom_top is not None,
            self.args.plates_right_left is not None,
        ])

    def display_panel(self, panel):
        self.disable_btns()
        self.sizer.Add(panel)
        self.SetSizer(self.sizer)
        self.Layout()
        self.Update()

    def launch_layout(self, _=None):
        panel = LayoutPanel(self, self.image)
        self.display_panel(panel)

    def launch_scaler(self, _=None):
        panel = ScalePanel(self, self.image)
        self.display_panel(panel)

    def launch_circle_colour(self, _=None):
        panel = LassoPanel(self, self.image)
        self.display_panel(panel)

    def launch_hull(self, _=None):
        logger.debug("launch wait panel")
        wait_panel = WaitPanel(self)
        self.display_panel(wait_panel)
        wx.Yield()
        if self.segmentor is None:
            self.segmentor = Segmentor(self.images)
            try:
                self.segmentor.run()
            except ValueError:
                wait_panel.Close(True)
                self.enable_btns()
        logger.debug("launch hull panel")
        wait_panel.Close()
        hull_panel = HullPanel(self, self.segmentor)
        self.display_panel(hull_panel)

    def on_exit(self, _=None):
        logger.debug("Exit top frame")
        if not self.done():
            raise ValueError("Not all required configuration values are set")
        logger.debug("Closing")
        self.Destroy()










