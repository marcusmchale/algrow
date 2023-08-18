import logging
import argparse
from typing import List
from pathlib import Path

import wx
from pubsub import pub


from ..image_loading import ImageLoaded
from ..layout import Layout
from ..options.update_and_verify import calibration_complete, layout_defined

from .loading import ImageLoader, Points, LayoutMultiLoader

from .measure_layout import LayoutPanel
from .measure_scale import ScalePanel
from .hull_pixels import HullPanel

from .lasso import LassoPanel

logger = logging.getLogger(__name__)


# this snippet is useful to catch exceptions from wx.Frame
import sys
import traceback
#
def excepthook(type, value, tb):
    message = 'Uncaught exception:\n'
    message += ''.join(traceback.format_exception(type, value, tb))
    logger.debug(message)

sys.excepthook = excepthook


class Calibrator(wx.App):
    # overriding init to pass in the arguments from CLI/configuration file(s)
    def __init__(self, image_filepaths: List[Path], args: argparse.Namespace, **kwargs):
        logger.debug("Start Calibration App")
        self.image_filepaths = image_filepaths
        self.args = args

        self.images = None
        self.points = None
        self.layouts = None
        self.frame = None

        logger.info("Loading: please wait")
        super().__init__(self, **kwargs)

    def OnInit(self):
        image_loader = ImageLoader(self.image_filepaths, self.args)
        image_loader.run()
        self.images: List[ImageLoaded] = image_loader.images

        points = Points(self.images)
        points.calculate()
        self.points = points

        multilayout = LayoutMultiLoader(self.images)
        multilayout.run()
        self.layouts = multilayout.layouts

        logger.info("Prepare configuration GUI")
        self.frame = TopFrame(self.images, self.points, self.layouts, self.args)
        logger.info("Start configuration GUI")
        self.frame.Show(True)
        return True

    def OnExit(self):
        if not calibration_complete(self.args):
            raise ValueError("Not all calibration parameters were defined")
        # Output a file summarising the calibration values: selected colours, alpha and delta values
        self.write_calibration()
        return int(1)

    def write_calibration(self):
        logger.info("Write out calibration parameters")
        with open(Path(self.args.out_dir, "calibration.conf"), 'w') as text_file:
            if self.args.whole_image:
                text_file.write(f"[Scale]\n")
                text_file.write(f"scale = {self.args.scale}\n")

                text_file.write(f"[Colour parameters]\n")
                hull_vertices_string = f'{[",".join([str(j) for j in i]) for i in self.args.hull_vertices]}'.replace(
                    "'", '"')
                text_file.write(f"hull_vertices = {hull_vertices_string}\n")
                text_file.write(f"alpha = {self.args.alpha}\n")
                text_file.write(f"delta = {self.args.delta}\n")
            else:
                text_file.write(f"[Scale]\n")
                text_file.write(f"scale = {self.args.scale}\n")

                text_file.write(f"[Colour parameters]\n")
                circle_colour_string = f"\"{','.join([str(i) for i in self.args.circle_colour])}\""
                hull_vertices_string = f'{[",".join([str(j) for j in i]) for i in self.args.hull_vertices]}'.replace(
                    "'", '"')
                text_file.write(f"circle_colour = {circle_colour_string}\n")
                text_file.write(f"hull_vertices = {hull_vertices_string}\n")
                text_file.write(f"alpha = {self.args.alpha}\n")
                text_file.write(f"delta = {self.args.delta}\n")

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


class TopFrame(wx.Frame):
    def __init__(self, images: List[ImageLoaded], points: Points, layouts: List[Layout], args: argparse.Namespace):
        logger.debug("Load top frame")
        super().__init__(None, title="AlGrow Calibration", size=(1500, 1000))
        self.figure_counter = 0
        self.images = images
        self.layouts = layouts
        self.points = points
        self.args = args

        # a single image for some calibration windows, taken from the middle
        #self.image = self.images[len(self.images) // 2]

        self.SetIcon(wx.Icon(str(Path(Path(__file__).parent, "bmp", "logo.png"))))

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
        if self.args.whole_image:
            active_buttons = [self.scaler_btn, self.hull_btn]
        else:
            #if self.args.fixed_layout is not None:
            #    active_buttons = [self.scaler_btn]
            #else:
            active_buttons = [self.scaler_btn, self.circle_colour_btn]
            if self.args.circle_colour is not None:  # and self.args.fixed_layout is None:
                active_buttons.append(self.layout_btn)
            if layout_defined(self.args):
                active_buttons.append(self.hull_btn)
        if calibration_complete(self.args):
            active_buttons.append(self.cont_btn)
        for b in active_buttons:
            b.Enable()
        self.set_btn_colours()

    def set_btn_colours(self):
        complete_buttons = []
        if self.args.whole_image:
            if self.args.scale is not None:
                complete_buttons.append(self.scaler_btn)
            if self.args.hull_vertices is not None and len(self.args.hull_vertices) >= 4:
                complete_buttons.append(self.hull_btn)
        else:
            if self.args.scale is not None:
                complete_buttons.append(self.scaler_btn)
            if self.args.circle_colour is not None:
                complete_buttons.append(self.circle_colour_btn)
            if layout_defined(self.args):
                complete_buttons.append(self.layout_btn)
            if self.args.hull_vertices is not None and len(self.args.hull_vertices) >= 4:
                complete_buttons.append(self.hull_btn)
        if calibration_complete(self.args):
            complete_buttons.append(self.cont_btn)
        for b in complete_buttons:
            b.SetBackgroundColour((0, 255, 0))

    def disable_btns(self):
        for b in [self.scaler_btn, self.circle_colour_btn, self.layout_btn, self.hull_btn, self.cont_btn]:
            b.SetBackgroundColour(wx.NullColour)
            b.Disable()

    def display_panel(self, panel):
        logger.debug("set button availability")
        self.disable_btns()
        logger.debug("add panel to sizer")
        self.sizer.Add(panel)
        logger.debug("set sizer")
        self.SetSizer(self.sizer)
        logger.debug("calculate layout")
        self.Layout()
        logger.debug("Update")
        self.Update()

    def launch_layout(self, _=None):
        panel = LayoutPanel(self, self.images[0], self.layouts)  # first image is usually better for layout detection
        # todo handle multiple images for all measure/scale panels
        self.display_panel(panel)

    def launch_scaler(self, _=None):
        panel = ScalePanel(self, self.images[0])
        self.display_panel(panel)

    def launch_circle_colour(self, _=None):
        panel = LassoPanel(self, self.images[0])
        self.display_panel(panel)

    def launch_hull(self, _=None):
        logger.debug("launch hull panel")
        #panel = HullPanel(self, self.images, self.points, self.layouts)
        #self.display_panel(panel)
        from .o3d_test import main
        main()

    def on_exit(self, _=None):
        logger.debug("Exit top frame")
        if not calibration_complete(self.args):
            if wx.MessageBox(
                    "Configuration is not complete... continue closing?",
                    "Please confirm",
                    wx.ICON_QUESTION | wx.YES_NO
            ) != wx.YES:
                return
            else:
                logger.warning("Not all required configuration values are set")
                self.Destroy()
        logger.debug("Closing")
        self.Destroy()
        return










