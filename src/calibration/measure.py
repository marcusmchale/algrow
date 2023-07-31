import logging
import numpy as np

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
from ..options.update_and_verify import update_arg


logger = logging.getLogger(__name__)

class MeasurePanel(wx.Panel):
    def __init__(self, parent, image: ImageLoaded):
        super().__init__(parent)
        logger.debug("Launch measure panel")

        self.image = image
        self.args = image.args

        self.fig, self.ax = plt.subplots()
        self.fig.set_dpi(150)
        self.cv = FigureCanvas(self, -1, self.fig)

        self.artist = self.ax.imshow(self.image.rgb)  # , picker=True, animated=True)

        self.click_cid = self.cv.mpl_connect('button_press_event', self.on_click)

        self.move_cid = None

        self.line_start = None
        self.line_end = None
        self.line = None
        self.line_colour = 'b'
        self.line_px = None
        # text input mapped to the corresponding argument (floats only) to set this on close
        self.input_to_arg = dict()
        self.index_to_arg = dict()

        self.nav_toolbar = NavigationToolbar(self.cv)
        self.nav_toolbar.Realize()
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(self.nav_toolbar, 0, wx.ALIGN_CENTER)
        panel_sizer.Add(self.cv, 1, wx.EXPAND)
        self.sizer.Add(panel_sizer, 1, wx.ALIGN_LEFT)
        self.nav_toolbar.update()
        self.SetSizer(self.sizer)

        self.set_titles()
        self.measured_input = None
        self.measured_inputs = []
        self.toolbar = None
        self.add_toolbar()
        self.load_args()

        self.Bind(wx.EVT_CLOSE, self.on_exit)

        self.cv.draw_idle()
        self.cv.flush_events()

    def set_measured_input(self, event):
        focus = self.FindFocus()
        if focus in self.measured_inputs:
            self.measured_input = focus

    def set_line_px(self):
        if all([
            self.measured_input is not None,
            self.line_start is not None,
            self.line_end is not None
        ]):
            #logger.debug(f"start: {self.line_start}, end: {self.line_end}")
            line_length = np.around(np.linalg.norm(np.array(self.line_start) - np.array(self.line_end)), decimals=1)
            self.line_px = line_length
            self.measured_input.SetValue(str(self.line_px))
            self.measured_input.SetBackgroundColour(wx.NullColour)
            self.update_dependents()

    def reset(self, _):
        if self.line is not None:
            self.line.remove()
            self.line = None
            self.cv.draw_idle()
        self.line_start = None
        self.line_end = None
        self.load_args()

    def on_click(self, click_event):
        coords = (click_event.xdata, click_event.ydata) if click_event.xdata is not None else None
        if coords is None or self.measured_input is None:
            return
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
            self.line_start = None
            self.line_end = None
            self.measured_input = None

    def update_line(self, mouse_event):
        if self.line_start and self.line_end:
            return

        mouse_coords = (mouse_event.xdata, mouse_event.ydata) if mouse_event.xdata is not None else None
        if mouse_coords is None:
            return

        if self.line is not None:
            self.line.remove()
            self.line = None

        if self.line_start and not self.line_end:
            xs = (self.line_start[0], mouse_coords[0])
            ys = (self.line_start[1], mouse_coords[1])
            self.line = Line2D(xs, ys, color=self.line_colour)
            self.ax.add_line(self.line)
            self.cv.draw_idle()

    def check_text_is_float(self, event):
        focus = self.FindFocus()
        if focus in self.measured_inputs:
            focus_text = focus
            value = focus_text.GetValue()
            try:
                float(value)
                self.measured_input.SetBackgroundColour(wx.NullColour)
            except ValueError:
                logger.debug(f"Could not coerce {value} to float")
                self.measured_input.SetBackgroundColour((255, 0, 0))

    def check_text_is_int(self, event):
        focus = self.FindFocus()
        if focus in self.measured_inputs:
            focus_text = focus
            value = focus_text.GetValue()
            try:
                value = int(value)
                self.measured_input.SetValue(value)
                self.measured_input.SetBackgroundColour(wx.NullColour)
            except ValueError:
                logger.debug(f"Could not coerce {value} to integer")
                self.measured_input.SetBackgroundColour((255, 0, 0))

    def load_args(self):
        for input_text, arg in self.input_to_arg.items():
            stored_value = vars(self.args)[arg]
            if stored_value is not None:
                input_text.SetValue(str(stored_value))
            else:
                input_text.SetValue("")
        for index, arg in self.index_to_arg.items():
            stored_value = vars(self.args)[arg]
            if stored_value is not None:
                self.toolbar.ToggleTool(index, stored_value)
            else:
                self.toolbar.ToggleTool(index, False)

    def disconnect(self):
        self.cv.mpl_disconnect(self.click_cid)
        if self.move_cid is not None:
            self.cv.mpl_disconnect(self.move_cid)

    # Override the below  to customise
    def add_toolbar(self):
        toolbar = wx.ToolBar(self, id=-1, style=wx.TB_VERTICAL | wx.TB_TEXT)
        # start again button
        toolbar.AddTool(
            1,
            "reset",
            wx.Image(str(Path(get_data_path(), "images", "back.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            'reset args',
            'reset args to stored values',
            None
        )
        self.Bind(wx.EVT_TOOL, self.reset, id=1)

        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(toolbar, label="Dummy float input"))
        dummy_float_input = wx.TextCtrl(toolbar, 2, "", style=wx.TE_PROCESS_ENTER)
        dummy_float_input.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)
        toolbar.AddControl(dummy_float_input, label="Dummy input")
        self.measured_inputs.append(dummy_float_input)
        self.input_to_arg[dummy_float_input] = "dummy_float_argument"
        dummy_float_input.Bind(wx.EVT_SET_FOCUS, self.set_measured_input)
        self.measured_input = dummy_float_input  # set this for the input you want selected first

        # add a close button
        toolbar.AddSeparator()
        close_btn = wx.Button(toolbar, 3, "Save and close")
        toolbar.AddControl(close_btn)
        close_btn.Bind(wx.EVT_BUTTON, self.on_exit)
        self.sizer.Add(toolbar, 0, wx.ALIGN_LEFT)
        toolbar.Realize()
        self.toolbar = toolbar

    def set_titles(self):
        self.fig.suptitle("Measure figure")
        self.ax.set_title("Measure image")

    def save_args(self, args=None):
        if args is None:
            args = self.args
            temporary = False
        else:
            temporary = True
        done = True
        for text_input, arg in self.input_to_arg.items():
            value = text_input.GetValue()
            try:
                update_arg(args, arg, value, temporary=temporary)
                text_input.SetBackgroundColour(wx.NullColour)
            except ValueError:
                logger.debug(f"Could not coerce {value} to float for {arg}")
                text_input.SetBackgroundColour(wx.Colour(255, 0, 0))
                done = False
        for index, arg in self.index_to_arg.items():
            value = self.toolbar.GetToolState(index)
            update_arg(args, arg, value, temporary=temporary)
        return done

    def on_exit(self, event):
        done = self.save_args()
        if done:
            self.disconnect()
            logger.debug("Close measure window")
            pub.sendMessage("enable_btns")
            plt.close()
            self.Destroy()

    def update_dependents(self):  # called when line length is set, used to trigger e.g. scale calculation
        pass

