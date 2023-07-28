import logging
import numpy as np

import wx
from pathlib import Path
from matplotlib import get_data_path  # we are recycling some matplotlib icons

from ..image_loading import ImageLoaded
from .measure import MeasurePanel

logger = logging.getLogger(__name__)


class ScalePanel(MeasurePanel):

    def __init__(self, parent, image: ImageLoaded):
        self.mm_text = None
        self.px_text = None
        self.scale_text = None

        super().__init__(parent, image)
        logger.debug("Launch scale panel")

    def set_titles(self):
        self.fig.suptitle("Define scale")
        self.ax.set_title("To measure, select the input and click to draw a line")

    def set_measured_input(self, event):
        pass

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
        toolbar.AddControl(wx.StaticText(toolbar, label="Line length (px)"))
        px_text = wx.TextCtrl(toolbar, 2, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(px_text, label="Line length")
        px_text.Bind(wx.EVT_SET_FOCUS, self.set_measured_input)
        px_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)
        self.measured_input = px_text
        self.measured_inputs.append(px_text)

        # Distance text input
        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(toolbar, label="Measured distance (mm)"))
        mm_text = wx.TextCtrl(toolbar, 3, "", style=wx.TE_PROCESS_ENTER)
        mm_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)
        toolbar.AddControl(mm_text, label="Measured distance")
        self.measured_inputs.append(mm_text)

        # Scale text output
        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(toolbar, label="Scale (px/mm)"))
        scale_text = wx.TextCtrl(toolbar, 4, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(scale_text, label="Scale")
        scale_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)
        self.input_to_arg[scale_text] = "scale"
        self.measured_inputs.append(scale_text)

        toolbar.AddSeparator()
        close_btn = wx.Button(toolbar, 5, "Save and close")
        toolbar.AddControl(close_btn)
        close_btn.Bind(wx.EVT_BUTTON, self.on_exit)

        self.sizer.Add(toolbar, 0, wx.ALIGN_LEFT)
        toolbar.Realize()

        mm_text.Bind(wx.EVT_TEXT_ENTER, self.calc_scale)
        px_text.Bind(wx.EVT_TEXT_ENTER, self.calc_scale)

        self.mm_text = mm_text
        self.px_text = px_text
        self.scale_text = scale_text
        self.toolbar = toolbar

    def calc_scale(self, event=None):
        try:
            px = float(self.px_text.GetValue())
        except ValueError:
            logger.debug("Could not coerce px to float")
            return
        try:
            mm = float(self.mm_text.GetValue())
        except ValueError:
            logger.debug("Could not coerce mm to float")
            return
        try:
            scale = np.around(px / mm, decimals=4)
            self.scale_text.SetValue(str(scale))
        except ZeroDivisionError:
            logger.debug("mm distance cannot be equal to 0")

    def update_dependents(self):  # called when line length is set, used to trigger e.g. scale calculation
        self.calc_scale()
