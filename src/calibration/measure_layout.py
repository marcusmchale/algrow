import logging

from typing import List

import wx
from pathlib import Path

from matplotlib import get_data_path  # we are recycling some matplotlib icons

from ..image_loading import ImageLoaded
from .measure import MeasurePanel
from ..options import layout_args_provided, update_arg
from ..layout import LayoutDetector, Layout, ImageContentException, InsufficientPlateDetection, InsufficientCircleDetection

logger = logging.getLogger(__name__)

"""
- calibration window for date, time, block regex
  - calibration window for layout
    - circle diameter, circle separation and plate width similar to scale.
    - enter text fields for circle expansion, circles per plate, n_plates
    - checkbox for ID increment fields
    - provide a "test layout" button to ensure detection 
      - display dendrograms in side window, click through each of them

          
          "--circle_expansion",
          help="Optional expansion factor for circles (increases radius to search, circles must not overlap)",
          



"""


class LayoutPanel(MeasurePanel):
    def __init__(self, parent, images: List[ImageLoaded]):
        super().__init__(parent, images)
        logger.debug("Launch layout panel")

    def set_titles(self):
        self.fig.suptitle("Define layout detection parameters")
        self.ax.set_title("Select the box for each parameter then draw the corresponding line")

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
        toolbar.AddControl(wx.StaticText(toolbar, label="Circle diameter"))
        circle_dia_text = wx.TextCtrl(toolbar, 2, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(circle_dia_text, label="Circle diameter")
        self.input_to_arg[circle_dia_text] = "circle_diameter"
        circle_dia_text.Bind(wx.EVT_SET_FOCUS, self.set_measured_input)
        circle_dia_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)
        self.measured_input = circle_dia_text  # set this for the input you want selected first
        self.measured_inputs.append(circle_dia_text)

        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(toolbar, label="Circle expansion factor"))
        circle_expansion_text = wx.TextCtrl(toolbar, 3, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(circle_expansion_text, label="Circle expansion factor")
        self.input_to_arg[circle_expansion_text] = "circle_expansion"
        circle_expansion_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)

        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(toolbar, label="Circle separation\n(within plate)"))
        circle_sep_text = wx.TextCtrl(toolbar, 4, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(circle_sep_text, label="Circle separation")
        self.input_to_arg[circle_sep_text] = "plate_circle_separation"
        circle_sep_text.Bind(wx.EVT_SET_FOCUS, self.set_measured_input)
        circle_sep_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)
        self.measured_inputs.append(circle_sep_text)

        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(toolbar, label="Plate width"))
        plate_width_text = wx.TextCtrl(toolbar, 5, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(plate_width_text, label="Plate width")
        self.input_to_arg[plate_width_text] = "plate_width"
        plate_width_text.Bind(wx.EVT_SET_FOCUS, self.set_measured_input)
        plate_width_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)
        self.measured_inputs.append(plate_width_text)

        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(toolbar, label="Circles per plate"))
        circles_per_plate_text = wx.TextCtrl(toolbar, 6, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(circles_per_plate_text, label="Circles per plate")
        self.input_to_arg[circles_per_plate_text] = "circles_per_plate"
        circles_per_plate_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_int)

        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(toolbar, label="Plates per image"))
        plates_per_image_text = wx.TextCtrl(toolbar, 7, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(plates_per_image_text, label="Plates per image")
        self.input_to_arg[plates_per_image_text] = "n_plates"
        plates_per_image_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_int)

        checkbox_toolbar = wx.ToolBar(self, id=-1, style=wx.TB_VERTICAL | wx.TB_TEXT)
        checkbox_toolbar.AddControl(wx.StaticText(checkbox_toolbar, label="ID increment options"))
        checkbox_toolbar.AddCheckTool(
            8,
            "Plates in columns",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "plates_in_columns.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Plates in columns",
            'Plates in columns'
        )
        self.index_to_arg[8] = "plates_cols_first"
        checkbox_toolbar.AddCheckTool(
            9,
            "Plates right to left",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "plates_right_to_left.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Plates right to left",
            'Plates right to left'
        )
        self.index_to_arg[9] = "plates_right_left"
        checkbox_toolbar.AddCheckTool(
            10,
            "Plates bottom to top",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "plates_bottom_to_top.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Plates bottom to top",
            'Plates bottom to top'
        )
        self.index_to_arg[10] = "plates_bottom_top"
        checkbox_toolbar.AddCheckTool(
            11,
            "Circles in columns",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "circles_in_columns.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Circles in columns",
            'Circles in columns'
        )
        self.index_to_arg[11] = "circles_cols_first"
        checkbox_toolbar.AddCheckTool(
            12,
            "Circles right to left",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "circles_right_to_left.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Circles right to left",
            'Circles right to left'
        )
        self.index_to_arg[12] = "circles_right_left"
        checkbox_toolbar.AddCheckTool(
            13,
            "Circles bottom to top",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "circles_bottom_to_top.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Circles bottom to top",
            'Circles bottom to top'
        )
        self.index_to_arg[13] = "circles_bottom_top"

        toolbar.AddSeparator()
        self.test_btn = wx.Button(toolbar, 14, "Test layout")
        toolbar.AddControl(self.test_btn)
        self.test_btn.Bind(wx.EVT_BUTTON, self.test_layout)

        toolbar.AddSeparator()
        close_btn = wx.Button(toolbar, 15, "Save and close")
        toolbar.AddControl(close_btn)
        close_btn.Bind(wx.EVT_BUTTON, self.on_exit)

        self.sizer.Add(toolbar, 0, wx.ALIGN_LEFT)
        self.sizer.Add(checkbox_toolbar, 0, wx.ALIGN_LEFT)
        toolbar.Realize()
        checkbox_toolbar.Realize()
        self.toolbar = checkbox_toolbar

    def test_layout(self, _=None):
        logger.debug(self.args.plate_width)
        image = self.image.copy()  # deep copy so this doesn't share args with the main args
        done = self.save_args(image.args)  # write the current displayed args to this temporary copy for testing
        if done and layout_args_provided(image.args):  # todo highlight any missing parameters
            try:
                layout: Layout = LayoutDetector(image).get_layout()
                self.test_btn.SetBackgroundColour(wx.NullColour)
            except (ImageContentException, InsufficientPlateDetection, InsufficientCircleDetection):
                self.test_btn.SetBackgroundColour(wx.Colour(255, 0, 0))
                return  # todo improve feedback on failure...
            self.artist.set_data(layout.overlay)
        self.cv.draw_idle()
        self.cv.flush_events()
