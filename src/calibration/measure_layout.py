import logging

import wx
from pathlib import Path

from matplotlib import get_data_path  # we are recycling some matplotlib icons
from matplotlib.patches import Circle

from ..image_loading import ImageLoaded
from .measure import MeasurePanel
from ..options.update_and_verify import layout_defined
from ..layout import LayoutDetector, Plate, Layout, InsufficientPlateDetection, ImageContentException, InsufficientCircleDetection
from .popframe import PopFrame

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class LayoutPanel(MeasurePanel):
    def __init__(self, parent, image: ImageLoaded):
        super().__init__(parent, image)
        logger.debug("Launch layout panel")
        self.annotations = list()

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
        toolbar.AddControl(wx.StaticText(toolbar, label="Measure Circle diameter"))
        circle_dia_text = wx.TextCtrl(toolbar, 2, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(circle_dia_text, label="Circle diameter")
        self.input_to_arg[circle_dia_text] = "circle_diameter"
        circle_dia_text.Bind(wx.EVT_SET_FOCUS, self.set_measured_input)
        circle_dia_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)
        #self.measured_input = circle_dia_text  # set this for the input you want selected first
        self.measured_inputs.append(circle_dia_text)

        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(toolbar, label="Circle expansion factor"))
        circle_expansion_text = wx.TextCtrl(toolbar, 3, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(circle_expansion_text, label="Circle expansion factor")
        self.input_to_arg[circle_expansion_text] = "circle_expansion"
        circle_expansion_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)

        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(
            toolbar,
            label="Measure circle separation\n(distance between edges \nwithin a plate)")
        )
        circle_sep_text = wx.TextCtrl(toolbar, 4, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(circle_sep_text, label="Circle separation")
        self.input_to_arg[circle_sep_text] = "plate_circle_separation"
        circle_sep_text.Bind(wx.EVT_SET_FOCUS, self.set_measured_input)
        circle_sep_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)
        self.measured_inputs.append(circle_sep_text)

        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(toolbar, label="Plate cut-height expansion"))
        plate_cut_expansion_text = wx.TextCtrl(toolbar, -1, "", style=wx.TE_PROCESS_ENTER)
        toolbar.AddControl(plate_cut_expansion_text, label="Plate cut-height expansion")
        self.input_to_arg[plate_cut_expansion_text] = "plate_cut_expansion"
        circle_expansion_text.Bind(wx.EVT_TEXT_ENTER, self.check_text_is_float)

        toolbar.AddSeparator()
        toolbar.AddControl(wx.StaticText(toolbar, label="Measure plate width"))
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

    def add_label(self, text, coords, color, size):
        an = self.ax.annotate(text, coords, color=color, size=size, ha='center', va='center')
        self.annotations.append(an)

    def add_circle(self, coords, radius, color):
        an = self.ax.add_patch(Circle(coords, radius, color=color, fill=False))
        self.annotations.append(an)

    def draw_overlay(self, layout):
        unit = 0
        logger.debug("Annotate canvas with layout")
        for p in layout.plates:
            logger.debug(f"Processing plate {p.id}")
            self.add_label(str(p.id), p.centroid, "red", 10)
            for j, c in enumerate(p.circles):
                unit += 1
                self.add_label(str(unit), (c[0], c[1]), "blue", 5)
                self.add_circle((c[0], c[1]), c[2], "white")
        self.cv.draw_idle()
        self.cv.flush_events()

    def test_layout(self, _=None):
        logger.debug(self.args.plate_width)
        image = self.image.copy()  # deep copy so this doesn't share args with the main args
        done = self.save_args(image.args)  # write the current displayed args to this temporary copy for testing

        logger.debug("Remove existing annotations")
        for an in self.annotations:
            an.remove()
        self.annotations.clear()

        logger.debug("Detect layout")
        if done and layout_defined(image.args):  # todo highlight any missing parameters
            layout_detector = LayoutDetector(image)
            plt.close()
            fig = self.image.figures.new_figure("Plate detection", cols=2, level="WARN")
            # force draw by setting level to warn
            # this is so the figure is available to plot to frame if it fails
            # we then need to later manually control the printing at debug level only if successful
            circles = None
            plates = None
            for i in range(5):  # try 5 times to find enough circles to make plates
                try:
                    circles = layout_detector.find_n_circles(image.args.circles_per_plate * image.args.n_plates, i, fig)
                except InsufficientCircleDetection:
                    logger.debug("Not enough circles found, relax parameters and try again")
                    continue
                try:
                    clusters, target_clusters = layout_detector.find_plate_clusters(
                        circles,
                        image.args.circles_per_plate,
                        image.args.n_plates,
                        fig=fig
                    )
                    plates = [
                        Plate(
                            cluster_id,
                            circles[[i for i, j in enumerate(clusters.flat) if j == cluster_id]],
                        ) for cluster_id in target_clusters
                    ]
                    plates = layout_detector.sort_plates(plates)
                    layout = Layout(plates, image)
                    self.draw_overlay(layout)
                    if self.args.image_debug_level <= 0:  # (i.e. debug level):
                        fig.print()
                    return
                except InsufficientPlateDetection:
                    logger.debug(f"Try again with detection of more circles")
                    continue
                except ImageContentException:
                    logger.debug(f"More plates were detected than defined")
                    break
            logger.info(f"Layout not detected")
            message = None
            if circles is None:
                message = f"No circles detected"

            elif len(circles) < image.args.circles_per_plate * image.args.n_plates:
                message = f"Insufficient circles detected: {len(circles)}"
            elif plates is None:
                message = f"No plates detected"
            elif len(plates) < image.args.n_plates:
                message = f"Insufficient plates detected: {len(plates)}"
            elif len(plates) > image.args.n_plates:
                message = f"Too many plates detected: {len(plates)}"
            logger.info(message)
            popframe = PopFrame(message, fig)
            popframe.Show(True)
            if self.args.image_debug_level <= 0:   # (i.e. debug level):
                fig.print()
