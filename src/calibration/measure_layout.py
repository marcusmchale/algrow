import logging

import wx


import pandas as pd
from pathlib import Path
from typing import List, Optional

from matplotlib import get_data_path, patheffects
from matplotlib.patches import Circle

from ..image_loading import ImageLoaded
from .measure import MeasurePanel
from ..options.update_and_verify import layout_defined, update_arg
from ..layout import (
    LayoutDetector,
    Plate,
    Layout,
    InsufficientPlateDetection,
    ImageContentException,
    InsufficientCircleDetection
)
from .loading import wait_for_multiprocessing
from ..logging import worker_log_configurer
from .popframe import PopFrame

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class LayoutPanel(MeasurePanel):
    def __init__(self, parent, image: ImageLoaded, layouts: List[Optional[Layout]]):
        self.plates = None
        for i, j in enumerate(layouts):  # clear stored layouts to force recalculation
            layouts[i] = None
        self.layouts = layouts
        super().__init__(parent, image)
        logger.debug("Launch layout panel")
        self.annotations = list()

    def set_titles(self):
        self.fig.suptitle("Define layout detection parameters")
        self.ax.set_title("To measure, select the input then click to draw a line")

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

        checkbox_toolbar.AddSeparator()
        checkbox_toolbar.AddCheckTool(
            15,
            "Fixed layout",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "fixed_layout.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Fixed layout",
            'Fixed layout'
        )
        #"fixed_layout" is disabled until test layout has run successfully and self.plates is not None
        if self.plates is None:
            checkbox_toolbar.EnableTool(15, False)

        toolbar.AddSeparator()
        close_btn = wx.Button(toolbar, 16, "Save and close")
        toolbar.AddControl(close_btn)
        close_btn.Bind(wx.EVT_BUTTON, self.on_exit)

        self.sizer.Add(toolbar, 0, wx.ALIGN_LEFT)
        self.sizer.Add(checkbox_toolbar, 0, wx.ALIGN_LEFT)
        toolbar.Realize()
        checkbox_toolbar.Realize()
        self.toolbar = checkbox_toolbar

    def write_plates(self):
        if self.plates is None:
            raise ValueError("No plate specification defined - must generate test layout first")

        circles_dicts = list()
        for i, p in enumerate(self.plates):
            for j, c in enumerate(p.circles):
                circles_dicts.append({
                    "plate_id": i + 1,
                    "plate_x": p.centroid[0],
                    "plate_y": p.centroid[1],
                    "circle_id": j + 1,
                    "circle_x": c[0],
                    "circle_y": c[1],
                    "circle_radius": c[2]
                })
        df = pd.DataFrame.from_records(circles_dicts, index=["plate_id", "circle_id"])
        if self.args.downscale != 1:
            df = df.multiply(self.args.downscale)
        outfile = Path(self.args.out_dir, "layout.csv")
        df.to_csv(outfile, index=True)
        update_arg(self.args, "fixed_layout", outfile)

    def on_exit(self, event):
        if self.toolbar.GetToolState(15):
            self.write_plates()
        super().on_exit(event)

    def add_label(self, text, coords, color, size):
        an = self.ax.annotate(
            text,
            coords,
            color=color,
            size=size,
            ha='center',
            va='center',
            path_effects=[patheffects.withStroke(linewidth=2, foreground='white')]
        )
        self.annotations.append(an)

    def add_circle(self, coords, radius):
        an = self.ax.add_patch(
            Circle(
                coords,
                radius,
                color="white",
                fill=False
            )
        )
        self.annotations.append(an)

    def draw_overlay(self, layout):
        unit = 0
        logger.debug("Annotate canvas with layout")
        for p in layout.plates:
            logger.debug(f"Processing plate {p.id}")
            self.add_label(str(p.id), p.centroid, "black", 10)
            for j, c in enumerate(p.circles):
                unit += 1
                self.add_label(str(unit), (c[0], c[1]), "black", 5)
                self.add_circle((c[0], c[1]), c[2])
        self.cv.draw_idle()
        self.cv.flush_events()

    def test_layout(self, _=None):
        image = self.image.copy()  # deep copy so this doesn't share args with the main args
        done = self.save_args(image.args)  # write the current displayed args to this temporary copy for testing

        logger.debug("Remove existing annotations")
        for an in self.annotations:
            an.remove()
        self.annotations.clear()

        if done and layout_defined(image.args):  # todo highlight any missing parameters
            logger.debug("Detect layout")
            fig = self.image.figures.new_figure("Plate detection", cols=2, level="WARN")
            kwargs_list = [{"image": image, 'fig': fig}]

            result = wait_for_multiprocessing("Detecting layout", 1, get_layout, kwargs_list)[0]

            plates = result['plates']
            circles = result['circles']
            fig = result['fig']
            if self.args.image_debug <= 0:  # (i.e. debug level):
                fig.print()
            self.plates = plates
            self.toolbar.EnableTool(15, True)

            try:
                logger.debug("prepare to draw layout")
                layout = Layout(plates, image)
                self.layouts[0] = layout
                self.draw_overlay(layout)
            except:
                message = None
                if circles is None:
                    message = f"No circles detected"
                elif len(circles) < image.args.circles_per_plate * image.args.n_plates:
                    message = f"Insufficient circles detected: {len(circles)}"
                elif plates is None:
                    message = f"Incorrect number of plates detected"
                #elif len(plates) < image.args.n_plates:
                #    message = f"Insufficient plates detected: {len(plates)}"
                #elif len(plates) > image.args.n_plates:
                #    message = f"Too many plates detected: {len(plates)}"
                logger.info(message)
                self.toolbar.EnableTool(15, False)
                popframe = PopFrame(message, fig)
                popframe.Show(True)


def get_layout(image, fig, log_queue=None):
    if log_queue is not None:
        worker_log_configurer(log_queue)

    layout_detector = LayoutDetector(image)
    plt.close()
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
            break
        except InsufficientPlateDetection:
            logger.debug(f"Try again with detection of more circles")
            continue
        except ImageContentException:
            logger.debug(f"More plates were detected than defined")
            break
    return {
        "circles": circles,
        "plates": plates,
        "fig": fig
    }
