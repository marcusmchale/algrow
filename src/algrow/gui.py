import logging
import argparse
import datetime
import open3d as o3d
import platform
import numpy as np
import pandas as pd
import re
import webbrowser

from enum import Enum
from pathlib import Path
from open3d.visualization import gui, rendering
from skimage.color import lab2rgb
from skimage.morphology import remove_small_holes, remove_small_objects

from .options import IMAGE_EXTENSIONS, configuration_complete, options, update_arg, layout_defined, postprocess, DebugEnum
from .hull import HullHolder
from .image_loading import ImageLoaded, MaskLoaded, CalibrationImage, LayoutLoader, LayoutDetector
from .figurebuilder import FigureMatplot
from .layout import Layout, ExcessPlatesException, InsufficientPlateDetection, InsufficientCircleDetection
from .panel import Panel
from .area_calculation import calculate_area
from .analysis import analyse
from ._version import __version__

from typing import Optional


isMacOS = (platform.system() == "Darwin")

logger = logging.getLogger(__name__)


class Activities(Enum):
    NONE = 0
    SCALE = 1
    CIRCLE = 2
    LAYOUT = 3
    TARGET = 4
    AREA = 5
    RGR = 6


class AppWindow:
    MENU_OPEN = 1
    MENU_MASK = 2
    MENU_LOAD_CONF = 3
    MENU_WRITE_CONF = 4
    MENU_QUIT = 7

    MENU_SCALE = 11
    MENU_CIRCLE = 12
    MENU_LAYOUT = 13
    MENU_TARGET = 14

    MENU_AREA = 21
    MENU_RGR = 22

    MENU_ABOUT = 40

    def __init__(self, width, height, fonts, args):
        self.fonts = fonts
        self.args: argparse.Namespace = args
        self.app = gui.Application.instance
        self.window = self.app.create_window("AlGrow", width, height)
        self.window.set_on_layout(self._on_layout)

        self.image = None
        self.image_window = None
        self.activity = Activities.NONE

        self.prior_lab = set()
        #
        self.hull_holder = None

        # these are for measuring and moving
        self.drag_start = None
        self.measure_start = None

        # For more consistent spacing across systems we use font size rather than fixed pixels
        self.em = self.window.theme.font_size

        # we need a generic material for the 3d plot
        # todo refactor as a mapping i.e. self.materials: dict()
        self.point_material = self.get_material("point")
        self.line_material = self.get_material("line")
        self.mesh_material = self.get_material("mesh")

        # prepare widgets, layouts and panels
        self.labels = list()  # used to store and subsequently remove 3d labels - todo remove by type?
        self.annotations = list()  # used to store and subsequently remove annotations from image widget

        logo_path = Path(Path(__file__).parent, "resources", "logo.png")
        logger.debug(f"Looking for logo at: {logo_path}")
        self.background_widget = gui.ImageWidget(str(logo_path))
        self.window.add_child(self.background_widget)

        self.lab_widget = self.get_lab_widget()
        self.info = self.get_info()
        self.image_widget = self.get_image_widget()

        self.tool_layout = gui.Horiz(0.5 * self.em, gui.Margins(0.5 * self.em))
        self.tool_layout.visible = False
        self.window.add_child(self.tool_layout)

        self.scale_panel = self.get_scale_panel()

        self.target_panel = self.get_target_panel()
        self.circle_panel = self.get_circle_panel()
        self.update_displayed_circle_colour()

        self.layout_panel = self.get_layout_panel()

        self.area_panel = self.get_area_panel()
        self.rgr_panel = self.get_rgr_panel()

        self.load_all_parameters()
        self.prepare_menu()

    def prepare_menu(self):
        logger.debug("Prepare menu")
        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if self.app.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_item("Load Image", AppWindow.MENU_OPEN)
            file_menu.add_item("Load Mask", AppWindow.MENU_MASK)
            file_menu.add_item("Load Configuration", AppWindow.MENU_LOAD_CONF)
            file_menu.add_item("Save Configuration", AppWindow.MENU_WRITE_CONF)
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            else:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            configuration_menu = gui.Menu()
            configuration_menu.add_item("Scale", AppWindow.MENU_SCALE)
            configuration_menu.add_item("Circle detection", AppWindow.MENU_CIRCLE)
            configuration_menu.add_item("Circle layout", AppWindow.MENU_LAYOUT)
            configuration_menu.add_item("Target colour", AppWindow.MENU_TARGET)
            analysis_menu = gui.Menu()
            analysis_menu.add_item("Area", AppWindow.MENU_AREA)
            analysis_menu.add_item("Growth", AppWindow.MENU_RGR)

            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)
            # todo add link to pdf with instructions and citation/paper when published to help menu

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("AlGrow", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Configuration", configuration_menu)
                menu.add_menu("Analysis", analysis_menu)
                # Don't include help menu unless it has something more than

            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Configuration", configuration_menu)
                menu.add_menu("Analysis", analysis_menu)
                menu.add_menu("Help", help_menu)
            self.app.menubar = menu
            self.set_menu_enabled()

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        self.window.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        self.window.set_on_menu_item_activated(AppWindow.MENU_MASK, self._on_menu_mask)
        self.window.set_on_menu_item_activated(AppWindow.MENU_WRITE_CONF, self._on_menu_write_conf)
        self.window.set_on_menu_item_activated(AppWindow.MENU_LOAD_CONF, self._on_menu_load_conf)
        self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        self.window.set_on_menu_item_activated(AppWindow.MENU_SCALE, self.start_scale)
        self.window.set_on_menu_item_activated(AppWindow.MENU_CIRCLE, self.start_circle)
        self.window.set_on_menu_item_activated(AppWindow.MENU_LAYOUT, self.start_layout)
        self.window.set_on_menu_item_activated(AppWindow.MENU_TARGET, self.start_target)
        self.window.set_on_menu_item_activated(AppWindow.MENU_AREA, self.start_area)
        self.window.set_on_menu_item_activated(AppWindow.MENU_RGR, self.start_rgr)
        self.window.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

    def get_material(self, key: str):
        logger.debug(f"Prepare material for {key}")
        if key == "point":
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = 2 * self.window.scaling
        elif key == "line":
            material = rendering.MaterialRecord()
            material.shader = "unlitLine"
            material.line_width = 0.2 * self.window.theme.font_size
        elif key == "mesh":
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
        else:
            raise KeyError("material type is not defined")
        return material

    def get_lab_widget(self):
        logger.debug("Prepare Lab scene (space for 3D model)")
        widget = gui.SceneWidget()
        widget.visible = False
        widget.scene = rendering.Open3DScene(self.window.renderer)
        widget.scene.set_background([0, 0, 0, 1])
        widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        widget.scene.scene.enable_sun_light(False)
        widget.scene.scene.enable_indirect_light(True)
        widget.scene.scene.set_indirect_light_intensity(60000)
        widget.set_on_mouse(self.on_mouse_lab_widget)
        self.window.add_child(widget)
        return widget

    def get_image_widget(self):
        widget = gui.ImageWidget()
        widget.visible = False
        widget.set_on_mouse(self.on_mouse_image_widget)
        self.window.add_child(widget)
        return widget

    def get_info(self):
        info = gui.Label("")
        info.visible = False
        self.window.add_child(info)
        return info

    def get_scale_panel(self):
        logger.debug("Prepare scale panel")
        scale_panel = Panel(gui.Vert(spacing=self.em, margins=gui.Margins(self.em)), self.tool_layout)
        scale_panel.add_label("Scale parameters", self.fonts['large'])
        scale_panel.add_label("Shift-click on two points to draw a line", self.fonts['small'])
        scale_panel.add_input("px", float, tooltip="Length of line", on_changed=self.update_scale)
        scale_panel.add_input("mm", float, tooltip="Physical distance", on_changed=self.update_scale)
        scale_panel.add_input("scale", float, tooltip="Scale (px/mm)", on_changed=self.set_scale)
        scale_panel.add_input("line colour", gui.Color, value=gui.Color(1.0, 0.0, 0.0), tooltip="Set line colour")
        return scale_panel

    def get_layout_panel(self):
        logger.debug("Prepare layout panel")
        layout_panel = Panel(gui.Vert(spacing=self.em, margins=gui.Margins(self.em)), self.tool_layout)
        layout_panel.add_label("Layout detection parameters", self.fonts['large'])
        layout_horiz = Panel(gui.Horiz(spacing=self.em, margins=gui.Margins(self.em)), layout_panel)
        layout_numbers = Panel(gui.Vert(spacing=self.em, margins=gui.Margins(self.em)), layout_horiz)
        layout_buttons = Panel(gui.Vert(spacing=self.em, margins=gui.Margins(self.em)), layout_horiz)

        layout_numbers.add_label(
            "Shift-click on two points to draw a line\nand copy the value into measured fields", self.fonts['small']
        )
        layout_numbers.add_input("px", float, tooltip="Line length")
        layout_numbers.add_input("line colour", gui.Color, value=gui.Color(1.0, 0.0, 0.0), tooltip="Set line colour")
        layout_numbers.add_separation(2)
        layout_numbers.add_label("Measured parameters", self.fonts['small'])
        layout_numbers.add_input(
            "circle diameter",
            float,
            tooltip="Diameter of circle (px)",  #used to detect circles and calculate plate cut height
            on_changed=lambda event: update_arg(
                self.args,
                "circle_diameter",
                event
            )
        )
        layout_numbers.set_value("circle diameter", self.args.circle_diameter)
        layout_numbers.add_input(
            "circle separation",
            float,
            tooltip="Maximum distance between edges of circles within a plate (px)",  # used to calculate plate cut height
            on_changed=lambda event: update_arg(self.args, "circle_separation", event)
        )
        layout_numbers.set_value("circle separation", self.args.circle_separation)
        layout_numbers.add_input(
            "plate width",
            float,
            tooltip="Shortest dimension of plate (px)",  # used to calculate cut height for clustering plates into rows or columns
            on_changed=lambda event: update_arg(self.args, "plate_width", event)
        )
        layout_numbers.set_value("plate width", self.args.plate_width)
        layout_numbers.add_separation(2)
        layout_numbers.add_label("Counts", self.fonts['small'])
        layout_numbers.add_input(
            "circles",
            int,
            tooltip="Number of circles to detect",
            on_changed=lambda event: update_arg(self.args, "circles", event)
        )
        layout_numbers.set_value("circles", self.args.circles)
        layout_numbers.add_input(
            "plates",
            int,
            tooltip="Number of plates to detect",
            on_changed=lambda event: update_arg(self.args, "plates", event)
        )
        layout_numbers.set_value("plates", self.args.plates)
        layout_numbers.add_input(
            "circles per plate",
            str,
            tooltip="Number of circles per plate (comma separated integers)",
            on_changed=lambda event: update_arg(self.args, "circles_per_plate", event.split(','))
        )
        layout_numbers.set_value("circles per plate", self.args.circles_per_plate)
        layout_numbers.add_separation(2)
        layout_numbers.add_label("Tolerance factors", self.fonts['small'])
        layout_numbers.add_input(
            "circle variability",
            float,
            tooltip="Higher values broaden the range of radii for circle detection",
            on_changed=lambda event: update_arg(self.args, "circle_variability", event)
        )
        layout_numbers.set_value("circle variability", self.args.circle_variability)
        layout_numbers.add_input(
            "circle expansion",
            float,
            tooltip="Applied to radius of detected circle to define the region of interest",
            on_changed=lambda event: update_arg(self.args, "circle_expansion", event)
        )
        layout_numbers.set_value("circle expansion", self.args.circle_expansion)
        layout_numbers.add_input(
            "circle separation tolerance",
            float,
            tooltip="Applied to cut height when clustering circles into plates",
            on_changed=lambda event: update_arg(self.args, "circle_separation_tolerance", event)
        )
        layout_numbers.set_value("circle separation tolerance", self.args.circle_separation_tolerance)
        layout_buttons.add_label("Plate ID incrementation", self.fonts['small'])
        layout_buttons.add_button(
            "plates in rows",
            lambda: self.toggle_layout_increment_button("plates in rows"),
            tooltip="Increment plates in rows",
            toggleable=True
        )
        layout_buttons.add_button(
            "plates start left",
            lambda: self.toggle_layout_increment_button("plates start left"),
            tooltip="Increment plates left to right",
            toggleable=True
        )
        layout_buttons.add_button(
            "plates start top",
            lambda: self.toggle_layout_increment_button("plates start top"),
            tooltip="Increment plates top to bottom",
            toggleable=True
        )
        layout_buttons.add_label("Circle ID incrementation", self.fonts['small'])
        layout_buttons.add_button(
            "circles in rows",
            lambda: self.toggle_layout_increment_button("circles in rows"),
            tooltip="Increment circles in rows",
            toggleable=True
        )
        layout_buttons.add_button(
            "circles start left",
            lambda: self.toggle_layout_increment_button("circles start left"),
            tooltip="Increment circles left to right",
            toggleable=True
        )
        layout_buttons.add_button(
            "circles start top",
            lambda: self.toggle_layout_increment_button("circles start top"),
            tooltip="Increment circles top to bottom",
            toggleable=True
        )
        layout_buttons.add_separation(2)

        layout_buttons.add_button("detect layout", self.detect_layout, tooltip="detect layout using these parameters")

        layout_buttons.add_button("save fixed layout", self._on_save_fixed, tooltip="Save a fixed layout to a file")
        layout_buttons.add_button("clear layout", self.change_layout, tooltip="clear detected or loaded layout")
        if self.image is not None:
            layout_buttons.buttons['save fixed layout'].enabled = self.image.layout is not None
        return layout_panel

    def detect_layout(self, event=None):
        if not self.save_layout():
            return

        fig = FigureMatplot("Plate detection", 0, self.args, cols=2, image_filepath=self.image.filepath)
        # creating fig to override the arg calibrated figure level
        # we need this to display regardless of the image debug level
        self.image.args = self.args  # todo: should consider whether image should really be carrying the args at all...
        try:
            layout_detector = LayoutDetector(self.image)
            logger.debug(f"find circles: {self.args.circles}")
            logger.debug(f"cluster into plates: {self.args.plates}")
            plates = layout_detector.find_plates(custom_fig=fig)
            plates = layout_detector.sort_plates(plates)

        except ExcessPlatesException:
            self.window.show_message_box("Error", "Excess plates detected")
            plates = None
        except InsufficientCircleDetection:
            self.window.show_message_box("Error", "Insufficient circles detected")
            plates = None
        except InsufficientPlateDetection:
            self.window.show_message_box("Error", "Insufficient plates detected")
            plates = None

        if plates is None:
            self.layout_panel.buttons["save fixed layout"].enabled = False
        else:
            self.change_layout(Layout(plates, self.image.rgb.shape[:2]))
            self.layout_panel.buttons["save fixed layout"].enabled = True

        if self.activity == Activities.LAYOUT and plates is None:
            self.update_image_with_array(fig.as_array(self.image.rgb.shape[1], self.image.rgb.shape[0]))

        else:
            self.update_image_widget()

    def change_layout(self, layout: Optional[Layout] = None):
        if not isinstance(layout, Layout):
            layout = None
            self.layout_panel.buttons["save fixed layout"].enabled = False
        self.image.change_layout(layout)
        # have to reset the cloud/voxel to image details as these won't map if layout is changed
        self.clear_selection()
        self.image.cloud = None
        self.image.voxel_to_image = None
        # also need to clear the selection panel for the same reason

        if layout is None:  # just so when this is called by the button it resets it, otherwise is updated elsewhere
            self.update_image_widget()

    def load_layout_parameters(self):
        self.layout_panel.set_value("circle diameter", self.args.circle_diameter)
        self.layout_panel.set_value("circle variability", self.args.circle_variability)
        self.layout_panel.set_value("circle separation", self.args.circle_separation)
        self.layout_panel.set_value("plate width", self.args.plate_width)
        self.layout_panel.set_value("circle expansion", self.args.circle_expansion)
        self.layout_panel.set_value("circle separation tolerance", self.args.circle_separation_tolerance)
        self.layout_panel.set_value("circles", self.args.circles)
        if self.args.circles_per_plate is not None:
            self.layout_panel.set_value("circles per plate", ','.join([str(i) for i in self.args.circles_per_plate]))
        self.layout_panel.set_value("plates", self.args.plates)
        self.layout_panel.buttons['plates in rows'].is_on = not self.args.plates_cols_first
        self.layout_panel.buttons['plates start left'].is_on = not self.args.plates_right_left
        self.layout_panel.buttons['plates start top'].is_on = not self.args.plates_bottom_top
        self.layout_panel.buttons['circles in rows'].is_on = not self.args.circles_cols_first
        self.layout_panel.buttons['circles start left'].is_on = not self.args.circles_right_left
        self.layout_panel.buttons['circles start top'].is_on = not self.args.circles_bottom_top

    def save_layout(self):
        update_arg(self.args, "circle_diameter", self.layout_panel.get_value("circle diameter"))
        update_arg(self.args, "circle_variability", self.layout_panel.get_value("circle variability"))
        update_arg(self.args, "circle_separation", self.layout_panel.get_value("circle separation"))
        update_arg(self.args, "plate_width", self.layout_panel.get_value("plate width"))
        update_arg(self.args, "circle_expansion", self.layout_panel.get_value("circle expansion"))
        update_arg(self.args, "circle_separation_tolerance", self.layout_panel.get_value("circle separation tolerance"))
        update_arg(self.args, "circles", self.layout_panel.get_value("circles"))
        update_arg(self.args, "plates", self.layout_panel.get_value("plates"))
        try:
            update_arg(self.args, "circles_per_plate", self.layout_panel.get_value("circles per plate").split(','))
        except ValueError:
            logger.debug("Value for circles per plate is malformed")
        update_arg(self.args, "plates_cols_first", not self.layout_panel.get_value("plates in rows"))
        update_arg(self.args, "plates_right_left", not self.layout_panel.get_value("plates start left"))
        update_arg(self.args, "plates_bottom_top", not self.layout_panel.get_value("plates start top"))
        update_arg(self.args, "circles_cols_first", not self.layout_panel.get_value("circles in rows"))
        update_arg(self.args, "circles_right_left", not self.layout_panel.get_value("circles start left"))
        update_arg(self.args, "circles_bottom_top", not self.layout_panel.get_value("circles start top"))
        #self.layout_panel.buttons['test layout'].enabled = layout_defined(self.args)
        self.set_menu_enabled()

        if layout_defined(self.args):
            return True
        else:
            self.window.show_message_box("Error", "Invalid layout definition")
            return False

    def toggle_layout_increment_button(self, key):
        key_to_arg = {  # negate values
            "plates in rows": "plates_cols_first",
            "plates start left": "plates_right_left",
            "plates start top": "plates_bottom_top",
            "circles in rows": "circles_cols_first",
            "circles start left": "circles_right_left",
            "circles start top": "circles_bottom_top"
        }
        key_to_alt = {  # alt text when false
            "plates in rows": "plates in columns",
            "plates start left": "plates start right",
            "plates start top": "plates start bottom",
            "circles in rows": "circles in columns",
            "circles start left": "circles start right",
            "circles start top": "circles start bottom"
        }
        update_arg(self.args, key_to_arg[key], not self.layout_panel.get_value(key))
        self.layout_panel.buttons[key].text = key if self.layout_panel.get_value(key) else key_to_alt[key]
        if self.image is not None and self.image.layout is not None:
            plates = self.image.layout.plates
            layout_detector = LayoutDetector(self.image)
            plates = layout_detector.sort_plates(plates)
            self.image.change_layout(Layout(plates, self.image.rgb.shape[:2]))
        self.update_image_widget()

    def _on_save_fixed(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save to", self.window.theme)
        extensions = [f".csv"]
        dlg.add_filter(" ".join(extensions), f"layout files ({', '.join(extensions)}")
        dlg.add_filter("", "All files")
        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_layout_dialog_done)
        self.window.show_dialog(dlg)

    def _on_layout_dialog_done(self, filename):
        filepath = Path(filename)
        self.window.close_dialog()
        self.save_fixed_layout(filepath)

    def save_fixed_layout(self, filepath):
        if self.image.layout is None:
            raise ValueError("Layout has not been defined")

        circles_dicts = list()
        for i, p in enumerate(self.image.layout.plates):
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
        df.to_csv(filepath, index=True)
        update_arg(self.args, "fixed_layout", filepath)

    def get_circle_panel(self):
        logger.debug("Prepare circle panel")
        panel = Panel(gui.ScrollableVert(spacing=self.em, margins=gui.Margins(self.em)), self.tool_layout)
        panel.add_label("Circle colour parameters", self.fonts['large'])
        panel.add_label("Shift-click to select pixels", self.fonts['small'])
        panel.add_button("Clear", self.clear_circle_selection_and_save, tooltip="Clear selection")
        panel.add_button_pool("circle colour")
        panel.add_label("selected", font_style=self.fonts['small'])
        panel.add_button_pool("selection")
        panel.add_stretch()
        return panel

    def get_target_panel(self):
        logger.debug("Prepare target panel")
        panel = Panel(gui.ScrollableVert(spacing=self.em, margins=gui.Margins(self.em)), self.tool_layout)
        panel.add_label("Target hull parameters", self.fonts['large'])
        target_horiz = Panel(gui.Horiz(spacing=self.em, margins=gui.Margins(self.em)), panel)
        self.add_target_buttons(target_horiz)
        self.add_selection_panel(target_horiz)
        self.add_prior_panel(target_horiz)
        return panel

    def add_target_buttons(self, parent: Panel):
        logger.debug("Prepare target buttons")
        target_buttons = Panel(gui.Vert(spacing=self.em, margins=gui.Margins(self.em)), parent)
        target_buttons.add_label("Shift-click to select voxels or pixels", self.fonts['small'])
        target_buttons.add_input(
            "min pixels",
            int,
            tooltip="Minimum number of pixels to display voxel in the 3D plot",
            on_changed=self.update_lab_widget
        )
        target_buttons.add_input(
            "alpha",
            float,
            tooltip="Radius to connect vertices (0 for convex hull)",
            on_changed=self.update_alpha
        )
        target_buttons.set_value("alpha", self.args.alpha)
        target_buttons.add_input(
            "delta",
            float,
            tooltip="Distance from hull to consider as target",
            on_changed=self.update_delta
        )
        target_buttons.set_value("delta", self.args.delta)
        target_buttons.add_input(
            "fill", int, tooltip="Fill holes smaller than this size", on_changed=self.update_fill
        )
        target_buttons.set_value("fill", self.args.fill)
        target_buttons.add_input(
            "remove", int, tooltip="Remove objects below this size", on_changed=self.update_remove
        )
        target_buttons.set_value("remove", self.args.remove)

        target_buttons.add_button(
            "hull from mask",
            tooltip="Calculate hull vertices from target mask (uses min pixels and alpha)",
            on_clicked=self.hull_from_mask,
            enabled=False
        )
        target_buttons.add_button("show hull", self.draw_hull, tooltip="Display hull in 3D plot", toggleable=True)
        target_buttons.add_button(
            "show selected",
            self.toggle_show_selected,
            tooltip="Highlight pixels from selected voxels in image",
            toggleable=True
        )
        target_buttons.add_button(
            "show target",
            self.toggle_show_target,
            tooltip="Highlight target pixels (inside or within delta of hull surface after fill and remove)",
            toggleable=True
        )
        target_buttons.add_input(
            "hull colour",
            gui.Color,
            value=gui.Color(1.0, 1.0, 1.0),
            tooltip="Set hull colour",
            on_changed=self.draw_hull
        )
        target_buttons.add_input(
            "selection colour",
            gui.Color,
            value=gui.Color(1.0, 0.0, 1.0),
            tooltip="Set selection highlighting colour",
            on_changed=self.update_image_widget
        )
        target_buttons.add_input(
            "target colour",
            gui.Color,
            value=gui.Color(1.0, 0.0, 0.0),
            tooltip="Set target highlighting colour",
            on_changed=self.update_image_widget
        )
        target_buttons.add_stretch()
        return target_buttons

    def add_selection_panel(self, parent: Panel):
        logger.debug("Prepare selection panel")
        panel = Panel(gui.Vert(spacing=self.em, margins=gui.Margins(self.em)), parent)
        panel.add_label("selected", font_style=self.fonts['small'])
        panel.add_button("clear", tooltip="Clear selected colours", on_clicked=self.clear_selection)
        panel.add_button("reduce", tooltip="Keep only hull vertices", on_clicked=self.reduce_selection)
        panel.add_button_pool("selection")
        return panel

    def add_prior_panel(self, parent: Panel):
        logger.debug("Prepare prior panel")
        panel = Panel(gui.Vert(spacing=self.em, margins=gui.Margins(self.em)), parent)
        panel.add_label("priors", font_style=self.fonts['small'])
        panel.add_button("clear", tooltip="Clear prior colours", on_clicked=self.clear_priors)
        panel.add_button('reduce', tooltip="Keep only hull vertices", on_clicked=self.reduce_priors)
        panel.add_button_pool("priors")
        return panel

    def get_area_panel(self):
        logger.debug("Prepare area panel")
        panel = Panel(gui.Vert(spacing=self.em, margins=gui.Margins(self.em)), self.window)
        panel.add_label("Area calculation", self.fonts['large'])

        file_selection_panel = Panel(gui.Horiz(spacing=self.em, margins=gui.Margins(self.em)), panel)
        file_selection_panel.add_button("Add directory", on_clicked=self._on_select_image_directory)
        file_selection_panel.add_button("Add file", on_clicked=self._on_select_image_file)
        file_selection_panel.add_button("Remove selected", lambda: panel.remove_selected_from_list("filename"))
        file_selection_panel.add_button("Remove all", lambda: panel.clear_list("filename"))

        file_details_panel = Panel(gui.Horiz(spacing=self.em, margins=gui.Margins(self.em)), panel)

        def parse_filenames():
            files = panel.list_view_lists["filename"]
            update_arg(self.args, "images", files)
            if files:
                try:
                    filename_groups = re.compile(self.args.filename_regex).groupindex
                    filename_searches = [re.search(self.args.filename_regex, f) for f in files]
                    if 'block' in filename_groups:
                        blocks = [b.group(filename_groups['block']) if b else "" for b in filename_searches]
                    else:
                        blocks = list()
                    panel.list_view_lists["block"] = blocks
                    time_values = ['year', 'month', 'day', 'hour', 'minute', 'second']
                    for i, tv in enumerate(time_values):
                        if tv not in filename_groups:
                            time_values = time_values[0:i]
                            break
                    time_tuples = [
                        tuple(
                            int(t.group(filename_groups[tv])) if t else None for tv in time_values
                        ) for t in filename_searches
                    ]
                    panel.list_view_lists["time"] = [
                        str(datetime.datetime(*tt)) if tt and all([n is not None for n in tt]) else "" for tt in time_tuples
                    ]
                    panel.list_views["block"].set_items(panel.list_view_lists["block"])
                    panel.list_views["time"].set_items(panel.list_view_lists["time"])
                except Exception as e:
                    panel.list_views["block"].set_items(list())
                    panel.list_views["time"].set_items(list())
                    logger.debug(e)
                    pass

        file_details_panel.add_list_view("filename", callback=parse_filenames)
        file_details_panel.add_list_view("block")
        file_details_panel.add_list_view("time")

        def update_regex(pattern):
            update_arg(self.args, 'filename_regex', pattern)
            parse_filenames()

        panel.add_input(
            "filename regex",
            str,
            value=self.args.filename_regex,
            tooltip="Regex to capture named groups from filename (year, month, day, hour, minute, second, block)",
            on_changed=lambda event: update_regex(event)
        )
        panel.add_checkbox(
            "detect layout",
            checked=self.args.detect_layout,
            on_checked=lambda event: update_arg(self.args, "detect_layout", panel.get_value('detect layout')),
            tooltip="Use configuration to detect layout"
        )
        panel.add_path_select(
            "fixed layout",
            on_add=self._on_select_layout,
            on_remove=self._on_remove_layout,
            tooltip="Fixed layout, layout detection will not be applied if this is specified",
            value=self.args.fixed_layout
        )
        panel.add_input(
            "processes",
            int,
            value=self.args.processes,
            tooltip="Number of parallel processes to run",
            on_changed=lambda event: update_arg(self.args, "processes", event)
        )
        panel.add_combobox(
            "debugging images",
            [i.name for i in DebugEnum],
            lambda s, i: update_arg(self.args, "image_debug", s),
            value=self.args.image_debug
        )
        if isMacOS:  # todo work out why comboboxes cause a crash on mac osx, for now just disabling
            panel.comboboxes['debugging images'].enabled = False
            panel.add_label(
                "Image debugging combobox is disabled on OSx\n"
                "Please use a configuration file or launch option"
            )
        panel.add_path_select(
            "output directory",
            on_add=self._on_select_output_directory,
            on_remove=self._on_remove_output_directory,
            tooltip="Output path, area.tsv will be created (or appended to) in this folder",
            value=str(Path(self.args.out_dir).resolve())
        )
        panel.add_button("Calculate", on_clicked=self.calculate_area)

        panel.visible = False
        return panel

    def calculate_area(self, _event=None):
        if configuration_complete(self.args):
            # todo add progress bar or similar
            try:
                calculate_area(self.args)
            except Exception as e:
                self.window.show_message_box("Error", f"Exception: {e}")
        else:
            self.window.show_message_box("Error", "Configuration is incomplete")

    def calculate_rgr(self, _event=None):
        if not self.args.area_file:
            self.window.show_message_box("Error", f"No area file selected")
        elif not self.args.area_file.is_file():
            self.window.show_message_box("Error", f"Area file not found")
        elif not self.args.samples:
            self.window.show_message_box("Error", f"No samples file selected")
        elif not self.args.samples.is_file:
            self.window.show_message_box("Error", f"Samples file not found")
        else:
            try:
                analyse(self.args)
            except Exception as e:
                self.window.show_message_box("Error", f"Exception: {e}")

    def _on_select_image_directory(self):
        logger.debug("Launch image selection directory dialog")
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose a directory", self.window.theme)
        dlg.tooltip = "Select a directory of images"
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_select_image_dialog_done)
        self.window.show_dialog(dlg)

    def _on_select_image_file(self):
        logger.debug("Launch image selection file dialog")
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose a file", self.window.theme)
        dlg.tooltip = "Select an image file"
        extensions = [f".{s}" for s in IMAGE_EXTENSIONS]
        dlg.add_filter(" ".join(extensions), f"Supported image files ({', '.join(extensions)}")
        dlg.add_filter("", "All files")
        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_select_image_dialog_done)
        self.window.show_dialog(dlg)

    def _on_select_image_dialog_done(self, path: str):
        self.window.close_dialog()

        path = Path(path)
        if path.is_dir():
            files = list()
            for ext in IMAGE_EXTENSIONS:
                files.extend(path.glob(f"*.{ext}"))
        elif path.is_file():
            files = [path]
        else:
            raise FileNotFoundError
        files = [str(f) for f in files]
        self.area_panel.add_to_list("filename", files)
        self.window.set_needs_layout()

    def _on_select_layout(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose layout file to load", self.window.theme)
        extensions = [f".csv"]
        dlg.add_filter(" ".join(extensions), f"Layout files ({', '.join(extensions)}")
        dlg.add_filter("", "All files")
        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_select_layout_dialog_done)
        self.window.show_dialog(dlg)

    def _on_select_layout_dialog_done(self, filename):
        filepath = Path(filename)
        self.window.close_dialog()
        self.load_layout(filepath)

    def load_layout(self, filepath: Optional[Path] = None):
        logger.info("Load fixed layout file")
        if filepath is None:
            if self.args.fixed_layout is None:
                return
            else:
                self.area_panel.set_value('detect layout', False)
        else:
            update_arg(self.args, "fixed_layout", filepath)

        if self.image is not None:
            layout_loader = LayoutLoader(self.image)
            layout = layout_loader.get_layout()
            self.change_layout(layout)
        self.area_panel.path_select_labels['fixed layout'].text = str(self.args.fixed_layout) if self.args.fixed_layout else ""
        # todo should parse the file to ensure it is valid, this will probably break with exceptions using
        # e.g. when change image dimensions with existing fixed layout
        self.set_menu_enabled()
        self.window.set_needs_layout()

    def _on_remove_layout(self):
        logger.info("Remove fixed layout file")
        update_arg(self.args, "fixed_layout", None)
        self.area_panel.path_select_labels['fixed layout'].text = str(self.args.fixed_layout) if self.args.samples else ""
        self.set_menu_enabled()
        self.window.set_needs_layout()

    def _on_select_output_directory(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose output directory", self.window.theme)
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_select_output_directory_dialog_done)
        self.window.show_dialog(dlg)

    def _on_select_output_directory_dialog_done(self, path):
        folder = Path(path).resolve()
        update_arg(self.args, "out_dir", folder)
        self.area_panel.path_select_labels['output directory'].text = str(self.args.out_dir) if self.args.out_dir else ""
        self.rgr_panel.path_select_labels['output directory'].text = str(self.args.out_dir) if self.args.out_dir else ""
        self.window.set_needs_layout()
        self.window.close_dialog()

    def _on_remove_output_directory(self):
        update_arg(self.args, "out_dir", None)
        self.area_panel.path_select_labels['output directory'].text = str(self.args.out_dir) if self.args.samples else ""
        self.rgr_panel.path_select_labels['output directory'].text = str(self.args.out_dir) if self.args.samples else ""
        self.set_menu_enabled()
        self.window.set_needs_layout()

    def get_rgr_panel(self):
        logger.debug("Prepare RGR panel")
        panel = Panel(gui.Vert(spacing=self.em, margins=gui.Margins(self.em)), self.window)
        panel.add_label("Growth rate analysis", self.fonts['large'])
        panel.add_path_select(
            "area file",
            on_add=self._on_select_area,
            on_remove=self._on_remove_area,
            tooltip="AlGrow area file",
            value=str(self.args.area_file.resolve() if self.args.area_file and self.args.area_file.is_file() else "")
        )
        panel.add_path_select(
            "samples map",
            on_add=self._on_select_samples,
            on_remove=self._on_remove_samples,
            tooltip="Sample ID map, used to establish sample groups",
            value=str(self.args.samples.resolve() if self.args.samples else "")
        )
        panel.add_path_select(
            "output directory",
            on_add=self._on_select_output_directory,
            on_remove=self._on_remove_output_directory,
            tooltip="Output path, area.tsv will be created (or appended to) in this folder",
            value=str(Path(self.args.out_dir).resolve())
        )
        panel.add_input(
            "Start day",
            float,
            value=self.args.fit_start,
            tooltip="First timepoint to consider in the fit (in days relative to first timepoint)",
            on_changed=lambda event: update_arg(self.args, "fit_start", event)
        )
        panel.add_input(
            "End day",
            float,
            value=self.args.fit_end,
            tooltip="Last timepoint to consider in the fit (in days relative to first timepoint)",
            on_changed=lambda event: update_arg(self.args, "fit_end", event)
        )
        panel.add_button("Calculate", on_clicked=self.calculate_rgr)
        panel.visible = False

        return panel

    def _on_select_samples(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose samples descriptor file to load", self.window.theme)
        extensions = [f".csv"]
        dlg.add_filter(" ".join(extensions), f"Samples files ({', '.join(extensions)})")
        dlg.add_filter("", "All files")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_select_samples_dialog_done)
        self.window.show_dialog(dlg)

    def _on_select_samples_dialog_done(self, filename):
        filepath = Path(filename)
        self.window.close_dialog()
        self.load_samples(filepath)

    def load_samples(self, filepath):
        logger.info("Load samples descriptor file")
        update_arg(self.args, "samples", filepath)
        self.rgr_panel.path_select_labels['samples map'].text = str(self.args.samples) if self.args.samples else ""
        # todo should parse the file to ensure it is valid
        self.set_menu_enabled()
        self.window.set_needs_layout()

    def _on_remove_samples(self):
        logger.info("Remove samples descriptor file")
        update_arg(self.args, "samples", None)
        self.rgr_panel.path_select_labels['samples map'].text = str(self.args.samples) if self.args.samples else ""
        self.set_menu_enabled()
        self.window.set_needs_layout()

    def _on_select_area(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose area file to load", self.window.theme)
        extensions = [f".csv"]
        dlg.add_filter(" ".join(extensions), f"Area files ({', '.join(extensions)})")
        dlg.add_filter("", "All files")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_select_area_dialog_done)
        self.window.show_dialog(dlg)

    def _on_select_area_dialog_done(self, filename):
        filepath = Path(filename)
        self.window.close_dialog()
        self.load_area(filepath)

    def load_area(self, filepath):
        logger.info("Load area file")
        update_arg(self.args, "area_file", filepath)
        self.rgr_panel.path_select_labels['area file'].text = str(self.args.area_file) if self.args.area_file else ""
        # todo should parse the file to ensure it is valid
        self.set_menu_enabled()
        self.window.set_needs_layout()

    def _on_remove_area(self):
        logger.info("Remove area file")
        update_arg(self.args, "area_file", None)
        self.rgr_panel.path_select_labels['area file'].text = str(self.args.area_file) if self.args.area_file else ""
        self.set_menu_enabled()
        self.window.set_needs_layout()

    def hide_all(self):
        self.info.visible = False
        self.lab_widget.visible = False
        self.image_widget.visible = False
        self.tool_layout.visible = True

        self.target_panel.visible = False
        self.scale_panel.visible = False
        self.circle_panel.visible = False
        self.layout_panel.visible = False
        self.area_panel.visible = False
        self.rgr_panel.visible = False

    def start_target(self):
        self.activity = Activities.TARGET
        self.hide_all()

        self.info.visible = True
        self.lab_widget.visible = True
        self.image_widget.visible = True
        self.tool_layout.visible = True
        self.target_panel.visible = True

        self.image.prepare_cloud()
        self.setup_lab_axes()
        self.update_lab_widget()
        self.update_image_widget()
        self.window.set_needs_layout()

    def start_scale(self):
        self.activity = Activities.SCALE
        self.hide_all()

        self.info.visible = True
        self.image_widget.visible = True
        self.tool_layout.visible = True
        self.scale_panel.visible = True

        self.update_image_widget()
        self.window.set_needs_layout()

    def start_circle(self):
        self.activity = Activities.CIRCLE
        self.hide_all()

        self.info.visible = True
        self.image_widget.visible = True
        self.tool_layout.visible = True
        self.circle_panel.visible = True

        self.update_image_widget()
        self.window.set_needs_layout()

    def start_layout(self):
        self.activity = Activities.LAYOUT
        self.hide_all()

        self.info.visible = True
        self.image_widget.visible = True
        self.tool_layout.visible = True
        self.layout_panel.visible = True

        if self.image.layout is not None:
            self.layout_panel.buttons["save fixed layout"].enabled = True
            self.update_image_with_array(self.image.layout_overlay)
        else:
            self.layout_panel.buttons["save fixed layout"].enabled = False
            self.update_image_widget()
        self.window.set_needs_layout()

    def start_area(self):
        self.activity = Activities.AREA
        self.hide_all()
        self.area_panel.visible = True
        self.window.set_needs_layout()

    def start_rgr(self):
        self.activity = Activities.RGR
        self.hide_all()
        self.rgr_panel.visible = True
        self.window.set_needs_layout()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        if self.activity == Activities.NONE:
            self.background_widget.visible = True
            self.background_widget.frame = gui.Rect(r.x, r.y, r.width, r.height)
        else:
            self.background_widget.visible = False

        if self.activity == Activities.AREA:
            self.area_panel.layout.frame = gui.Rect(r.x, r.y, r.width, r.height)
        elif self.activity == Activities.RGR:
            self.rgr_panel.layout.frame = gui.Rect(r.x, r.y, r.width, r.height)

        if self.image is not None:
            tool_pref = self.tool_layout.calc_preferred_size(layout_context, gui.Widget.Constraints())
            tool_width = tool_pref.width * self.tool_layout.visible

            toolbar_constraints = gui.Widget.Constraints()
            toolbar_constraints.width = tool_width
            info_pref = self.info.calc_preferred_size(layout_context, toolbar_constraints)

            self.info.frame = gui.Rect(
                r.x,
                r.get_bottom()-info_pref.height,
                tool_width if self.tool_layout.visible else r.width,
                info_pref.height
            )
            self.lab_widget.frame = gui.Rect(r.x, r.y, tool_width, tool_width)

            height_used = (
                    self.info.frame.height * self.info.visible +
                    self.lab_widget.frame.height * self.lab_widget.visible
            )
            if self.lab_widget.visible:
                tool_start_y = self.lab_widget.frame.get_bottom()
            else:
                tool_start_y = r.y
            self.tool_layout.frame = gui.Rect(
                r.x,
                tool_start_y,
                tool_width,
                r.height - height_used
            )

            image_start_x = self.tool_layout.frame.get_right() * self.tool_layout.visible
            self.image_widget.frame = gui.Rect(image_start_x, r.y, r.width - image_start_x, r.height)

    def on_mouse_lab_widget(self, event):

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.SHIFT):
            logger.debug("Shift-click occurred on lab widget")

            def depth_callback(depth_image):
                logger.debug("Depth callback running")
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x - self.lab_widget.frame.x
                y = event.y - self.lab_widget.frame.y
                logger.debug(f"Lab widget coords: {x, y}")

                sphere_lab = None
                nearest_index = None

                # have a reasonable selection radius and just get the point closest to the camera within that
                def nearest_within(radius):
                    logger.debug(f"Find nearest coords in depth image within {radius} of clicked")
                    image = np.asarray(depth_image)
                    ys = np.arange(0, image.shape[0])
                    xs = np.arange(0, image.shape[1])
                    mask = (xs[np.newaxis, :]-x)**2 + (ys[:, np.newaxis]-y)**2 >= radius**2
                    masked_array = np.ma.masked_array(image, mask=mask)
                    nearest = masked_array.argmin(fill_value=1)
                    # argmin returns 0 if masked as it is an integer array, just need to make sure it isn't in mask
                    if mask.reshape(-1)[nearest]:
                        # if in mask just return the original coords, we found nothing there anyway
                        return x, y
                    logger.debug(f"nearest index: {nearest}")
                    coords = np.unravel_index(nearest, image.shape)[::-1]
                    return coords

                x, y = nearest_within(10)
                logger.debug(f"Search for object at coords: {x, y}")
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth != 1.0:  # depth is 1.0 if clicked on nothing (i.e. the far plane)
                    logger.debug("Get Lab coords from x,y")
                    world = self.lab_widget.scene.camera.unproject(
                        x, y, depth, self.lab_widget.frame.width,
                        self.lab_widget.frame.height
                    )
                    # first check if there is a sphere at these coords in priors (i.e. won't be found in cloud)

                    logger.debug(f"Checking for sphere at {world}")
                    for lab in self.prior_lab:
                        logger.debug(f"Existing prior: {lab}")
                        logger.debug(f"Distance from world: {lab-world}")
                        if all(abs(lab - world) <= self.args.voxel_size):
                            # i.e. near to within the sphere radius (a grid and expanded but seems close enough for now)
                            sphere_lab = lab
                            logger.debug(f"Match within sphere radius of : {lab}")
                            break
                    if sphere_lab is None:
                        logger.debug(f"No sphere found at {world}, looking for closest point")
                        # otherwise get the nearest point from the cloud to these world coords
                        nearest_index = self.get_nearest_index_from_coords(world)
                else:
                    logger.debug("Nothing found at this point")
                # This is not called on the main thread, so we need to
                # post to the main thread to safely access UI items.

                def update_selected():
                    logger.debug(f"Update selected")
                    if sphere_lab is not None:
                        logger.debug(f"Update sphere at {sphere_lab}")
                        self.remove_prior(sphere_lab)
                    elif nearest_index is not None:
                        logger.debug(f"Update voxel: {nearest_index}")
                        self.toggle_voxel(nearest_index)

                self.app.post_to_main_thread(self.window, update_selected)

            self.lab_widget.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def get_nearest_index_from_coords(self, coords):
        logger.debug("get nearest index from coordinates in Lab")
        single_point_cloud = o3d.geometry.PointCloud()
        single_point_cloud.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector([coords]))
        dist = self.image.cloud.compute_point_cloud_distance(single_point_cloud)
        voxel_index = np.argmin(dist)
        logger.debug(f"Voxel found: {voxel_index}")
        return voxel_index

    def clear_selection(self, event=None):
        if self.image is None or self.image.cloud is None or not self.image.selected_voxel_indices:
            return
        selected_points = np.asarray(self.image.cloud.select_by_index(list(self.image.selected_voxel_indices)).points)

        for lab in selected_points:
            self.remove_sphere(lab)

        self.image.selected_voxel_indices.clear()
        self.target_panel.button_pools['selection'].clear()

        self.update_hull()
        self.update_image_widget()

    def clear_priors(self, event=None):
        if self.image is None or not self.prior_lab:
            return

        for lab in self.prior_lab:
            self.remove_sphere(lab)
        self.prior_lab.clear()
        self.target_panel.button_pools['priors'].clear()
        self.update_hull()
        self.update_image_widget()

    def reduce_selection(self, event=None):
        if self.hull_holder is not None and self.hull_holder.mesh is not None:
            hull_vertices = self.hull_holder.hull.vertices
            to_remove = list()
            for i in self.image.selected_voxel_indices:
                lab = self.image.cloud.points[i]
                if lab not in hull_vertices:
                    lab_text = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
                    self.lab_widget.scene.remove_geometry(lab_text)
                    self.target_panel.button_pools['selection'].remove_button(i)
                    to_remove.append((i, lab))
            logger.debug(f"Removing selected points: {len(to_remove)}")
            for i, lab in to_remove:
                self.image.selected_voxel_indices.remove(i)
                self.remove_sphere(lab)

    def reduce_priors(self, event=None):
        if self.hull_holder is not None and self.hull_holder.mesh is not None:
            hull_vertices = self.hull_holder.hull.vertices

            to_remove = list()
            for lab in self.prior_lab:
                if lab not in hull_vertices:
                    logger.debug(f"removing {lab}")
                    lab_text = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
                    self.target_panel.button_pools['priors'].remove_button(lab_text)
                    to_remove.append(lab)
            logger.debug(f"Removing selected points: {len(to_remove)}")
            for lab in to_remove:
                self.prior_lab.remove(lab)
                self.remove_sphere(lab)
        self.update_hull()

    def clear_circle_selection_and_save(self, _event=None):
        if self.image is None:
            return
        self.image.selected_circle_indices.clear()
        self.circle_panel.button_pools['selection'].clear()
        self.save_circle_colour()  # just this is different from the below
        self.update_displayed_circle_colour()
        self.update_image_widget()

    def clear_circle_selection(self):
        if self.image is None:
            return
        self.image.selected_circle_indices.clear()
        self.circle_panel.button_pools['selection'].clear()
        self.update_displayed_circle_colour()
        self.update_image_widget()

    def save_circle_colour(self, event=None):
        if self.image and self.image.selected_circle_indices:
            circle_lab = self.image.lab.reshape(-1, 3)[list(self.image.selected_circle_indices)]
            circle_lab = tuple(np.median(circle_lab, axis=0))
            update_arg(self.args, "circle_colour", circle_lab)
            self.set_menu_enabled()
        else:
            update_arg(self.args, "circle_colour", None)

    def update_scale(self, event=None):
        px = self.scale_panel.get_value('px')
        mm = self.scale_panel.get_value('mm')
        if px and mm:
            scale = float(np.around(px / mm, decimals=4))
            self.scale_panel.set_value("scale", scale)
            self.set_scale()

    def set_scale(self, event=None):
        logger.debug("save scale to args")
        scale = self.scale_panel.get_value("scale")
        update_arg(self.args, "scale", scale)
        self.set_menu_enabled()

    def update_alpha(self, event=None):
        alpha = self.target_panel.get_value("alpha")
        update_arg(self.args, "alpha", self.target_panel.get_value("alpha"))
        if self.hull_holder is not None:
            self.hull_holder.update_alpha(alpha)
            self.update_hull()
        self.update_target()

    def update_delta(self, event=None):
        delta = self.target_panel.get_value("delta")
        update_arg(self.args, "delta", delta)
        self.update_target()

    def update_fill(self, event=None):
        fill = self.target_panel.get_value("fill")
        update_arg(self.args, "fill", fill)
        self.update_target()

    def update_remove(self, event=None):
        remove = self.target_panel.get_value("remove")
        update_arg(self.args, "remove", remove)
        self.update_target()

    def image_widget_to_image_coords(self, event_x, event_y):
        logger.debug(f"event x,y: {event_x, event_y}")
        logger.debug(f"frame x,y: {self.image_widget.frame.x, self.image_widget.frame.y}")
        # frame spacing depends on image ratio
        # due to scaling, when image height is less than frame height there is a margin equally split
        frame_width = self.image_widget.frame.width
        frame_height = self.image_widget.frame.height
        logger.debug(f"Frame dimensions: {frame_width, frame_height}")
        frame_ratio = frame_height / frame_width
        image_ratio = self.image.height / self.image.width
        logger.debug(f"Image ratio: {image_ratio}")
        if image_ratio > frame_ratio:
            # i.e. space in width
            logger.debug(f"frame width: {frame_width}")
            frame_image_width = frame_height / image_ratio
            logger.debug(f"Framed image width: {frame_image_width}")
            frame_space_width = frame_width - frame_image_width
            x_offset = frame_space_width/2
            logger.debug(f"x offset: {x_offset}")
        else:
            x_offset = 0
            frame_image_width = frame_width
        if image_ratio < frame_ratio:
            # i.e. space in height
            logger.debug(f"frame height: {frame_height}")
            frame_image_height = frame_width * image_ratio
            logger.debug(f"Framed image height: {frame_image_height}")
            frame_space_height = frame_height - frame_image_height
            y_offset = frame_space_height/2
            logger.debug(f"y offset: {y_offset}")
        else:
            y_offset = 0
            frame_image_height = frame_height

        frame_x = event_x - self.image_widget.frame.x
        frame_y = event_y - self.image_widget.frame.y
        logger.debug(f"frame coords: {frame_x, frame_y}")
        framed_image_x = frame_x - x_offset
        framed_image_y = frame_y - y_offset
        logger.debug(f"frame image coords: {framed_image_x, framed_image_y}")
        image_fraction_x = framed_image_x / frame_image_width
        image_fraction_y = framed_image_y / frame_image_height
        #displayed_image_x = np.floor(self.image.width * image_fraction_x)
        displayed_image_x = self.image.width * image_fraction_x
        #displayed_image_y = np.floor(self.image.height * image_fraction_y)
        displayed_image_y = self.image.height * image_fraction_y
        logger.debug(f"displayed coords: {displayed_image_x, displayed_image_y}")
        image_x = np.floor((displayed_image_x*self.image.zoom_factor)) + self.image.displayed_start_x
        image_y = np.floor((displayed_image_y*self.image.zoom_factor)) + self.image.displayed_start_y
        logger.debug(f"image coords: {image_x, image_y}")
        x = int(image_x)
        y = int(image_y)
        # todo still not quite perfect, particularly at very high zoom levels
        #  maybe a rounding error/indexing error or something to do with using fractions?
        # it is workable mostly though...
        return x, y

    def on_mouse_image_widget(self, event):

        if event.type in [
            gui.MouseEvent.Type.BUTTON_DOWN,
            gui.MouseEvent.Type.BUTTON_UP,
            gui.MouseEvent.Type.WHEEL
        ]:
            x, y = self.image_widget_to_image_coords(event.x, event.y)
            if not all([x >= 0, x < self.image.width, y >= 0, y < self.image.height]):
                return gui.Widget.EventCallbackResult.IGNORED

            if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                    gui.KeyModifier.SHIFT):

                if self.activity == Activities.TARGET:
                    voxel_index = self.image.voxel_map[y, x]
                    if voxel_index >= 0:  # -1 used for mask
                        self.toggle_voxel(voxel_index)

                elif self.activity == Activities.CIRCLE:
                    pixel_index = self.image.coord_to_pixel(x, y)
                    self.toggle_circle_pixel(pixel_index)

                elif self.activity == Activities.SCALE:
                    line_colour = self.scale_panel.get_value("line colour")
                    distance = self.draw_line(x, y, line_colour)
                    if distance is not None:
                        self.scale_panel.set_value("px", distance)
                        self.update_scale()
                        self.measure_start = None

                elif self.activity == Activities.LAYOUT:
                    line_colour = self.layout_panel.get_value("line colour")
                    distance = self.draw_line(x, y, line_colour)
                    if distance is not None:
                        self.layout_panel.set_value("px", distance)
                        self.update_scale()
                        self.measure_start = None

            elif event.type == gui.MouseEvent.Type.WHEEL:
                # only attempt to zoom if not already at limit
                if event.wheel_dy > 0 and self.image.zoom_index == len(self.image.divisors) - 1:
                    pass
                elif event.wheel_dy < 0 and self.image.zoom_index == 0:
                    pass
                else:
                    self.zoom_image(x, y, event.wheel_dy)

            elif event.type == gui.MouseEvent.BUTTON_DOWN:
                self.drag_start = (x, y)

            elif event.type == gui.MouseEvent.BUTTON_UP and self.drag_start is not None:
                drag_end = (x, y)
                drag_dif = np.array(self.drag_start) - np.array(drag_end)
                drag_x = drag_dif[0]
                drag_y = drag_dif[1]
                if drag_x or drag_y:
                    self.image.drag(drag_x, drag_y)
                    self.update_image_widget()
                self.drag_start = None

            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def draw_line(self, x, y, line_colour) -> Optional[float]:  # returns the length of the line if drawn else None
        logger.debug("Draw line")
        if self.measure_start is None:
            self.measure_start = (x, y)
            self.image_widget.update_image(self.image.get_displayed_with_disk(x, y, line_colour))
            return None
        else:
            measure_end = (x, y)
            distance = float(np.around(np.linalg.norm(
                np.array(self.measure_start) - np.array(measure_end)
            ), decimals=1))
            logger.debug(f"distance:{distance}")
            self.image_widget.update_image(
                self.image.get_displayed_with_line(*self.measure_start, *measure_end, line_colour)
            )
            return distance

    def zoom_image(self, x, y, dy):
        logger.debug(f"Zooming at {x, y}, zoom factor {self.image.zoom_factor}")
        self.image.zoom(x, y, dy)
        self.update_image_widget()

    def toggle_circle_pixel(self, pixel_index):
        logger.debug("toggle circle pixel")
        lab = self.image.lab.reshape(-1, 3)[pixel_index]
        rgb = self.image.rgb.reshape(-1, 3)[pixel_index]
        if pixel_index in self.image.selected_circle_indices:
            logger.debug(f"remove pixel index: {pixel_index}")
            self.image.selected_circle_indices.remove(pixel_index)
            self.circle_panel.button_pools['selection'].remove_button(pixel_index)
        else:
            logger.debug(f"add pixel index: {pixel_index}")
            self.image.selected_circle_indices.add(pixel_index)
            self.add_circle_button(pixel_index, lab, rgb)
        self.save_circle_colour()
        self.update_displayed_circle_colour()
        self.update_image_widget()

    def toggle_voxel(self, voxel_index):
        logger.debug(f"Toggle voxel: {voxel_index}")
        selected_lab = self.image.cloud.points[voxel_index]
        selected_rgb = self.image.cloud.colors[voxel_index]
        if voxel_index in self.image.selected_voxel_indices:
            logger.debug(f"Remove: {selected_lab}")
            self.image.selected_voxel_indices.remove(voxel_index)
            self.remove_sphere(selected_lab)
            self.target_panel.button_pools['selection'].remove_button(voxel_index)
        else:
            logger.debug(f"Add: {selected_lab}")
            self.image.selected_voxel_indices.add(voxel_index)
            self.add_sphere(selected_lab, selected_rgb)
            self.add_voxel_button(voxel_index, selected_lab, selected_rgb)
        logger.debug("get selected points to update hull holder")
        self.update_hull()

    def add_voxel_button(self, lab_index: int, lab, rgb):
        key = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
        b = gui.Button(key)
        b.background_color = gui.Color(*rgb)
        self.target_panel.button_pools['selection'].add_button(lab_index, b)
        b.set_on_clicked(lambda: self.toggle_voxel(lab_index))

    def add_circle_button(self, pixel_index: int, lab, rgb):
        key = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
        b = gui.Button(key)
        b.background_color = gui.Color(*rgb)
        self.circle_panel.button_pools['selection'].add_button(pixel_index, b)
        b.set_on_clicked(lambda: self.toggle_circle_pixel(pixel_index))

    def update_displayed_circle_colour(self):
        pool = self.circle_panel.button_pools['circle colour']
        lab = self.args.circle_colour
        pool.clear()
        if lab is not None:
            b = gui.Button('circle colour')
            b.background_color = gui.Color(*lab2rgb(lab))
            pool.add_button('circle colour', b)

    def add_sphere(self, lab, rgb):
        logger.debug(f"Adding sphere for {lab}")
        key = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
        sphere = o3d.t.geometry.TriangleMesh().create_sphere(radius=self.args.voxel_size/2)
        sphere.compute_vertex_normals()
        triangle_colours = np.tile(rgb, (sphere.triangle.indices.shape[0], 1))
        sphere.triangle["colors"] = o3d.core.Tensor(triangle_colours)
        sphere.translate(lab)
        self.lab_widget.scene.add_geometry(key, sphere, self.mesh_material)

    def remove_sphere(self, lab):
        key = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
        self.lab_widget.scene.remove_geometry(key)

    def add_prior(self, lab: tuple[float, float, float]):
        self.prior_lab.add(lab)
        rgb = lab2rgb([lab])[0]
        self.add_sphere(lab, rgb)
        key = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
        b = gui.Button(key)
        b.background_color = gui.Color(*rgb)
        self.target_panel.button_pools['priors'].add_button(key, b)
        b.set_on_clicked(lambda: self.remove_prior(lab))

    def remove_prior(self, lab):
        key = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
        self.target_panel.button_pools['priors'].remove_button(key)
        self.remove_sphere(lab)
        self.prior_lab.remove(lab)
        self.update_hull()
        self.update_image_widget()

    def update_hull(self):
        logger.debug("Update hull in GUI")
        if self.image.selected_voxel_indices:
            points = self.image.cloud.select_by_index(list(self.image.selected_voxel_indices)).points
            points.extend(self.prior_lab)
            points = np.asarray(points)
        else:
            points = list(self.prior_lab)

        logger.debug(f"Update hull points {len(points)}")
        if len(points) > 0:
            update_arg(self.args, "hull_vertices", list(map(tuple, points)))
            self.hull_holder = HullHolder(points, self.target_panel.get_value("alpha"))
            self.hull_holder.update_hull()
        else:
            self.hull_holder = None
        self.draw_hull()
        self.update_target()
        self.set_menu_enabled()

    def update_target(self, _event=None):
        if self.target_panel.buttons['show target'].is_on:
            logger.debug("Update target highlighting (and dice coefficient if true mask is provided)")
            #self.image.target_mask, dice = self.get_target_mask()
            self.image.target_mask = self.get_target_mask_from_voxels()
            dice = self.calculate_dice(self.image.target_mask)
            self.update_info(dice)
        self.update_image_widget()

    def calculate_dice(self, target_mask):
        if self.image.true_mask is None or target_mask is None:
            return None
        else:
            logger.debug("Compare target mask to true mask")

            true_mask = self.image.true_mask.mask
            tp = np.sum(true_mask[target_mask])
            tn = np.sum(~true_mask[~target_mask])
            fp = np.sum(~true_mask[target_mask])
            fn = np.sum(true_mask[~target_mask])
            sensitivity = (tp / (tp + fn))
            specificity = (tn / (tn + fp))
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            dice = 2 * tp / ((2 * tp) + fp + fn)
            logger.info(
                f"Dice coefficient: {dice}, "
                f"Accuracy: {accuracy}, "
                f"Sensitivity: {sensitivity}, "
                f"Specificity: {specificity}"
            )
            return dice

    def get_target_mask_from_voxels(self):
        if self.hull_holder is None or self.hull_holder.mesh is None or self.hull_holder.scene is None:
            logger.debug("No target hull prepared")
            return None

        logger.debug("Prepare target mask")
        # first detect occupancy, we then only need to calculate distances for those outside
        occupancy = self.hull_holder.get_occupancy(np.asarray(self.image.cloud.points))
        logger.debug("Create distances array, where occupants are 0 and outside are 1")
        inverted_occupancy = ~occupancy  # have to do this step first, can't stream invert
        distances = inverted_occupancy.astype('float')
        if self.args.delta != 0:
            logger.debug(f"Calculate distance from hull surface for points outside")
            distances[~occupancy] = self.hull_holder.get_distances(np.asarray(self.image.cloud.points)[~occupancy])

        logger.debug(f"Use distances to create a target mask")
        distances = distances[self.image.voxel_map]
        target_mask = distances <= self.target_panel.get_value("delta")
        # fix for -1 indices representing the mask  # todo this is still clunky, consider refactoring
        target_mask[self.image.voxel_map == -1] = 0

        if self.target_panel.get_value("fill"):
            logger.debug("Fill small holes")
            target_mask = remove_small_holes(target_mask, self.target_panel.get_value("fill"))
        if self.target_panel.get_value("remove"):
            logger.debug("Remove small objects")
            target_mask = remove_small_objects(target_mask, self.target_panel.get_value("remove"))

        return target_mask

    #def get_target_mask(self): This is the old version where I didn't compute distance from voxels
    # keeping it as I might want to provide an option (slower) to keep doing this as it is more consistent
    # with the way it is handled in the final analysis.
    #    if self.hull_holder is None or self.hull_holder.mesh is None:
    #        logger.debug("No target hull prepared")
    #        return None, None
    #
    #    logger.debug("Prepare target mask")
    #    logger.debug(f"Calculate distance from hull")
    #    if self.image.layout is None:
    #        distances = self.hull_holder.get_distances(self.image.lab)
    #    else:
    #        distances = self.hull_holder.get_distances(self.image.lab[self.image.layout_mask])
    #
    #    if distances is not None:
    #        logger.debug(f"Use distances to create a target mask")
    #        target = distances <= self.target_panel.get_value("delta")
    #        logger.debug(f"Target pixels {np.sum(target)}")
    #        if self.image.layout is None:
    #            target_mask = target.reshape(self.image.lab.shape[:2])
    #        else:
    #            target_mask = np.zeros(self.image.lab.shape[:2], dtype='bool')
    #            target_mask[self.image.layout_mask] = target
    #
    #        if self.target_panel.get_value("fill"):
    #            logger.debug("Fill small holes")
    #            target_mask = remove_small_holes(target_mask, self.target_panel.get_value("fill"))
    #        if self.target_panel.get_value("remove"):
    #            logger.debug("Remove small objects")
    #            target_mask = remove_small_objects(target_mask, self.target_panel.get_value("remove"))
    #        return target_mask

    def toggle_show_target(self):
        if self.target_panel.buttons['show target'].is_on:
            self.update_target()
        else:
            self.update_image_widget()

    def toggle_show_selected(self):
        self.update_image_widget()

    def update_image_widget(self, _event=None):
        logger.debug(f"update image widget")
        if self.activity == Activities.TARGET:
            if self.target_panel.buttons['show target'].is_on:
                target_colour = self.target_panel.get_value("target colour")
            else:
                target_colour = None
            if self.target_panel.buttons['show selected'].is_on:
                selected_colour = self.target_panel.get_value("selection colour")
            else:
                selected_colour = None
            self.image_widget.update_image(
                self.image.get_displayed_with_target(
                    target_colour=target_colour,
                    selected_colour=selected_colour
                )
            )
        elif self.activity == Activities.CIRCLE:
            self.image_widget.update_image(self.image.get_displayed_as_circle_distance())
        elif self.activity == Activities.LAYOUT:
            self.image_widget.update_image(self.image.get_displayed_with_layout())
        else:
            self.image_widget.update_image(self.image.get_displayed_with_target())
        logger.debug("Image update complete")
        return gui.Widget.EventCallbackResult.HANDLED

    def update_image_with_array(self, array):
        logger.debug(f"update image widget with an array")
        displayed = self.image.apply_zoom(array)
        to_render = o3d.geometry.Image(displayed.astype(np.float32))
        self.image_widget.update_image(to_render)
        return gui.Widget.EventCallbackResult.HANDLED

    def update_info(self, dice=None):
        if self.image is None:
            self.info.text = ""
        elif self.image.true_mask is None:
            self.info.text = f"Image file: {str(self.image.filepath.name)}\n"
        elif self.image.true_mask is not None and dice is None:
            self.info.text = (
                f"Image file: {str(self.image.filepath.name)}\n"
                f"Mask file: {str(self.image.true_mask.filepath.name)}\n"
            )
        else:
            self.info.text = (
                f"Image file: {str(self.image.filepath.name)}\n"
                f"Mask file: {str(self.image.true_mask.filepath.name)}\n"
                f"Dice coefficient: {float(dice)}"
            )

    def draw_hull(self, event=None):
        if self.lab_widget.scene is not None:
            logger.debug("Remove existing mesh")
            self.lab_widget.scene.remove_geometry('mesh')
        if all([
            self.target_panel.buttons['show hull'].is_on,
            self.hull_holder is not None and self.hull_holder.mesh is not None
        ]):
            logger.debug("Add mesh to scene")
            hull_colour = self.target_panel.get_value("hull colour")
            self.hull_holder.mesh.compute_vertex_normals()
            triangle_colours = np.tile(hull_colour, (self.hull_holder.mesh.triangle.indices.shape[0], 1))
            self.hull_holder.mesh.triangle["colors"] = o3d.core.Tensor(triangle_colours)
            self.lab_widget.scene.add_geometry("mesh", self.hull_holder.mesh, self.mesh_material)

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose image to load", self.window.theme)
        extensions = [f".{s}" for s in IMAGE_EXTENSIONS]
        dlg.add_filter(" ".join(extensions), f"Supported image files ({', '.join(extensions)}")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_image_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_image_dialog_done(self, filename):
        filepath = Path(filename)
        self.window.close_dialog()
        self.load_image(filepath)

    def _on_menu_mask(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose mask image to load", self.window.theme)
        extensions = [f".{s}" for s in IMAGE_EXTENSIONS]
        dlg.add_filter(" ".join(extensions), f"Supported image files ({', '.join(extensions)}")
        dlg.add_filter("", "All files")
        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_mask_dialog_done)
        self.window.show_dialog(dlg)

    def _on_load_mask_dialog_done(self, filename):
        filepath = Path(filename)
        self.window.close_dialog()
        self.load_true_mask(filepath)
        dice = self.calculate_dice(self.image.target_mask)
        self.update_info(dice)

    def _on_menu_write_conf(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save to", self.window.theme)
        extensions = [f".conf"]
        dlg.add_filter(" ".join(extensions), f"Configuration files ({', '.join(extensions)}")
        dlg.add_filter("", "All files")
        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_write_conf_dialog_done)
        self.window.show_dialog(dlg)

    def _on_write_conf_dialog_done(self, filename):
        filepath = Path(filename)
        self.window.close_dialog()
        self.write_configuration(filepath)

    def _on_menu_load_conf(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose configuration file to load", self.window.theme)
        extensions = [f".conf"]
        dlg.add_filter(" ".join(extensions), f"Configuration files ({', '.join(extensions)}")
        dlg.add_filter("", "All files")
        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_conf_dialog_done)
        self.window.show_dialog(dlg)

    def _on_load_conf_dialog_done(self, filename):
        filepath = Path(filename)
        self.window.close_dialog()
        self.load_configuration(filepath)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(self.em, gui.Margins(self.em, self.em, self.em, self.em))
        dlg_layout.add_child(gui.Label(f"AlGrow {__version__}"))
        open_license = gui.Button("License")
        open_license.set_on_clicked(self._on_license)
        dlg_layout.add_child(open_license)

        open_dep_license = gui.Button("Dependency Licenses")
        open_dep_license.set_on_clicked(self._on_dep_license)
        dlg_layout.add_child(open_dep_license)

        # Add the Ok button. We need to define a callback function to handle the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def _on_license(self):
        # when running a pyinstaller executable, these resources are placed in tmp
        # snaps e.g. mozilla browser don't share the same tmp, see: https://bugs.launchpad.net/snapd/+bug/1972762
        # so we need to copy to a constant path, use the out_dir
        src_path = Path(Path(__file__).parent, "resources", "LICENSE.txt")
        dest_path = Path(Path(self.args.out_dir).resolve(), "AlGrow_LICENSE.txt")
        if not dest_path.is_file():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Copy text to {dest_path}")
            dest_path.write_text(src_path.read_text())
        webbrowser.open(str(dest_path))

    def _on_dep_license(self):
        src_path = Path(Path(__file__).parent, "resources", "dependency_licenses.txt")
        dest_path = Path(Path(self.args.out_dir).resolve(), "AlGrow_dependency_licenses.txt")
        if not dest_path.is_file():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Copy text to {dest_path}")
            dest_path.write_text(src_path.read_text())
        webbrowser.open(str(dest_path))

    def set_menu_enabled(self):
        if self.image is None:
            self.app.menubar.set_enabled(self.MENU_MASK, False)
            self.app.menubar.set_enabled(self.MENU_TARGET, False)
            self.app.menubar.set_enabled(self.MENU_SCALE, False)
            self.app.menubar.set_enabled(self.MENU_CIRCLE, False)
            self.app.menubar.set_enabled(self.MENU_LAYOUT, False)
        else:
            self.app.menubar.set_enabled(self.MENU_MASK, True)
            self.app.menubar.set_enabled(self.MENU_TARGET, True)
            self.app.menubar.set_enabled(self.MENU_SCALE, True)
            self.app.menubar.set_enabled(self.MENU_CIRCLE, True)
            self.app.menubar.set_enabled(self.MENU_LAYOUT, self.args.circle_colour is not None)

        self.app.menubar.set_enabled(self.MENU_AREA, True)
        self.app.menubar.set_enabled(self.MENU_RGR, True)

    def move_selected_to_priors(self):
        if self.image is not None:
            logger.debug("Unload existing image and copy layout if found")
            if self.image.cloud is not None:
                selected_points = np.asarray(self.image.cloud.select_by_index(list(self.image.selected_voxel_indices)).points)
                self.clear_selection()
                # add the previously selected points back as priors for the new plot
                for lab in selected_points:
                    lab = tuple(lab)
                    self.add_prior(lab)

    def load_image(self, path):
        self.move_selected_to_priors()

        logger.debug("Load new image")
        self.image = CalibrationImage(ImageLoaded(path, self.args))

        self.info.visible = True
        self.update_info()

        self.target_panel.buttons['hull from mask'].enabled = False
        self.set_menu_enabled()

        logger.debug("Display new image")
        self.image_widget.visible = True
        self.update_image_widget()
        self.update_lab_widget()

    def load_true_mask(self, path):
        try:
            self.image.true_mask = MaskLoaded(path)
            self.target_panel.buttons['hull from mask'].enabled = True
            self.update_info()
            self.update_image_widget()  # just needed to calculate dice
            self.window.set_needs_layout()
        except Exception as e:
            logger.debug(f"Failed to load mask {e}")
            self.window.show_message_box("Error", f"Failed to load mask {e}")

    def hull_from_mask(self):
        logger.debug("Prepare points from provided mask")
        hh = HullHolder.get_from_mask(
            self.image,
            self.target_panel.get_value("alpha"),
            self.target_panel.get_value("min pixels"),
            self.args.voxel_size
        )
        hh.update_hull()
        if hh is None:
            return None

        elif hh.hull is not None:
            logger.debug("Hull holder from vertices")
            points = hh.hull.vertices
        else:
            logger.debug("Hull holder from points")
            points = hh.points

        hull_vertices_string = f'{[",".join([str(j) for j in i]) for i in points]}'.replace("'", '"')
        logger.debug(f"Hull vertices or points from mask: {hull_vertices_string}")
        for lab in points:
            lab = tuple(lab)
            voxel_index = self.get_nearest_index_from_coords(lab)
            if voxel_index not in self.image.selected_voxel_indices:
                self.image.selected_voxel_indices.add(voxel_index)
                selected_lab = self.image.cloud.points[voxel_index]
                selected_rgb = self.image.cloud.colors[voxel_index]
                self.add_sphere(selected_lab, selected_rgb)
                self.add_voxel_button(voxel_index, selected_lab, selected_rgb)
        self.update_lab_widget()
        self.update_image_widget()

    def setup_lab_axes(self):
        logger.debug("Setup axes")
        self.lab_widget.scene.remove_geometry("lines")
        for label in self.labels:
            self.lab_widget.remove_3d_label(label)
        self.labels.clear()

        bbox = self.image.cloud.get_axis_aligned_bounding_box()
        logger.debug("Setup camera")
        center = bbox.get_center()
        bbox_geom = o3d.geometry.OrientedBoundingBox().create_from_axis_aligned_bounding_box(bbox)
        lineset = o3d.geometry.LineSet().create_from_oriented_bounding_box(bbox_geom)
        to_remove = np.unique([np.asarray(line) for line in np.asarray(lineset.lines) if ~np.any(line == 0)], axis=1)
        [lineset.lines.remove(line) for line in to_remove]
        for i in range(1, 4):
            point = lineset.points[i]
            axis = np.array(["L*", "a*", "b*"])[point != lineset.points[0]]
            label: gui.Label3D = self.lab_widget.add_3d_label(point, str(axis[0]))
            self.labels.append(label)
            label.color = gui.Color(1, 1, 1)

        self.lab_widget.scene.add_geometry("lines", lineset, self.line_material)
        # in the below 60 is default field of view
        self.lab_widget.setup_camera(60, bbox, bbox.get_center())
        self.lab_widget.look_at(center, [-200, 0, 0], [-1, 1, 0])

    def update_lab_widget(self, _event=None):
        if self.activity == Activities.TARGET:
            logger.debug("Update lab widget")
            logger.debug("Filter by pixels per voxel")
            if self.image.cloud is None:
                self.image.prepare_cloud()

            common_indices = [i for i, j in enumerate(self.image.voxel_to_image) if len(j) >= self.target_panel.get_value("min pixels")]
            logger.debug(f"Selected voxels: {len(common_indices)}")
            cloud = self.image.cloud.select_by_index(common_indices)
            # need to remap indices from source cloud due to radius downsampling
            logger.debug(f"cloud size : {len(cloud.points)}")

            # Lab is not visible yet but is still loaded here
            self.lab_widget.scene.remove_geometry("points")

            logger.debug("Add point cloud to scene")
            self.lab_widget.scene.add_geometry("points", cloud, self.point_material)
            self.update_hull()

    def export_image(self, path):
        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self.lab_widget.scene.scene.render_to_image(on_image)

    def write_configuration(self, filepath):
        logger.info("Write out configured parameters")
        with open(filepath, 'w') as text_file:
            for arg in vars(self.args):
                value = getattr(self.args, arg)
                logger.debug(f"Writing arg:{arg}, value:{value}")
                if value is None:
                    continue
                elif isinstance(value, Enum):
                    value = value.name
                elif isinstance(value, Path):
                    value = str(value.resolve())
                if arg == "images":
                    continue  # todo if want to write images then need to fix parser to handle it
                    # if isinstance(value, list):
                    #    value = f"\"{','.join([str(i) for i in value])}\""
                    # else:
                    #     value = str(value)
                if arg == "circle_colour":
                    value = f"\"{','.join([str(i) for i in value])}\""
                if arg == "hull_vertices":
                    value = f'{[",".join([str(j) for j in i]) for i in value]}'.replace(
                    "'", '"')
                text_file.write(f"{arg} = {value}\n")
            logger.debug(f"Finished writing to configuration file {str(filepath)}")

    def load_all_parameters(self):
        logger.debug("Load parameters to GUI")
        if self.args.scale is not None:
            self.scale_panel.set_value("scale", self.args.scale)

        if self.args.alpha is not None:
            self.target_panel.set_value("alpha", self.args.alpha)

        if self.args.delta is not None:
            self.target_panel.set_value("delta", self.args.delta)

        if self.args.fill is not None:
            self.target_panel.set_value("fill", self.args.fill)

        if self.args.remove is not None:
            self.target_panel.set_value("remove", self.args.remove)

        logger.debug("Reset circle colour then load the argument")
        self.clear_circle_selection()
        self.update_displayed_circle_colour()

        logger.debug("Load layout")
        self.load_layout_parameters()

        logger.debug("Clear existing selected hull vertices")
        self.clear_selection()
        logger.debug("Clear existing priors")
        self.clear_priors()
        logger.debug("add points from configuration file as priors")

        if self.args.hull_vertices is not None:
            for lab in self.args.hull_vertices:
                self.add_prior(lab)
        logger.debug("detect layout and prepare lab cloud")

        self.area_panel.set_value("filename regex", self.args.filename_regex)

        if self.args.fixed_layout:
            self.area_panel.path_select_labels['fixed layout'].text = str(Path(self.args.fixed_layout).resolve())
            self.area_panel.checkboxes['detect layout'].checked = False
        else:
            self.area_panel.checkboxes['detect layout'].checked = self.args.detect_layout
        self.area_panel.path_select_labels['output directory'].text = str(Path(self.args.out_dir).resolve()) if self.args.out_dir else ""
        self.area_panel.set_value("processes", self.args.processes)

        if self.image is not None:
            # need to update args associated with image
            self.image.args = self.args
            self.update_image_widget()
            if self.args.fixed_layout is not None:
                # this could be handled better, e.g. keep a previously loaded layout if valid
                self.load_layout()
            # only load, don't detect as it slows things down too much, detect only when prompted
            #elif self.args.detect_layout:
                #self.detect_layout()
            self.image.prepare_cloud()
            self.update_lab_widget()

    def load_configuration(self, filepath):
        logger.info(f"Load configuration parameters from conf file: {str(filepath)}")
        try:
            parser = options(filepath)
            args, _ = parser.parse_known_args()
            #args = parser.parse_args()
            logger.debug("Update args")
            args = postprocess(args)
            update_arg(args, "images", self.args.images)  # we don't update this from the file
            self.args = args
            self.load_all_parameters()
            self.set_menu_enabled()
            self.window.set_needs_layout()
            logger.debug("Configuration loaded")
        except SystemExit:
            logger.warning(f"Failed to load configuration file")
            self.window.show_message_box("Error", "Invalid configuration file")
        except FileExistsError as e:
            self.window.show_message_box("Error", f"The output file already exists in this folder: {e}")
        except FileNotFoundError as e:
            self.window.show_message_box("Error", f"File not found: {e}")
