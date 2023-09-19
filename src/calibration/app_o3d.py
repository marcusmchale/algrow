import logging
import argparse

import open3d as o3d
from open3d.visualization import gui, rendering
import platform
from scipy.ndimage import zoom
from skimage import draw
from skimage.color import lab2rgb, deltaE_cie76, gray2rgb
from skimage.morphology import remove_small_holes, remove_small_objects
import numpy as np
import pandas as pd

from datetime import datetime

from queue import Queue, Empty
from pathlib import Path

import time

from ..options.custom_types import IMAGE_EXTENSIONS
from .hull_o3d import HullHolder, hull_from_mask

from ..image_loading import ImageLoaded
from .loading import CalibrationImage
from ..layout import get_layout, Layout

from .panel import Panel

from enum import Enum

from ..options.update_and_verify import update_arg, layout_defined

isMacOS = (platform.system() == "Darwin")

logger = logging.getLogger(__name__)


class Activities(Enum):
    NONE = 0
    SCALE = 1
    CIRCLE = 2
    LAYOUT = 3
    TARGET = 4


class AppWindow:
    MENU_OPEN = 1
    MENU_MASK = 2
    MENU_SAMPLES = 3
    MENU_QUIT = 7

    MENU_SCALE = 11
    MENU_CIRCLE = 12
    MENU_LAYOUT = 13
    MENU_TARGET = 14
    MENU_WRITE = 15

    MENU_AREA = 21
    MENU_RGR = 22

    MENU_SHOW_SETTINGS = 30
    MENU_ABOUT = 40

    def __init__(self, width, height, fonts, args):
        self.fonts = fonts
        self.args: argparse.Namespace = args
        self.app = gui.Application.instance
        self.window = self.app.create_window("AlGrow Calibration", width, height)
        self.window.set_on_layout(self._on_layout)

        self.image = None
        self.target_mask = None
        self.activity = Activities.NONE

        #selections
        self.circle_indices = set()
        self.target_indices = set()
        self.prior_lab = set()
        self.layout = None
        #
        self.hull_holder = None

        # these are for measuring and moving
        self.drag_start = None
        self.measure_start = None

        # queues are used to free up the main (GUI) thread by performing some actions in the background
        self.zoom_queue = Queue()
        self.update_queue = Queue()
        self.toggle_queue = Queue()

        # For more consistent spacing across systems we use font size rather than fixed pixels
        self.em = self.window.theme.font_size

        # we need a generic material for the 3d plot
        self.point_material = self.get_material("point")
        self.line_material = self.get_material("line")

        # prepare widgets, layouts and panels
        self.labels = list()  # used to store and subsequently remove 3d labels - todo remove by type?
        self.annotations = list()  # used to store and subsequently remove annotatios from image widget

        logo_path = Path(Path(__file__).parent, "bmp", "logo.png")
        logger.debug(f"{logo_path}")
        self.background_widget = gui.ImageWidget(str(logo_path))
        self.window.add_child(self.background_widget)

        self.lab_widget = self.get_lab_widget()
        self.info = self.get_info()
        self.image_widget = self.get_image_widget()
        self.debug_widget = self.get_debug_widget()

        self.tool_layout = gui.Horiz(0.5 * self.em, gui.Margins(0.5 * self.em))
        self.tool_layout.visible = False
        self.window.add_child(self.tool_layout)

        self.scale_panel = self.get_scale_panel()

        self.target_panel = self.get_target_panel()
        self.circle_panel = self.get_circle_panel()

        self.layout_panel = self.get_layout_panel()
        self.load_layout_parameters()

        if self.args.hull_vertices is not None:
            for lab in self.args.hull_vertices:
                self.add_prior(lab)

        self.prepare_menu()

    def prepare_menu(self):
        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_item("Image", AppWindow.MENU_OPEN)
            file_menu.add_item("Mask", AppWindow.MENU_MASK)
            file_menu.add_item("Samples", AppWindow.MENU_SAMPLES)
            file_menu.set_enabled(AppWindow.MENU_MASK, False)
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            else:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            calibration_menu = gui.Menu()
            calibration_menu.add_item("Set scale", AppWindow.MENU_SCALE)
            calibration_menu.set_enabled(AppWindow.MENU_SCALE, False)
            calibration_menu.add_item("Define Circle Colour", AppWindow.MENU_CIRCLE)
            calibration_menu.set_enabled(AppWindow.MENU_CIRCLE, False)
            calibration_menu.add_item("Define Layout", AppWindow.MENU_LAYOUT)
            calibration_menu.set_enabled(AppWindow.MENU_LAYOUT, False)
            calibration_menu.add_item("Define Target Hull", AppWindow.MENU_TARGET)
            calibration_menu.set_enabled(AppWindow.MENU_TARGET, False)
            calibration_menu.add_item("Save Calibration", AppWindow.MENU_WRITE)


            settings_menu = gui.Menu()
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("AlGrow", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Calibration", calibration_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...  # todo add link to pdf with instructions and citation/paper when published
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Calibration", calibration_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        self.window.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        self.window.set_on_menu_item_activated(AppWindow.MENU_MASK, self._on_menu_mask)
        self.window.set_on_menu_item_activated(AppWindow.MENU_WRITE, self._on_menu_write)
        self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        self.window.set_on_menu_item_activated(AppWindow.MENU_SCALE, self.start_scale)
        self.window.set_on_menu_item_activated(AppWindow.MENU_CIRCLE, self.start_circle)
        self.window.set_on_menu_item_activated(AppWindow.MENU_LAYOUT, self.start_layout)
        self.window.set_on_menu_item_activated(AppWindow.MENU_TARGET, self.start_target)
        self.window.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS, self._on_menu_toggle_settings_panel)
        self.window.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

    def get_material(self, key: str):
        logger.debug("Prepare material for {key}")
        if key == "point":
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = 2 * self.window.scaling
        elif key == "line":
            material = rendering.MaterialRecord()
            material.shader = "unlitLine"
            material.line_width = 0.2 * self.window.theme.font_size
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

    def get_debug_widget(self):
        widget = gui.ImageWidget()
        widget.visible = False
        self.window.add_child(widget)
        return widget

    def get_info(self):
        info = gui.Label("")
        info.visible = False
        self.window.add_child(info)
        return info

    def get_scale_panel(self):
        logger.debug("Prepare scale panel")
        scale_panel = Panel(gui.Vert, 0.5 * self.em, parent=self.tool_layout)
        scale_panel.add_label("Scale parameters", self.fonts['large'])
        scale_panel.add_label("Shift-click on two points to draw a line", self.fonts['small'])
        scale_panel.add_input("px", float, tooltip="Length of line", on_changed=self.update_scale)
        scale_panel.add_input("mm", float, tooltip="Physical distance", on_changed=self.update_scale)
        scale_panel.add_input("scale", float, tooltip="Scale (px/mm)")
        scale_panel.add_button("save", self.set_scale, tooltip="Save scale parameter")
        scale_panel.add_input("line colour", gui.Color, value=gui.Color(1.0, 0.0, 0.0), tooltip="Set line colour")
        return scale_panel

    def get_layout_panel(self):
        logger.debug("Prepare layout panel")
        layout_panel = Panel(gui.Vert, 0.5 * self.em, parent=self.tool_layout)
        layout_panel.add_label("Layout parameters", self.fonts['large'])
        layout_horiz = Panel(gui.Horiz, 0.5 * self.em, parent=layout_panel)
        layout_numbers = Panel(gui.Vert, 0.5 * self.em, parent=layout_horiz)
        layout_buttons = Panel(gui.Vert, 0.5 * self.em, parent=layout_horiz)

        layout_numbers.add_label(
            "Shift-click on two points to draw a line\nand copy the value into measured fields", self.fonts['small']
        )
        layout_numbers.add_input("px", float, tooltip="Line length")
        layout_numbers.add_input("line colour", gui.Color, value=gui.Color(1.0, 0.0, 0.0), tooltip="Set line colour")
        layout_numbers.add_separation(2)
        layout_panel.add_label("Measured parameters", self.fonts['small'])
        layout_numbers.add_input("circle diameter", float, tooltip="Diameter of circle (px)")
        layout_numbers.add_input("circle separation", float, tooltip="Maximum distance between edges of circles within a plate (px)")
        layout_numbers.add_input("plate width", float, tooltip="Shortest dimension of plate edge (used to calculate plate cut-height)")
        layout_numbers.add_separation(2)
        layout_numbers.add_label("Counts", self.fonts['small'])
        layout_numbers.add_input("circles", int, tooltip="Number of circles per plate")
        layout_numbers.add_input("plates", int, tooltip="Number of plates per image")
        layout_numbers.add_separation(2)
        layout_numbers.add_label("Expansion factors", self.fonts['small'])
        layout_numbers.add_input("circle expansion", float, tooltip="Applied to radius of circle for final region of interest (not used during search)")
        layout_numbers.add_input("circle separation tolerance", float, tooltip="Applied calculating cut height for plate clustering")
        layout_buttons.add_label("Plate ID incrementation", self.fonts['small'])
        layout_buttons.add_button(  #todo these should change text when toggled (e.g. plates in cols), use something like what was done with the button pool
            "plates in rows",
            None,
            tooltip="Increment plates in rows",
            toggleable=True
        )
        layout_buttons.add_button(
            "plates start left",
            None,
            tooltip="Increment plates left to right",
            toggleable=True
        )
        layout_buttons.add_button(
            "plates start top",
            None,
            tooltip="Increment plates top to bottom",
            toggleable=True
        )
        layout_buttons.add_label("Circle ID incrementation", self.fonts['small'])
        layout_buttons.add_button(
            "circles in rows",
            None,
            tooltip="Increment circles in rows",
            toggleable=True
        )
        layout_buttons.add_button(
            "circles start left",
            None,
            tooltip="Increment circles left to right",
            toggleable=True
        )
        layout_buttons.add_button(
            "circles start top",
            None,
            tooltip="Increment circles top to bottom",
            toggleable=True
        )
        layout_buttons.add_separation(2)

        layout_buttons.add_button("save parameters", self.save_layout, tooltip="Save all parameters")

        layout_buttons.add_button("test layout", self.test_layout, tooltip="Test layout detection")
        layout_buttons.buttons['test layout'].enabled = layout_defined(self.args)

        layout_buttons.add_button("save fixed layout", self.save_fixed_layout, tooltip="Save fixed layout")
        layout_buttons.buttons['save fixed layout'].enabled = self.layout is not None
        return layout_panel

    def test_layout(self, event=None):
        self.save_layout()
        fig = self.image.figures.new_figure("Plate detection", cols=2, level="WARN")
        circles, plates, fig = get_layout(self.image, fig)
        if plates is None:
            self.layout_panel.buttons["save fixed layout"].enabled = False
            self.update_layout_image(fig.as_array())
            self.debug_widget.visible = True
            self.window.set_needs_layout()
        else:
            self.layout = Layout(plates, self.image)  # Note ambiguous terms, plate layout from this and layout in o3d.
            self.layout_panel.buttons["save fixed layout"].enabled = True
            self.update_layout_image(self.layout.overlay)
            self.debug_widget.visible = True
            self.window.set_needs_layout()

    def load_layout_parameters(self):
        self.layout_panel.set_value("circle diameter", self.args.circle_diameter)
        self.layout_panel.set_value("circle separation", self.args.circle_separation)
        self.layout_panel.set_value("plate width", self.args.plate_width)
        self.layout_panel.set_value("circle expansion", self.args.circle_expansion)
        self.layout_panel.set_value("circle separation tolerance", self.args.circle_separation_tolerance)
        self.layout_panel.set_value("circles", self.args.circles_per_plate)
        self.layout_panel.set_value("plates", self.args.plates)
        self.layout_panel.buttons['plates in rows'].is_on = not self.args.plates_cols_first
        self.layout_panel.buttons['plates start left'].is_on = not self.args.plates_right_left
        self.layout_panel.buttons['plates start top'].is_on = not self.args.plates_bottom_top
        self.layout_panel.buttons['circles in rows'].is_on = not self.args.circles_cols_first
        self.layout_panel.buttons['circles start left'].is_on = not self.args.circles_right_left
        self.layout_panel.buttons['circles start top'].is_on = not self.args.circles_bottom_top

    def save_target_parameters(self):
        self.update_hull()
        if self.hull_holder is None:
            logger.debug("No points to save")
            return
        elif self.hull_holder.mesh is None:
            logger.warning("Incomplete hull")
        update_arg(self.args, "hull_vertices", list(map(tuple,  self.hull_holder.points)))
        update_arg(self.args, "alpha", self.target_panel.get_value("alpha"))
        update_arg(self.args, "delta", self.target_panel.get_value("delta"))
        update_arg(self.args, "fill", self.target_panel.get_value("fill"))
        update_arg(self.args, "remove", self.target_panel.get_value("remove"))


    def save_layout(self):
        update_arg(self.args, "circle_diameter", self.layout_panel.get_value("circle diameter"))
        update_arg(self.args, "circle_separation", self.layout_panel.get_value("circle separation"))
        update_arg(self.args, "plate_width", self.layout_panel.get_value("plate width"))
        update_arg(self.args, "circle_expansion", self.layout_panel.get_value("circle expansion"))
        update_arg(self.args, "circle_separation_tolerance", self.layout_panel.get_value("circle separation tolerance"))
        update_arg(self.args, "circles_per_plate", self.layout_panel.get_value("circles"))
        update_arg(self.args, "plates", self.layout_panel.get_value("plates"))
        update_arg(self.args, "plates_cols_first", self.layout_panel.get_value("plates in rows"))
        update_arg(self.args, "plates_right_left", self.layout_panel.get_value("plates start left"))
        update_arg(self.args, "plates_bottom_top", self.layout_panel.get_value("plates start top"))
        update_arg(self.args, "circles_cols_first", self.layout_panel.get_value("circles in rows"))
        update_arg(self.args, "circles_right_left", self.layout_panel.get_value("circles start left"))
        update_arg(self.args, "circles_bottom_top", self.layout_panel.get_value("circles start top"))
        self.layout_panel.buttons['test layout'].enabled = layout_defined(self.args)


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
        if self.layout is None:
            raise ValueError("Layout has not been defined")

        circles_dicts = list()
        for i, p in enumerate(self.layout.plates):
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
        circle_panel = Panel(gui.ScrollableVert, 0.5 * self.em, parent=self.tool_layout)
        circle_panel.add_label("Circle colour parameters", self.fonts['large'])
        circle_panel.add_label("Shift-click to select pixels", self.fonts['small'])
        circle_panel.add_button("Clear", self.clear_circle_selection, tooltip="Clear selected pixels")
        circle_panel.add_button("Save", self.save_circle_colour, tooltip="Save selected colour")
        circle_panel.add_separation(2)
        circle_panel.add_input(
            "pixel colour",
            gui.Color,
            value=gui.Color(1.0, 1.0, 1.0),
            tooltip="Set selected pixel colour",
            on_changed=self.update_image_widget
        )
        return circle_panel

    def get_target_panel(self):
        logger.debug("Prepare target panel")
        target_panel = Panel(gui.ScrollableVert, 0.5 * self.em, parent=self.tool_layout)
        target_panel.add_label("Target hull parameters", self.fonts['large'])
        target_horiz = Panel(gui.Horiz, self.em, parent=target_panel)
        self.add_target_buttons(target_horiz)
        self.add_selection_panel(target_horiz)
        self.add_prior_panel(target_horiz)
        return target_panel

    def add_target_buttons(self, parent: Panel):
        logger.debug("Prepare target buttons")
        target_buttons = Panel(gui.Vert, 0.5 * self.em, parent=parent)
        target_buttons.add_label("Shift-click to select colours", self.fonts['small'])
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
            "min pixels",
            int,
            tooltip="Minimum number of pixels to display voxel",
            on_changed=self.set_min_pixels
        )
        target_buttons.add_button("show hull", self.draw_hull, tooltip="Display hull in 3D plot", toggleable=True)
        target_buttons.add_button(
            "show selected",
            self.update_image_widget,
            tooltip="Highlight selected pixels in image plot",
            toggleable=True
        )
        target_buttons.add_button(
            "show within",
            self.update_image_widget,
            tooltip="Highlight pixels inside and within delta of hull in image plot",
            toggleable=True
        )
        target_buttons.add_input(
            "fill", int, tooltip="Fill holes smaller than this size", on_changed=self.update_image_widget
        )
        target_buttons.set_value("fill", self.args.fill)
        target_buttons.add_input(
            "remove", int, tooltip="Remove objects below this size", on_changed=self.update_image_widget
        )
        target_buttons.set_value("remove", self.args.remove)
        target_buttons.add_separation(1)
        target_buttons.add_button(
            "from mask",
            tooltip="Calculate hull vertices from target mask (uses min pixels and alpha)",
            on_clicked=self.hull_from_mask,
            enabled=False
        )
        target_buttons.add_button(
            "Save",
            tooltip="Save selected points as arguments (consider reducing to hull vertices first)",
            on_clicked=self.save_target_parameters
        )
        target_buttons.add_separation(2)
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
            "within colour",
            gui.Color,
            value=gui.Color(1.0, 0.0, 0.0),
            tooltip="Set selection highlighting colour",
            on_changed=self.update_image_widget
        )

        return target_buttons

    def add_selection_panel(self, parent: Panel):
        logger.debug("Prepare selection panel")
        selection_panel = Panel(gui.Vert, 0.5 * self.em, parent=parent)
        selection_panel.add_label("selected", font_style=self.fonts['small'])
        selection_panel.add_button("clear", tooltip="Clear selected colours", on_clicked=self.clear_selection)
        selection_panel.add_button("reduce", tooltip="Keep only hull vertices", on_clicked=self.reduce_selection)
        selection_panel.add_button_pool("selection")
        return selection_panel

    def add_prior_panel(self, parent: Panel):
        logger.debug("Prepare prior panel")
        prior_panel = Panel(gui.Vert, 0.5 * self.em, parent=parent)
        prior_panel.add_label("priors", font_style=self.fonts['small'])
        prior_panel.add_button("clear", tooltip="Clear prior colours", on_clicked=self.clear_priors)
        prior_panel.add_button('reduce', tooltip="Keep only hull vertices", on_clicked=self.reduce_priors)
        prior_panel.add_button_pool("priors")
        return prior_panel

    def hide_all(self):
        self.info.visible = False
        self.lab_widget.visible = False
        self.image_widget.visible = False
        self.debug_widget.visible = False
        self.tool_layout.visible = True

        self.target_panel.visible = False
        self.scale_panel.visible = False
        self.circle_panel.visible = False
        self.layout_panel.visible = False

    def start_target(self):
        self.activity = Activities.TARGET
        self.hide_all()

        self.info.visible = True
        self.lab_widget.visible = True
        self.image_widget.visible = True
        self.tool_layout.visible = True
        self.target_panel.visible = True

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
        self.debug_widget.visible = True

        self.update_image_widget()
        self.update_distance_widget()
        self.window.set_needs_layout()

    def start_layout(self):
        self.activity = Activities.LAYOUT
        self.hide_all()

        self.info.visible = True
        self.image_widget.visible = True
        self.tool_layout.visible = True
        self.layout_panel.visible = True

        self.update_image_widget()
        self.window.set_needs_layout()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        if self.activity == Activities.NONE:
            self.background_widget.visible = True
            self.background_widget.frame = gui.Rect(r.x, r.y, r.width, r.height)
        if self.image is not None:
            self.background_widget.visible = False
            tool_pref = self.tool_layout.calc_preferred_size(layout_context, gui.Widget.Constraints())

            if self.activity in [Activities.CIRCLE, Activities.LAYOUT]:
                tool_width = int(r.width/2)
            else:
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
            image_ratio = self.image.displayed.shape[0] / self.image.displayed.shape[1]  # height/width

            self.debug_widget.frame = gui.Rect(r.x, r.y, tool_width, tool_width * image_ratio)

            height_used = (
                    self.info.frame.height * self.info.visible +
                    self.lab_widget.frame.height * self.lab_widget.visible +
                    self.debug_widget.frame.height * self.debug_widget.visible
            )
            if self.lab_widget.visible:
                tool_start_y = self.lab_widget.frame.get_bottom()
            elif self.debug_widget.visible:
                tool_start_y = self.debug_widget.frame.get_bottom()
            else:
                tool_start_y = r.y
            self.tool_layout.frame = gui.Rect(
                r.x,
                tool_start_y,
                tool_width,
                r.height - height_used
            )

            # if not sized to fill :
            #  - we get deadspace if just in window and deadspace doesn't render properly
            #  - we could fix this with a stretch in a panel, but then we get forced to have a scrollbar
            #    and scrollbar doesn't provide a position to compute the pixels (as far as I can tell)
            # so we ensure we always fill the space such that the coords have a consistent origin
            # find which axis (width or height) is greater
            #
            # we also need to handle the pop of a debug image below the image
            image_width = r.width - (tool_width * self.tool_layout.visible)
            image_start_x = self.tool_layout.frame.get_right() * self.tool_layout.visible

            # Then we find where there is space to fill
            if r.height >= image_width * image_ratio:  # space in height so fix to fill this
                image_frame_height = r.height
                image_frame_width = image_frame_height / image_ratio
            else:  # space in width so fix, rely on stretch to fill this
                image_frame_width = image_width
                image_frame_height = image_frame_width * image_ratio

            # need to handle debug panel when visible:
            self.image_widget.frame = gui.Rect(image_start_x, r.y, image_frame_width, image_frame_height)
            self.image_widget.frame = gui.Rect(image_start_x, r.y, image_frame_width, image_frame_height)


    def on_mouse_lab_widget(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.SHIFT):

            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x - self.lab_widget.frame.x
                y = event.y - self.lab_widget.frame.y

                # have a reasonable selection radius and just get the point closest to the camera within that
                def nearest_within(radius):
                    image = np.asarray(depth_image)
                    ys = np.arange(0, image.shape[0])
                    xs = np.arange(0, image.shape[1])
                    mask = (xs[np.newaxis,:]-x)**2 + (ys[:,np.newaxis]-y)**2 >= radius**2
                    masked_array = np.ma.masked_array(image, mask=mask)
                    nearest_index = masked_array.argmin(fill_value=1)
                    return self.image.pixel_to_coord(image, nearest_index)

                x, y = nearest_within(3)
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # 1 if clicked on nothing (i.e. the far plane)
                    nearest_index = None
                else:
                    world = self.lab_widget.scene.camera.unproject(
                        x, y, depth, self.lab_widget.frame.width,
                        self.lab_widget.frame.height
                    )
                    # get from world coords to the nearest point in the input array
                    nearest_index = self.get_nearest_index_from_coords(world)
                # This is not called on the main thread, so we need to
                # post to the main thread to safely access UI items.

                def update_selected():
                    if nearest_index is not None:
                        self.toggle_voxel(nearest_index)
                        #self.window.set_needs_layout()

                self.app.post_to_main_thread(self.window, update_selected)

            self.lab_widget.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def get_nearest_index_from_coords(self, coords):
        single_point_cloud = o3d.geometry.PointCloud()
        single_point_cloud.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector([coords]))
        dist = self.image.cloud.compute_point_cloud_distance(single_point_cloud)
        return np.argmin(dist)

    def clear_selection(self, event=None):
        selected_points = self.image.cloud.select_by_index(list(self.target_indices)).points
        for lab in selected_points:
            lab_text = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
            self.lab_widget.scene.remove_geometry(lab_text)
        self.target_indices.clear()
        self.target_panel.button_pools['selection'].clear()
        self.update_hull()
        self.update_image_widget()

    def clear_priors(self, event=None):
        self.prior_lab.clear()
        self.target_panel.button_pools['priors'].clear()
        self.update_hull()
        self.update_image_widget()

    def reduce_selection(self, event=None):
        if self.hull_holder.mesh is not None:
            hull_vertices = self.hull_holder.hull.vertices
            to_remove = list()
            for i in self.target_indices:
                lab = self.image.cloud.points[i]
                if lab not in hull_vertices:
                    lab_text = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
                    self.lab_widget.scene.remove_geometry(lab_text)
                    self.target_panel.button_pools['selection'].remove_button(i)
                    to_remove.append(i)
            logger.debug(f"Removing selected points: {len(to_remove)}")
            [self.target_indices.remove(i) for i in to_remove]

    def reduce_priors(self, event=None):
        if self.hull_holder.mesh is not None:
            hull_vertices = self.hull_holder.hull.vertices
            to_remove = list()
            for lab in self.prior_lab:
                if lab not in hull_vertices:
                    logger.debug(f"removing {lab}")
                    lab_text = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
                    self.target_panel.button_pools['priors'].remove_button(lab_text)
                    to_remove.append(lab)
            logger.debug(f"Removing selected points: {len(to_remove)}")
            [self.prior_lab.remove(lab) for lab in to_remove]
        self.update_hull()

    def clear_circle_selection(self, event=None):
        self.circle_indices.clear()
        self.update_image_widget()
        self.update_distance_widget()
        self.args.circle_colour = None

    def save_circle_colour(self, event=None):
        circle_lab = self.image.lab.reshape(-1, 3)[list(self.circle_indices)]
        circle_lab = tuple(np.mean(circle_lab, axis=0))
        update_arg(self.args, "circle_colour", circle_lab)
        gui.Application.instance.menubar.set_enabled(self.MENU_LAYOUT, True)

    def update_scale(self, event=None):
        px = self.scale_panel.get_value('px')
        mm = self.scale_panel.get_value('mm')
        if px and mm:
            scale = float(np.around(px / mm, decimals=4))
            self.scale_panel.set_value("scale", scale)

    def set_scale(self, event=None):
        logger.debug("save scale to args")
        scale = self.scale_panel.get_value("scale")
        update_arg(self.args, "scale", scale)

    def update_alpha(self, event=None):
        if self.hull_holder is not None:
            self.hull_holder.update_alpha(self.target_panel.get_value("alpha"))
            self.update_hull()
            self.update_image_widget()

    def update_delta(self, event=None):
        self.update_image_widget()

    def set_min_pixels(self, event=None):
        self.update_lab_widget()

    def image_widget_to_image_coords(self, event_x, event_y):
        #logger.debug(f"event x,y: {event_x, event_y}")
        #logger.debug(f"frame x,y: {self.image_widget.frame.x, self.image_widget.frame.y}")
        frame_x = event_x - self.image_widget.frame.x  # todo this fails when scrollbar present
        frame_y = event_y - self.image_widget.frame.y
        #logger.debug(f"frame coords: {frame_x, frame_y}")
        frame_fraction_x = frame_x / self.image_widget.frame.width
        frame_fraction_y = frame_y / self.image_widget.frame.height

        #frame spacing depends on image ratio due to scaling, when image height is less than frame height there is a margin equally split

        displayed_image_x = np.floor(self.image.width * frame_fraction_x)
        displayed_image_y = np.floor(self.image.height * frame_fraction_y)
        #logger.debug(f"displayed coords: {displayed_image_x, displayed_image_y}")
        image_x = np.floor((displayed_image_x*self.image.zoom_factor)) + self.image.displayed_start_x
        image_y = np.floor((displayed_image_y*self.image.zoom_factor)) + self.image.displayed_start_y
        #logger.debug(f"image coords: {image_x, image_y}")
        x = int(image_x)
        y = int(image_y)
        return x, y

    def image_to_displayed_coords(self, x, y):  # todo , this seems to be redundant with some of the image functions
        x = int(np.floor((x - self.image.displayed_start_x) / self.image.zoom_factor))
        y = int(np.floor((y - self.image.displayed_start_y) / self.image.zoom_factor))
        return x, y

    def on_mouse_image_widget(self, event):

        if event.type in [
            gui.MouseEvent.Type.BUTTON_DOWN,
            gui.MouseEvent.Type.BUTTON_UP,
            gui.MouseEvent.Type.WHEEL
        ]:
            x, y = self.image_widget_to_image_coords(event.x, event.y)

            if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                    gui.KeyModifier.SHIFT):

                if self.activity == Activities.TARGET:
                    # logger.debug(f"Image coordinates: {x, y}")
                    pixel_index = self.image.coord_to_pixel(self.image.rgb, x, y)
                    voxel_index = self.image.image_to_voxel[pixel_index]
                    # logger.debug(f"Voxel index {voxel_index}")
                    if voxel_index is not None:
                        self.toggle_voxel(voxel_index)

                elif self.activity == Activities.CIRCLE:
                    self.toggle_circle_pixel(x, y)

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
                # the queue is used to defer zooming rather than do it for each event
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
                self.image.drag(drag_x, drag_y)
                self.update_image_widget()
                self.drag_start = None

            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def draw_line(self, x, y, line_colour):
        if self.measure_start is None:
            self.measure_start = (x, y)
            start_x, start_y = self.image_to_displayed_coords(*self.measure_start)
            displayed = self.image.displayed.copy()
            disk = draw.disk((start_y, start_x), 5, shape=displayed.shape)
            displayed[disk] = line_colour
            to_render = o3d.geometry.Image(displayed.astype(np.float32))
            self.image_widget.update_image(to_render)
            return None
        else:
            measure_end = (x, y)
            start_x, start_y = self.image_to_displayed_coords(*self.measure_start)
            end_x, end_y = self.image_to_displayed_coords(*measure_end)
            displayed = self.image.displayed.copy()
            yy, xx, val = draw.line_aa(start_y, start_x, end_y, end_x)
            line_vals = np.multiply.outer(val, line_colour)
            displayed[yy, xx] = line_vals
            to_render = o3d.geometry.Image(displayed.astype(np.float32))
            self.image_widget.update_image(to_render)
            distance = float(np.around(np.linalg.norm(
                np.array(self.measure_start) - np.array(measure_end)
            ), decimals=1))
            logger.debug(f"distance:{distance}")
            return distance

    def zoom_image(self, x, y, dy):
        if not self.zoom_queue.empty():  # if existing zoom requests in queue then we just add them
            logger.debug("added to zoom queue")
            self.zoom_queue.put((x, y, dy))
        else:  # if empty we add but start a thread to watch for more then calculate
            logger.debug("adding to queue and preparing thread to catch more")
            self.zoom_queue.put((x, y, dy))

            def zoom_aggregate():
                time.sleep(0.1)  # wait a bit in case more requests are coming in
                zoom_increment = 0
                zoom_x = None
                zoom_y = None
                start = True
                while True:
                    try:
                        zx, zy, dy = self.zoom_queue.get(False)
                        if start:
                            zoom_x = zx
                            zoom_y = zy
                            start = False
                        zoom_increment += dy  # collect the delta from each event
                    except Empty:
                        self.zoom_queue.put((zoom_x, zoom_y, zoom_increment))
                        logger.debug(f"Finished aggregating queue: increment = {zoom_increment}")
                        break

                while True:
                    try:
                        zoom_x, zoom_y, zoom_increment = self.zoom_queue.get(False)
                        if zoom_increment != 0 and zoom_x is not None:
                            cropped_rescaled, start_x, start_y = self.image.calculate_zoom(
                                zoom_x,
                                zoom_y,
                                zoom_increment
                            )
                            logger.debug("Calculated zoom")

                            def do_zoom():
                                logger.debug(f"Zooming at {x, y}, zoom factor {self.image.zoom_factor}")
                                self.image.apply_zoom(cropped_rescaled, start_x, start_y)
                                self.update_image_widget()

                                if not self.zoom_queue.empty:
                                    # check if any more came in since we processed the queue,
                                    # we may need to start again
                                    logger.debug("Restart zoom watcher")
                                    self.app.run_in_thread(zoom_aggregate)

                            self.app.post_to_main_thread(self.window, do_zoom)

                    except Empty:
                        break

            self.app.run_in_thread(zoom_aggregate)

    def toggle_circle_pixel(self, x, y):
        logger.debug("toggle circle pixel")
        pixel_index = self.image.coord_to_pixel(self.image.rgb, x, y)
        if pixel_index in self.circle_indices:
            logger.debug(f"remove pixel index: {pixel_index}")
            self.circle_indices.remove(pixel_index)
        else:
            logger.debug(f"add pixel index: {pixel_index}")
            self.circle_indices.add(pixel_index)
        self.update_image_widget()
        self.update_distance_widget()

    def toggle_voxel(self, lab_index):
        if not self.toggle_queue.empty():
            self.toggle_queue.put(lab_index)
        else:
            self.toggle_queue.put(lab_index)

            def toggle_aggregate():
                time.sleep(0.2)  # wait in case more come in
                indices = set()

                while True:
                    try:
                        ind = self.toggle_queue.get(False)
                        if ind in indices:
                            indices.remove(ind)
                        else:
                            indices.add(ind)
                    except Empty:
                        logger.debug("Finished agregating selected toggles")
                        break

                def do_toggle():

                    for i in indices:
                        selected_lab = self.image.cloud.points[i]
                        selected_rgb = self.image.cloud.colors[i]

                        if i in self.target_indices:
                            logger.debug(f"Remove: {selected_lab}")
                            self.target_indices.remove(i)
                            self.remove_sphere(selected_lab)
                            self.target_panel.button_pools['selection'].remove_button(i)
                        else:
                            self.target_indices.add(i)
                            self.add_sphere(selected_lab, selected_rgb)
                            self.add_voxel_button(i, selected_lab, selected_rgb)
                        logger.debug("get selected points to update hull holder")

                    self.update_hull()
                    self.update_image_widget()
                    if not self.toggle_queue.empty():
                        logger.debug("Restarting toggle watcher")
                        self.app.run_in_thread(toggle_aggregate)

                self.app.post_to_main_thread(self.window, do_toggle)

            self.app.run_in_thread(toggle_aggregate)

    def add_voxel_button(self, lab_index: int, lab, rgb):
        key = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
        b = gui.Button(key)
        b.background_color = gui.Color(*rgb)
        self.target_panel.button_pools['selection'].add_button(lab_index, b)
        b.set_on_clicked(lambda: self.toggle_voxel(lab_index))

    def add_sphere(self, lab, rgb):
        logger.debug(f"Adding sphere for {lab}")
        key = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
        sphere = o3d.geometry.TriangleMesh().create_sphere(radius=self.args.colour_rounding/2)
        sphere.paint_uniform_color(rgb)
        sphere.compute_vertex_normals()
        sphere.translate(lab)
        self.lab_widget.scene.add_geometry(key, sphere, self.point_material)

    def remove_sphere(self, lab):
        key = "({:.1f}, {:.1f}, {:.1f})".format(lab[0], lab[1], lab[2])
        self.lab_widget.scene.remove_geometry(key)

    def add_prior(self, lab: tuple[float, float, float]):
        self.prior_lab.add(lab)
        rgb = lab2rgb(lab)
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
        if self.target_indices:
            points = self.image.cloud.select_by_index(list(self.target_indices)).points
            points.extend(self.prior_lab)
        else:
            points = list(self.prior_lab)

        logger.debug(f"Update hull points {len(points)}")
        self.hull_holder = HullHolder(points, self.target_panel.get_value("alpha"))
        if self.hull_holder is not None:
            self.draw_hull()

    def highlight_target(self):
        logger.debug("Highlight from queue")
        time.sleep(1)
        while True:
            try:
                self.update_queue.get(False)  # we wait for calls to stop before calculating
            except Empty:
                break
        if self.target_panel.buttons['show selected'].is_on:
            selected = [j for i in list(self.target_indices) for j in self.image.indices[i]]
            selected = self.image.indices_in_displayed(selected)
        else:
            selected = []
        if all([
            self.target_panel.buttons['show within'].is_on,
            self.hull_holder is not None,
            self.hull_holder.mesh is not None
        ]):
            logger.debug(
                f"Calculate distance from hull for {self.image.displayed_lab.reshape(-1, 3).shape[0]} pixels"
            )
            distances = self.hull_holder.get_distances(self.image.displayed_lab)
            if distances is not None:
                try:
                    distances = distances.reshape(self.image.displayed_lab.shape[0:2])
                except Exception as e:
                    logger.debug(f"exception during highlight aggregate: {e}")
                    return  # i think this exception is just due to the async request for the distances todo fix
                logger.debug("rescale distances to displayed")
                distances = zoom(
                    distances,
                    self.image.divisors[self.image.zoom_index],
                    order=0,
                    grid_mode=True,
                    mode='nearest'
                )
                logger.debug("Find within")
                within = np.where(distances.reshape(-1) <= self.target_panel.get_value("delta"))[0]
            else:
                within = []
        else:
            within = []
        logger.debug("get a copy of the displayed image to highlight")
        highlighted = self.image.displayed.copy()  # make a copy
        logger.debug("show within")

        dice = None

        within_mask = np.zeros(self.image.rgb.shape[0:2], bool)
        within_mask.reshape(-1)[within] = 1
        if self.target_panel.get_value("fill"):
            logger.debug("Fill small holes")
            fill_size = self.target_panel.get_value("fill")
            zoomed_fill = int(((fill_size ** (1/2)) * (1/self.image.zoom_factor))**2)
            logger.debug(f"zoomed fill size : {zoomed_fill}")
            within_mask = remove_small_holes(within_mask, zoomed_fill)
        if self.target_panel.get_value("remove"):
            logger.debug("Remove small objects")
            remove_size = self.target_panel.get_value("remove")
            zoomed_remove = int(((remove_size ** (1/2)) * (1/self.image.zoom_factor))**2)
            logger.debug(f"zoomed remove size : {zoomed_remove}")
            within_mask = remove_small_objects(within_mask, zoomed_remove)
        within_mask = within_mask.reshape(-1)

        if self.target_mask is not None and self.image.zoom_factor == 1:
            logger.debug("Prepare statistics")
            true_mask = np.all(self.target_mask.rgb == [1, 1, 1], axis=2).reshape(-1)
            tp = np.sum(true_mask[within_mask])
            tn = np.sum(~true_mask[~within_mask])
            fp = np.sum(~true_mask[within_mask])
            fn = np.sum(true_mask[~within_mask])
            sensitivity = (tp/(tp + fn))
            specificity = (tn/(tn + fp))
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            dice = 2 * tp / ((2 * tp) + fp + fn)
            logger.debug(f"Dice coefficient: {dice},  Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}")
        logger.debug("prepare image to render")

        within_colour = self.target_panel.get_value("within colour")
        highlighted.reshape(-1,3)[within_mask] = within_colour
        logger.debug("show selected")
        selected_colour = self.target_panel.get_value("selection colour")
        highlighted.reshape(-1, 3)[selected] = selected_colour


        to_render = o3d.geometry.Image(highlighted.astype(np.float32))

        def do_highlighting():
            logger.debug("Do highlight")
            self.image_widget.update_image(to_render)
            self.update_info(dice)

            self.window.set_needs_layout()
            if not self.update_queue.empty:
                logger.debug("Restart highlight aggregation")
                self.app.run_in_thread(self.highlight_target)

        self.app.post_to_main_thread(self.window, do_highlighting)

    def update_image_widget(self, _event=None):
        logger.debug(f"update image widget")
        if self.activity == Activities.TARGET:
            self.app.run_in_thread(self.highlight_target)
        elif self.activity == Activities.CIRCLE:
            self.highlight_circle_pixels()
        else:
            to_render = o3d.geometry.Image(self.image.displayed.astype(np.float32))
            self.image_widget.update_image(to_render)
        self.window.set_needs_layout()
        return gui.Widget.EventCallbackResult.HANDLED

    def update_layout_image(self, array):
        logger.debug(f"update layout widget")
        to_render = o3d.geometry.Image(array.astype(np.float32))
        self.debug_widget.update_image(to_render)
        self.window.set_needs_layout()
        return gui.Widget.EventCallbackResult.HANDLED

    def update_distance_widget(self, _event=None):
        logger.debug("update distance widget")
        if self.activity == Activities.CIRCLE:
            distance_image = self.get_distance_image()
            if distance_image is not None:
                to_render = o3d.geometry.Image(gray2rgb(distance_image).astype(np.float32))
                self.debug_widget.update_image(to_render)
            else:
                to_render = o3d.geometry.Image(np.zeros_like(self.image.rgb).astype(np.float32))
                self.debug_widget.update_image(to_render)
        return gui.Widget.EventCallbackResult.HANDLED

    def get_distance_image(self):
        if self.activity == Activities.CIRCLE and self.circle_indices:
            logger.debug("get distance image")
            if self.args.circle_colour is None:
                circle_lab = self.image.lab.reshape(-1, 3)[list(self.circle_indices)]
                circle_lab = np.mean(circle_lab, axis=0)
            else:
                circle_lab = self.args.circle_colour

            circles_like = np.full_like(self.image.lab, circle_lab)
            return 1-deltaE_cie76(self.image.lab, circles_like)/255
        else:
            return None

    def highlight_circle_pixels(self):
        logger.debug("highlight circle pixels")
        if self.circle_indices:
            highlighted = self.image.displayed.copy()
            colour = self.circle_panel.get_value("pixel colour")
            displayed_indices = self.image.indices_in_displayed(self.circle_indices)
            highlighted.reshape(-1, 3)[displayed_indices] = colour
            to_render = o3d.geometry.Image(highlighted.astype(np.float32))
        else:
            to_render = o3d.geometry.Image(self.image.displayed.astype(np.float32))
        logger.debug("update image")
        self.image_widget.update_image(to_render)

    def update_info(self, dice=None):
        if self.image is None:
            self.info.text = ""
            return
        elif self.target_mask is None:
            self.info.text = f"Image file: {str(self.image.filepath.name)}\n"
        elif self.target_mask is not None and dice is None:
            self.info.text = (
                f"Image file: {str(self.image.filepath.name)}\n"
                f"Mask file: {str(self.target_mask.filepath.name)}\n"
            )
        else:
            self.info.text = (
                f"Image file: {str(self.image.filepath.name)}\n"
                f"Mask file: {str(self.target_mask.filepath.name)}\n"
                f"Dice coefficient: {float(dice)}"
            )

    def draw_hull(self, event=None):
        if self.lab_widget.scene is not None:
            logger.debug("Remove existing mesh")
            try:
                self.lab_widget.scene.remove_geometry('mesh')
            except Exception as e:
                logger.debug(f'mesh not found {e}')
        if all([
            self.target_panel.buttons['show hull'].is_on,
            self.hull_holder is not None,
            self.hull_holder.mesh is not None
        ]):
            logger.debug("Add mesh to scene")
            hull_colour = self.target_panel.get_value("hull colour")
            self.hull_holder.mesh.compute_vertex_normals()
            self.hull_holder.mesh.paint_uniform_color(hull_colour)
            self.lab_widget.scene.add_geometry("mesh", self.hull_holder.mesh, self.point_material)

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose image to load", self.window.theme)
        extensions = [f".{s}" for s in IMAGE_EXTENSIONS]
        dlg.add_filter(" ".join(extensions), f"Supported image files ({', '.join(extensions)}")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_image_dialog_done)
        self.window.show_dialog(dlg)

    def _on_menu_mask(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose mask image to load", self.window.theme)
        extensions = [f".{s}" for s in IMAGE_EXTENSIONS]
        dlg.add_filter(" ".join(extensions), f"Supported image files ({', '.join(extensions)}")
        dlg.add_filter("", "All files")
        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_mask_dialog_done)
        self.window.show_dialog(dlg)

    def _on_menu_write(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save to", self.window.theme)
        extensions = [f".conf"]
        dlg.add_filter(" ".join(extensions), f"Configuration files ({', '.join(extensions)}")
        dlg.add_filter("", "All files")
        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_write_dialog_done)
        self.window.show_dialog(dlg)


    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_image_dialog_done(self, filename):
        filepath = Path(filename)
        self.window.close_dialog()
        self.load_image(filepath)

    def _on_load_mask_dialog_done(self, filename):
        filepath = Path(filename)
        self.window.close_dialog()
        self.load_mask(filepath)

    def _on_write_dialog_done(self, filename):
        filepath = Path(filename)
        self.window.close_dialog()
        self.write_calibration(filepath)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        pass
        #self._settings_panel.visible = not self._settings_panel.visible
        #gui.Application.instance.menubar.set_checked(
        #    AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(self.em, gui.Margins(self.em, self.em, self.em, self.em))
        dlg_layout.add_child(gui.Label("AlGrow Calibration GUI"))

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

    def load_image(self, path):
        if self.image is not None:
            selected_points = self.image.cloud.select_by_index(list(self.target_indices)).points
            for lab in selected_points:
                lab = tuple(lab)
                self.add_prior(lab)
            self.clear_selection()
            self.circle_indices.clear()

        self.image = CalibrationImage(ImageLoaded(path, self.args))
        self.info.visible = True
        self.update_info()

        self.target_mask = None  # expect a new mask for each
        self.target_panel.buttons['from mask'].enabled = False

        gui.Application.instance.menubar.set_enabled(self.MENU_MASK, True)
        gui.Application.instance.menubar.set_enabled(self.MENU_TARGET, True)
        gui.Application.instance.menubar.set_enabled(self.MENU_SCALE, True)
        gui.Application.instance.menubar.set_enabled(self.MENU_CIRCLE, True)
        if self.args.circle_colour is not None:
            gui.Application.instance.menubar.set_enabled(self.MENU_LAYOUT, True)

        logger.debug("Load rgb image")
        #self.image_widget.update_image(self.image.as_o3d)
        self.image_widget.visible = True
        self.image.prepare_cloud()
        self.setup_lab_axes()
        self.update_lab_widget()
        self.update_image_widget()
        self.update_distance_widget()

    def load_mask(self, path):
        try:
            self.target_mask = ImageLoaded(path, self.args)
            # todo popup warning if it doesn't look like a boolean image, consider coercing to bool also to reduce memory
            self.target_panel.buttons['from mask'].enabled = True
            self.update_info()
            self.update_image_widget()  # just needed to calculate dice
            self.window.set_needs_layout()
        except Exception as e:
            logger.debug(f"Failed to load mask {e}")

    def hull_from_mask(self):
        logger.debug("Prepare points from provided mask")
        hh = hull_from_mask(
            self.image,
            self.target_mask,
            self.target_panel.get_value("alpha"),
            self.target_panel.get_value("min pixels")
        )
        if hh is None:
            return
        elif hh.hull is not None:
            points = hh.hull.vertices
        else:
            points = hh.points

        hull_vertices_string = f'{[",".join([str(j) for j in i]) for i in points]}'.replace("'", '"')
        logger.debug(f"Hull vertices from mask: {hull_vertices_string}")
        for lab in points:
            lab = tuple(lab)
            self.add_prior(lab)  # consider running from thread
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
        to_remove = np.unique([np.asarray(line) for line in lineset.lines if ~np.any(line == 0)], axis=1)
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

    def update_lab_widget(self):
        logger.debug("Update lab widget")
        logger.debug("Filter by pixels per voxel")
        common_indices = [i for i, j in enumerate(self.image.indices) if len(j) >= self.target_panel.get_value("min pixels")]
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

    def write_calibration(self, filepath):
        logger.info("Write out calibration parameters")
        with open(filepath, 'w') as text_file:
            if self.args.circle_colour is not None:
                circle_colour_string = f"\"{','.join([str(i) for i in self.args.circle_colour])}\""
            else:
                circle_colour_string = ""

            if self.args.hull_vertices is not None:
                hull_vertices_string = f'{[",".join([str(j) for j in i]) for i in self.args.hull_vertices]}'.replace(
                    "'", '"')
            else:
                hull_vertices_string = ""

            text_file.write(f"[Scale]\n")
            text_file.write(f"scale = {self.args.scale}\n")
            text_file.write(f"[Colour parameters]\n")
            text_file.write(f"circle_colour = {circle_colour_string}\n")
            text_file.write(f"hull_vertices = {hull_vertices_string}\n")
            text_file.write(f"alpha = {self.args.alpha}\n")
            text_file.write(f"delta = {self.args.delta}\n")

            if self.args.fixed_layout is not None:
                text_file.write(f"[Layout]\n")
                text_file.write(f"fixed_layout = {str(self.args.fixed_layout)}\n")

            text_file.write(f"[Layout detection (not used if fixed_layout or whole_image arguments are provided)]\n")
            text_file.write(f"circle_diameter = {self.args.circle_diameter}\n")
            text_file.write(f"circle_expansion = {self.args.circle_expansion}\n")
            text_file.write(f"circle_separation = {self.args.circle_separation}\n")
            text_file.write(f"plate_width = {self.args.plate_width}\n")
            text_file.write(f"circles_per_plate = {self.args.circles_per_plate}\n")
            text_file.write(f"plates = {self.args.plates}\n")
            text_file.write(f"plates_cols_first = {self.args.plates_cols_first}\n")
            text_file.write(f"plates_bottom_top = {self.args.plates_bottom_top}\n")
            text_file.write(f"plates_right_left = {self.args.plates_right_left}\n")
            text_file.write(f"circles_cols_first = {self.args.circles_cols_first}\n")
            text_file.write(f"circles_bottom_top = {self.args.circles_bottom_top}\n")
            text_file.write(f"circles_right_left = {self.args.circles_right_left}\n")

            logger.debug("Finished writing to calibration file")

