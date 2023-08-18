import logging
import argparse

import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering
import platform

import numpy as np

from skimage.color import lab2rgb

from ..options.custom_types import IMAGE_EXTENSIONS
from ..image_loading import ImageLoaded
from .points import ColourPoints
from .loading import wait_for_result

isMacOS = (platform.system() == "Darwin")

logger = logging.getLogger(__name__)


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    def __init__(self, width, height, args):
        self.args: argparse.Namespace = args
        self.window = gui.Application.instance.create_window("AlGrow Calibration", width, height)
        self.window.set_on_layout(self._on_layout)

        # the below are loaded when an image is loaded
        self.cloud = None
        self.indices = None
        self.selected = set()

        # a small dialog to update with info about the last selected
        self.info = gui.Label("")
        self.info.visible = False
        self.window.add_child(self.info)

        # Prepare a scene for lab colours in 3d
        logger.debug("Prepare Lab scene (space for 3D model)")

        self.lab_widget = gui.SceneWidget()
        self.lab_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.lab_widget.scene.set_background([0, 0, 0, 1])
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.point_size = 5 * self.window.scaling
        self.material.shader = "defaultUnlit"
        self.lab_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        logger.debug("Set lighting for Lab scene")
        self.lab_widget.scene.scene.enable_sun_light(False)
        self.lab_widget.scene.scene.enable_indirect_light(True)
        self.lab_widget.scene.scene.set_indirect_light_intensity(60000)
        self.lab_widget.set_on_mouse(self.on_mouse_lab_widget)
        self.window.add_child(self.lab_widget)

        # Prepare a panel to display the source image
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        self.panel.add_child(gui.Label("Color image"))
        self.rgb_widget = None

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
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
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        self.window.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        self.window.set_on_menu_item_activated(AppWindow.MENU_EXPORT, self._on_menu_export)
        self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        self.window.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS, self._on_menu_toggle_settings_panel)
        self.window.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 0.5 * r.width
        self.lab_widget.frame = gui.Rect(
            r.x, r.y, r.width - panel_width, r.height
        )
        self.panel.frame = gui.Rect(
            self.lab_widget.frame.get_right(), r.y, panel_width, r.height
        )
        pref = self.info.calc_preferred_size(layout_context,  gui.Widget.Constraints())
        self.info.frame = gui.Rect(r.x, r.get_bottom() - pref.height, pref.width, pref.height)

    def on_mouse_lab_widget(self, event):
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
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
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # 1 if clicked on nothing (i.e. the far plane)
                    world = None
                else:
                    world = self.lab_widget.scene.camera.unproject(
                        x, y, depth, self.lab_widget.frame.width,
                        self.lab_widget.frame.height
                    )

                # This is not called on the main thread, so we need to
                # post to the main thread to safely access UI items.
                def update_selected():
                    if world is None:   # need this check as might get called before the above is run
                        return
                    # get from world coords to the nearest point in the input array
                    single_point_cloud = o3d.geometry.PointCloud()
                    single_point_cloud.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector([world]))
                    dist = self.cloud.compute_point_cloud_distance(single_point_cloud)
                    nearest_index = np.argmin(dist)
                    selected_lab = self.cloud.points[nearest_index]
                    selected_rgb = self.cloud.colors[nearest_index]
                    text = "({:.1f}, {:.1f}, {:.1f})".format(
                        selected_lab[0], selected_lab[1], selected_lab[2])

                    if nearest_index in self.selected:
                        logger.debug(f"Remove: {selected_lab}")
                        self.selected.remove(nearest_index)
                        self.lab_widget.scene.remove_geometry(text)
                    else:
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.args.delta)
                        sphere.paint_uniform_color(selected_rgb)
                        sphere.compute_vertex_normals()
                        sphere.translate(selected_lab)
                        self.lab_widget.scene.add_geometry(text, sphere, self.material)
                        logger.debug(f"Select: {selected_lab}")
                        self.selected.add(nearest_index)

                    self.info.text = text
                    self.info.visible = world is not None
                    # We are sizing the info label to be exactly the right size,
                    # so since the text likely changed width, we need to
                    # re-layout to set the new frame.
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(self.window, update_selected)

            self.lab_widget.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose image to load", self.window.theme)
        extensions = [f".{s}" for s in IMAGE_EXTENSIONS]
        dlg.add_filter(" ".join(extensions), f"Supported image files ({', '.join(extensions)}")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self.lab_widget.frame
        self.export_image(filename)

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
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("AlGrow Calibration GUI"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
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

    def load(self, path):
        self.lab_widget.scene.clear_geometry()
        logger.debug("Load image")
        image = ImageLoaded(path, self.args)
        logger.debug("Calculate points summary")

        o3d_img = o3d.geometry.Image(image.rgb.astype(np.float32))  # can't load from float64 apparently

        if self.rgb_widget is None:
            self.rgb_widget = gui.ImageWidget(o3d_img)
            self.panel.add_child(self.rgb_widget)
            self.window.add_child(self.panel)
        else:
            self.rgb_widget.update_image(o3d_img)

        logger.debug("Add to scene")
        self.cloud, self.indices = self.get_downscaled_cloud_and_indices(image)

        self.lab_widget.scene.add_geometry("points", self.cloud, self.material)
        logger.debug("Setup camera")
        bbox = self.cloud.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        # in the below 60 is default field of view, [5,0,0] is just the middle of the Lab space
        self.lab_widget.setup_camera(60, bbox, [50, 0, 0])
        self.lab_widget.look_at(center, [-200, 0, 0], [-1, 1, 0])

    #@wait_for_result
    def get_downscaled_cloud_and_indices(self, image):  # indices are the back reference to the image pixels
        cloud = o3d.geometry.PointCloud()
        logger.debug("flatten image")
        lab = image.lab.reshape(-1, 3)
        rgb = image.rgb.reshape(-1, 3)
        logger.debug("Set points")
        cloud.points = o3d.utility.Vector3dVector(lab)
        logger.debug("Set point colours")
        cloud.colors = o3d.utility.Vector3dVector(rgb)
        logger.debug("Downsample")
        cloud, _, indices = cloud.voxel_down_sample_and_trace(voxel_size=self.args.colour_rounding, min_bound=[-128, -128, 0], max_bound=[127, 127, 100])
        return cloud, indices


    def export_image(self, path):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self.lab_scene.scene.scene.render_to_image(on_image)




