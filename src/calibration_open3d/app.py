import logging
import argparse

import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering
import platform
from skimage.color import lab2rgb
import numpy as np

from ..options.custom_types import IMAGE_EXTENSIONS
from ..image_loading import ImageLoaded
from .hull import HullHolder

from typing import Tuple

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
        self.image = None
        self.cloud = None
        self.indices = None
        self.image_to_voxel = None
        self.hull_holder = None
        self.alpha = self.args.alpha
        self.delta = self.args.delta
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
        self.material.point_size = 2 * self.window.scaling
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
        self.image_widget = gui.ImageWidget()
        self.image_widget.set_on_mouse(self.on_mouse_rgb_widget)
        self.panel.add_child(self.image_widget)
        self.window.add_child(self.panel)

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

                # have a reasonable selection radius and just get the closest point within that
                def closest_within(radius):
                    image = np.asarray(depth_image)
                    ys = np.arange(0, image.shape[0])
                    xs = np.arange(0, image.shape[1])
                    mask = (xs[np.newaxis,:]-x)**2 + (ys[:,np.newaxis]-y)**2 >= radius**2
                    masked_array = np.ma.masked_array(image, mask=mask)
                    closest_index = masked_array.argmin(fill_value=1)
                    return self.index_to_coord(image, closest_index)

                y, x = closest_within(3)
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
                        self.info.visible = False
                        return
                    # get from world coords to the nearest point in the input array
                    single_point_cloud = o3d.geometry.PointCloud()
                    single_point_cloud.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector([world]))
                    dist = self.cloud.compute_point_cloud_distance(single_point_cloud)
                    nearest_index = np.argmin(dist)
                    self.toggle_voxel(nearest_index)
                    self.info.visible = True
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(self.window, update_selected)

            self.lab_widget.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    @staticmethod
    def coord_to_index(image, x, y) -> int:
        x_length = image.shape[1]
        return (y * x_length) + x

    @staticmethod
    def index_to_coord(image, i) -> Tuple[int, int]:  # in y, x order
        return np.unravel_index(i, image.shape)
        #x_length = image.shape[1]
        #return i % x_length, int(np.floor(i / x_length))  # in x,y order

    def on_mouse_rgb_widget(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            frame_x = event.x - self.image_widget.frame.x
            frame_y = event.y - self.image_widget.frame.y
            frame_fraction_x = frame_x / self.image_widget.frame.width
            frame_fraction_y = frame_y / self.image_widget.frame.height
            image_x = self.image.rgb.shape[1] * frame_fraction_x
            image_y = self.image.rgb.shape[0] * frame_fraction_y
            x = int(np.around(image_x, decimals=0))
            y = int(np.around(image_y, decimals=0))
            logger.debug(f"Image coordinates: {x, y}")
            image_index = self.coord_to_index(self.image.rgb, x, y)
            logger.debug(f"Image index {image_index}")
            voxel_index = self.image_to_voxel[image_index]
            logger.debug(f"Voxel index {voxel_index}")
            self.toggle_voxel(voxel_index)
            self.window.set_needs_layout()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def toggle_voxel(self, index):
        selected_lab = self.cloud.points[index]
        selected_rgb = self.cloud.colors[index]
        text = "({:.1f}, {:.1f}, {:.1f})".format(
            selected_lab[0], selected_lab[1], selected_lab[2])
        self.info.text = text

        if index in self.selected:
            logger.debug(f"Remove: {selected_lab}")
            self.selected.remove(index)
            self.lab_widget.scene.remove_geometry(text)
        else:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.args.delta)
            sphere.paint_uniform_color(selected_rgb)
            sphere.compute_vertex_normals()
            sphere.translate(selected_lab)
            self.lab_widget.scene.add_geometry(text, sphere, self.material)
            logger.debug(f"Select: {selected_lab}")
            self.selected.add(index)
        logger.debug("get selected points to update hull holder")
        self.update_hull(self.cloud.select_by_index(list(self.selected)).points, self.alpha)

    def update_hull(self, points=None, alpha=None):
        logger.debug("Update hull holder")
        if points is not None:
            self.hull_holder = HullHolder(points, alpha)
        elif alpha is not None:
            self.hull_holder.update_alpha(alpha)
        self.draw_hull()
        self.update_image()

    def update_image(self):
        image32 = self.image.rgb.astype(np.float32)
        if self.hull_holder.mesh is not None:
            logger.debug(f"Calculate distance from hull for image with: {image32.reshape(-1, 3).shape[0]} pixels")
            distances = self.hull_holder.get_distances(self.image.lab)

            image32.reshape(-1, 3)[distances <= self.delta] = [1, 0, 1]  # todo configure colour for selection in gui
        o3d_img = o3d.geometry.Image(image32)
        self.image_widget.update_image(o3d_img)

    def draw_hull(self):
        if self.lab_widget.scene is not None and self.hull_holder.mesh is not None:
            self.lab_widget.scene.remove_geometry('mesh')
            logger.debug("Add mesh to scene")
            self.lab_widget.scene.add_geometry("mesh", self.hull_holder.mesh, self.material)
            
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
        self.image = ImageLoaded(path, self.args)
        logger.debug("Load rgb image")

        o3d_img = o3d.geometry.Image(self.image.rgb.astype(np.float32))  # can't load from float64 apparently
        self.image_widget.update_image(o3d_img)

        logger.debug("Get point cloud")
        self.cloud, self.indices = self.get_downscaled_cloud_and_indices(self.image)
        # need to build a reverse mapping from image index to voxel in cloud
        logger.debug("Build map from image to voxel")  #todo refactor this, it is slow
        self.image_to_voxel = dict()

        for i, jj in enumerate(self.indices):
            for j in jj:
                self.image_to_voxel[j] = i
        logger.debug("Add point cloud to scene")
        self.lab_widget.scene.add_geometry("points", self.cloud, self.material)
        logger.debug("Setup camera")

        bbox = self.cloud.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        bbox_geom = o3d.geometry.OrientedBoundingBox().create_from_axis_aligned_bounding_box(bbox)
        bbox_material = o3d.visualization.rendering.MaterialRecord()
        bbox_material.shader = "unlitLine"
        bbox_material.line_width = 0.2 * self.window.theme.font_size
        lineset = o3d.geometry.LineSet().create_from_oriented_bounding_box(bbox_geom)
        to_remove = np.unique([np.asarray(line) for line in lineset.lines if ~np.any(line == 0)], axis=1)
        for i in range(1,4):
            point = lineset.points[i]
            axis = np.array(["L*", "a*", "b*"])[point != lineset.points[0]]
            label:  o3d.visualization.gui.Label3D = self.lab_widget.add_3d_label(point, str(axis[0]))
            label.color = o3d.visualization.gui.Color(1, 1, 1)
        [lineset.lines.remove(line) for line in to_remove]
        self.lab_widget.scene.add_geometry("lines", lineset, bbox_material)
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
        cloud, _, indices = cloud.voxel_down_sample_and_trace(voxel_size=self.args.colour_rounding, min_bound=[0, -128, -128], max_bound=[100, 127, 127])
        return cloud, indices

    def export_image(self, path):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self.lab_widget.scene.scene.render_to_image(on_image)



