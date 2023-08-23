import logging
import argparse

import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering
import platform
from scipy.ndimage import zoom
import numpy as np

from ..options.custom_types import IMAGE_EXTENSIONS
from ..image_loading import ImageLoaded
from .hull import HullHolder

from typing import Tuple, List, Dict, Set, Optional

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
        self.alpha = self.args.alpha if self.args.alpha is not None else 0
        self.delta = self.args.delta if self.args.delta is not None else 0
        self.selected = set()

        # Prepare a right_panel to display the source image
        self.em = self.window.theme.font_size
        margin = 0.5 * self.em

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
        self.labels = list()

        self.image_panel = gui.Vert(margin, gui.Margins(margin))
        self.window.add_child(self.image_panel)
        self.image_panel.add_child(gui.Label("Image RGB"))

        self.image_widget = gui.ImageWidget()
        self.image_widget.visible = False
        self.image_widget.set_on_mouse(self.on_mouse_image_widget)
        self.image_panel.add_child(self.image_widget)

        self.tool_panel = gui.Horiz(margin, gui.Margins(margin))

        self.tool_panel.visible = False
        self.image_panel.add_child(self.tool_panel)
        self.button_panel = gui.Vert(margin, gui.Margins(margin))
        self.tool_panel.add_child(self.button_panel)

        alpha_panel = gui.Horiz(margin, gui.Margins(margin))
        self.button_panel.add_child(alpha_panel)
        alpha_label = gui.Label("Alpha")
        alpha_panel.add_child(alpha_label)
        self.alpha_input = gui.TextEdit()
        self.alpha_input.placeholder_text = "Alpha"
        self.alpha_input.text_value = str(self.args.alpha)
        self.alpha_input.tooltip = "Alpha input"
        self.alpha_input.set_on_value_changed(self.update_alpha)
        alpha_panel.add_child(self.alpha_input)

        delta_panel = gui.Horiz(margin, gui.Margins(margin))
        self.button_panel.add_child(delta_panel)
        delta_label = gui.Label("Delta")
        delta_panel.add_child(delta_label)
        self.delta_input = gui.TextEdit()
        self.delta_input.placeholder_text = "Delta"
        self.delta_input.text_value = str(self.delta)
        self.delta_input.tooltip = "Delta input"
        self.delta_input.set_on_value_changed(self.update_delta)
        delta_panel.add_child(self.delta_input)

        self.show_selected_button = gui.Button("Highlight Selected")
        self.show_selected_button.toggleable = True
        self.show_selected_button.is_on = True
        self.show_selected_button.set_on_clicked(self.update_image_highlighting)
        self.button_panel.add_child(self.show_selected_button)

        self.show_within_button = gui.Button("Highlight Within")
        self.show_within_button.toggleable = True
        self.show_within_button.is_on = True
        self.show_within_button.set_on_clicked(self.update_image_highlighting)
        self.button_panel.add_child(self.show_within_button)

        self.show_hull_button = gui.Button("Show Hull")
        self.show_hull_button.toggleable = True
        self.show_hull_button.is_on = True
        self.show_hull_button.set_on_clicked(self.draw_hull)
        self.button_panel.add_child(self.show_hull_button)


        self.selection_panel = gui.ScrollableVert(margin, gui.Margins(margin))
        clear_selection = gui.Button("Clear Selection")
        clear_selection.tooltip = "Clear selected colours"
        clear_selection.set_on_clicked(self.clear_selection)
        self.selection_panel.add_child(clear_selection)
        self.tool_panel.add_child(self.selection_panel)
        self.button_pool = ButtonPool(self.selection_panel)
        self.image_panel.add_stretch()
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
            r.x, r.y, 0.5 * r.width, r.height
        )
        
        #panel_size = self.image_panel.calc_preferred_size(layout_context, self.image_panel.Constraints())
        self.image_panel.frame = gui.Rect(
            self.lab_widget.frame.get_right(), r.y, panel_width, r.height
        )
        #image_pref = self.image_widget.calc_preferred_size(layout_context, gui.Widget.Constraints())
        #self.image_widget.frame = gui.Rect(
        #   self.lab_widget.frame.get_right(), self.lab_widget.frame.y, image_pref.width, image_pref.height
        #)


        #info_pref = self.info.calc_preferred_size(layout_context,  gui.Widget.Constraints())
        #self.info.frame = gui.Rect(
        #    r.x,
        #    r.get_bottom() - info_pref.height,
        #    info_pref.width,
        #    info_pref.height
        #)


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
                    return self.image.index_to_coord(image, nearest_index)

                y, x = nearest_within(3)
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
                    nearest_index = self.get_nearest_index_from_coords(world)
                    self.toggle_voxel(nearest_index)
                    self.info.visible = True
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(self.window, update_selected)

            self.lab_widget.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def get_nearest_index_from_coords(self, coords):
        single_point_cloud = o3d.geometry.PointCloud()
        single_point_cloud.points = o3d.utility.Vector3dVector(o3d.utility.Vector3dVector([coords]))
        dist = self.cloud.compute_point_cloud_distance(single_point_cloud)
        return np.argmin(dist)

    def clear_selection(self, event=None):
        self.selected.clear()
        self.button_pool.clear()
        self.update_hull(self.cloud.select_by_index(list(self.selected)).points, self.alpha)

    def update_alpha(self, event=None):
        try:
            alpha = float(self.alpha_input.text_value)
        except ValueError:
            alpha = 0
            self.alpha = 0
            logger.debug("Could not coerce to float")
        self.alpha = alpha
        self.alpha_input.text_value = str(alpha)
        self.update_hull(alpha=alpha)

    def update_delta(self, event=None):
        try:
            delta = float(self.delta_input.text_value)
        except ValueError:
            delta = 0
            self.delta = 0
            logger.debug("Could not coerce to float")
        self.delta = delta
        self.update_image_highlighting()

    def image_widget_to_image_coords(self, event_x, event_y):
        frame_x = event_x - self.image_widget.frame.x
        frame_y = event_y - self.image_widget.frame.y
        logger.debug(f"frame coords: {frame_x, frame_y}")
        frame_fraction_x = frame_x / self.image_widget.frame.width
        frame_fraction_y = frame_y / self.image_widget.frame.height
        displayed_image_x = np.floor(self.image.displayed.shape[1] * frame_fraction_x)
        displayed_image_y = np.floor(self.image.displayed.shape[0] * frame_fraction_y)
        logger.debug(f"displayed coords: {displayed_image_x, displayed_image_y}")
        image_x = np.floor((displayed_image_x/self.image.zoom_factor)) + self.image.displayed_start_x
        image_y = np.floor((displayed_image_y/self.image.zoom_factor)) + self.image.displayed_start_y
        logger.debug(f"image coords: {image_x, image_y}")
        x = int(image_x)
        y = int(image_y)
        return x, y

    def on_mouse_image_widget(self, event):
        if event.type in [
            gui.MouseEvent.Type.BUTTON_DOWN,
            gui.MouseEvent.Type.WHEEL
        ]:
            x, y = self.image_widget_to_image_coords(event.x, event.y)

            if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                    gui.KeyModifier.SHIFT):
                logger.debug(f"Image coordinates: {x, y}")
                image_index = self.image.coord_to_index(self.image.rgb, x, y)
                logger.debug(f"Image index {image_index}")
                voxel_index = self.image_to_voxel[image_index]
                logger.debug(f"Voxel index {voxel_index}")
                self.toggle_voxel(voxel_index, remove_if_found=False)
                self.window.set_needs_layout()
                return gui.Widget.EventCallbackResult.HANDLED
            elif event.type == gui.MouseEvent.Type.WHEEL:
                self.image.zoom(x, y, event.wheel_dy)
                self.update_image_highlighting()
        return gui.Widget.EventCallbackResult.IGNORED

    def toggle_voxel(self, lab_index, remove_if_found=True):
        selected_lab = self.cloud.points[lab_index]
        selected_rgb = self.cloud.colors[lab_index]
        lab_text = "({:.1f}, {:.1f}, {:.1f})".format(
            selected_lab[0], selected_lab[1], selected_lab[2])
        self.info.text = lab_text

        if lab_index in self.selected:
            if remove_if_found:
                logger.debug(f"Remove: {selected_lab}")
                self.selected.remove(lab_index)
                self.lab_widget.scene.remove_geometry(lab_text)
                self.button_pool.remove_button(lab_index)
                self.update_hull(self.cloud.select_by_index(list(self.selected)).points, self.alpha)
            else:
                logger.debug(f"Found: {selected_lab}")
        else:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.args.delta)
            sphere.paint_uniform_color(selected_rgb)
            sphere.compute_vertex_normals()
            sphere.translate(selected_lab)
            self.lab_widget.scene.add_geometry(lab_text, sphere, self.material)
            self.selected.add(lab_index)
            self.add_button(lab_index, lab_text, selected_rgb)
        logger.debug("get selected points to update hull holder")
        self.update_hull(self.cloud.select_by_index(list(self.selected)).points, self.alpha)

    def add_button(self, lab_index: int, lab_text: str, rgb):
        b = gui.Button(lab_text)
        b.background_color = gui.Color(*rgb)
        self.button_pool.add_button(lab_index, b)
        b.set_on_clicked(lambda: self.toggle_voxel(lab_index, remove_if_found=True))

    def update_hull(self, points=None, alpha=None):
        if points is not None:
            logger.debug("Update hull holder")
            if alpha is None:
                alpha = self.alpha
            self.hull_holder = HullHolder(points, alpha)
        elif alpha is not None and self.hull_holder is not None:
            logger.debug("Update alpha")
            self.hull_holder.update_alpha(alpha)
        self.draw_hull()
        self.update_image_highlighting()

    def update_image_highlighting(self):
        if self.image is None:
            self.image_widget.visible = False
            self.tool_panel.visible = False
        else:
            if self.show_selected_button.is_on:
                selected = [j for i in list(self.selected) for j in self.indices[i]]
            else:
                selected = []
            if self.show_within_button.is_on and self.hull_holder is not None and self.hull_holder.mesh is not None:
                logger.debug(f"Calculate distance from hull for image with: {self.image.rgb.reshape(-1, 3).shape[0]} pixels")
                distances = self.hull_holder.get_distances(self.image.lab)
                if distances is not None:
                    logger.debug("Apply mask to selected")
                    within = np.where(distances.reshape(-1) <= self.delta)[0]
                else:
                    within = []

            else:
                within = []

            mask_indices = self.image.indices_in_displayed(np.unique(np.append(selected, within)).astype(int))
            displayed = self.image.displayed.copy()  # make a copy
            displayed.reshape(-1, 3)[mask_indices] = self.image.highlight_colour
            self.image_widget.update_image(o3d.geometry.Image(displayed.astype(np.float32)))
            self.image_widget.visible = True
            self.tool_panel.visible = True

    def draw_hull(self):
        if self.lab_widget.scene is not None:
            logger.debug("Remove existing mesh")
            self.lab_widget.scene.remove_geometry('mesh')
        if self.show_hull_button.is_on and self.hull_holder is not None and self.hull_holder.mesh is not None:
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

    def load(self, path):
        self.lab_widget.scene.clear_geometry()
        for label in self.labels:
            self.lab_widget.remove_3d_label(label)

        logger.debug("Load image")
        self.image = ZoomableImage(ImageLoaded(path, self.args))
        logger.debug("Load rgb image")
        self.image_widget.update_image(self.image.as_o3d)
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
            label:  gui.Label3D = self.lab_widget.add_3d_label(point, str(axis[0]))
            self.labels.append(label)
            label.color = gui.Color(1, 1, 1)
        [lineset.lines.remove(line) for line in to_remove]
        self.lab_widget.scene.add_geometry("lines", lineset, bbox_material)
        # in the below 60 is default field of view, [5,0,0] is just the middle of the Lab space
        self.lab_widget.setup_camera(60, bbox, [50, 0, 0])
        self.lab_widget.look_at(center, [-200, 0, 0], [-1, 1, 0])
        self.image_widget.visible = True
        self.tool_panel.visible = True

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


class ButtonPool:
    def __init__(self, selection_panel: gui.Vert):
        self.selection_panel = selection_panel
        self.buttons: Dict[str, gui.WidgetProxy] = dict()

    def add_button(self, lab_index, button: gui.Button):
        if lab_index in self.buttons.keys():
            logger.debug(f"Replacing existing button")
            self.buttons[lab_index].set_widget(button)
            return

        # check for hidden buttons to replace
        for existing_index, button_proxy in self.buttons.items():
            if not button_proxy.enabled:  # ok to replace
                logger.debug(f"Replacing {existing_index} with {lab_index}")
                button_proxy.set_widget(button)
                self.buttons[lab_index] = button_proxy
                del self.buttons[existing_index]
                return

        # finally if none to be replaced we add a new one
        logger.debug(f"Adding a new button for {lab_index}")
        self.buttons[lab_index] = gui.WidgetProxy()
        self.buttons[lab_index].set_widget(button)
        self.selection_panel.add_child(self.buttons[lab_index])

    def remove_button(self, lab_index):
        if lab_index in self.buttons:
            logger.debug(f"Disabling button {lab_index}")
            self.buttons[lab_index].enabled = False
            self.buttons[lab_index].visible = False
        else:
            logger.debug(f"Button not found {lab_index}")

    def clear(self):
        for name, button in self.buttons.items():
            button.enabled = False
            button.visible = False


class ZoomableImage:  # an adapter to allow zooming and easier loading

    def __init__(self, image: ImageLoaded):
        self._image = image
        self.displayed = self._image.rgb.copy()
        self.zoom_factor = 1
        self.displayed_start_x = 0
        self.displayed_start_y = 0

        self.highlight_colour = [1, 0, 1]  # todo make this colour configurable

    @property
    def lab(self):
        return self._image.lab

    @property
    def rgb(self):
        return self._image.rgb

    @property
    def as_o3d(self):
        return o3d.geometry.Image(self.displayed.astype(np.float32))

    def zoom(self, x, y, factor_increment: int = 1):
        self.zoom_factor += factor_increment
        self.zoom_factor = 1 if self.zoom_factor < 1 else self.zoom_factor
        # factor should be integer,
        # we don't want interpolation for now
        cropped_width = int(self._image.rgb.shape[1] / self.zoom_factor)
        cropped_height = int(self._image.rgb.shape[0] / self.zoom_factor)
        logger.debug(f"x:{x}, y:{y}, zoom_factor{self.zoom_factor}")
        logger.debug(f"cropped_width: {cropped_width}, cropped_height: {cropped_height}")
        if x < cropped_width/2:
            zoom_start_x = 0
        elif x > (self._image.rgb.shape[1] - (cropped_width/2)):
            zoom_start_x = self._image.rgb.shape[1] - cropped_width
        else:
            zoom_start_x = x - (cropped_width/2)
        if y < cropped_height/2:
            zoom_start_y = 0
        elif y > (self._image.rgb.shape[0] - (cropped_height/2)):
            zoom_start_y = self._image.rgb.shape[0] - cropped_height
        else:
            zoom_start_y = y - (cropped_height/2)
        self.displayed_start_x = int(zoom_start_x)
        self.displayed_start_y = int(zoom_start_y)
        logger.debug(f"displayed_start_x = {self.displayed_start_x}, displayed_start_y = {self.displayed_start_y}")
        cropped = self._image.rgb[
                  self.displayed_start_y:self.displayed_start_y + cropped_height,
                  self.displayed_start_x:self.displayed_start_x + cropped_width
                  ]
        logger.debug(f"cropped_shape: {cropped.shape}")
        cropped_rescaled = zoom(cropped, (self.zoom_factor, self.zoom_factor, 1), order=0, grid_mode=True, mode='nearest')
        self.displayed = cropped_rescaled

    def reload_displayed(self):
        self.displayed = self._image.rgb.copy()
        cropped_width = int(self._image.rgb.shape[1] / self.zoom_factor)
        cropped_height = int(self._image.rgb.shape[0] / self.zoom_factor)
        cropped = self._image.rgb[
                  self.displayed_start_y:self.displayed_start_y + cropped_height,
                  self.displayed_start_x:self.displayed_start_x + cropped_width
                  ]
        #cropped_rescaled = rescale(cropped, self.zoom_factor, channel_axis=2)
        cropped_rescaled = zoom(cropped, (self.zoom_factor, self.zoom_factor, 1), order=0, grid_mode=True, mode='nearest')
        self.displayed = cropped_rescaled
        return self.displayed

    def indices_in_displayed(self, selected: List[int]) -> List[int]:
        selected_in_displayed = [self.image_index_to_displayed_indices(i) for i in selected]
        return [j for i in selected_in_displayed if i is not None for j in i]

    @staticmethod
    def coord_to_index(image, x, y) -> int:
        if all([x >= 0, x < image.shape[1], y >= 0, y < image.shape[0]]):
            return (y * image.shape[1]) + x
        else:
            raise ValueError("Coordinates are outside of image")

    @staticmethod
    def index_to_coord(image, i: int) -> Tuple[int, int]:  # in y, x order
        return np.unravel_index(i, image.shape[0:2])
        #x_length = image.shape[1]
        #return i % x_length, int(np.floor(i / x_length))  # in x,y order

    def image_index_to_displayed_indices(self, image_index: int) -> Optional[List[int]]:  # can be many due to zooming but at least one
        image_y, image_x = self.index_to_coord(self.rgb, image_index)
        displayed_x = (image_x - self.displayed_start_x) * self.zoom_factor
        displayed_y = (image_y - self.displayed_start_y) * self.zoom_factor
        try:
            starting_coord = self.coord_to_index(self.displayed, displayed_x, displayed_y)
            logger.debug(f"starting_coord {starting_coord}")
            # need to handle expansion due to zooming
            if self.zoom_factor % 2:
                #pixel_expansion = range(int((-self.zoom_factor+1)/2), int((self.zoom_factor-1)/2))
                pixel_expansion = range(0, self.zoom_factor)
            else:
                pixel_expansion = range(-1, self.zoom_factor)
                #pixel_expansion = range(int((-self.zoom_factor/2)+1), int(self.zoom_factor/2))

            #x_expanded_coords = [starting_coord + i for i in range(0, self.zoom_factor)] + [starting_coord - i for i in range(0, self.zoom_factor)]
            x_expanded_coords = [starting_coord + i for i in pixel_expansion]
            #logger.debug(f"x_expanded {x_expanded_coords}")
            #all_expanded_coords = [j + self.displayed.shape[1] * i for i in range(0, self.zoom_factor) for j in x_expanded_coords] + [j - self.displayed.shape[1] * i for i in range(0, self.zoom_factor) for j in x_expanded_coords]
            all_expanded_coords = [j + self.displayed.shape[1] * i for i in pixel_expansion for j in x_expanded_coords]
            #logger.debug(f"all_expanded :{all_expanded_coords}")
            return all_expanded_coords
        except ValueError:
            return None
