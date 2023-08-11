import logging
from typing import List, Tuple, Optional

import wx
from pubsub import pub

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigureCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar
)
from matplotlib.figure import Figure
from matplotlib import get_data_path  # we are recycling some matplotlib icons

from alphashape import alphashape, optimizealpha
from trimesh import (
    PointCloud,
    Trimesh
)
import open3d as o3d

from ..image_loading import ImageLoaded
from ..layout import Layout
from ..figurebuilder import FigureMatplot, FigureNone
from ..options.update_and_verify import update_arg
from ..options.custom_types import DebugEnum

from .loading import Points, LayoutMultiLoader, wait_for_results
from ..logging import worker_log_configurer

logger = logging.getLogger(__name__)


class HullPanel(wx.Panel):

    def __init__(self, parent, images: List[ImageLoaded], points: Points, layouts: List[Optional[Layout]]):
        super().__init__(parent)
        self.parent = parent
        self.images = images
        self.args = self.images[0].args
        self.points = points

        layouts_to_load = [i for i, layout in enumerate(layouts) if layout is None]
        logger.debug(f"{layouts_to_load}")
        if layouts_to_load:
            layoutmultiloader = LayoutMultiLoader([image for i, image in enumerate(self.images) if i in layouts_to_load])
            for j in layouts_to_load:
                layouts[layouts_to_load[j]] = layoutmultiloader.layouts[j]

        # Prepare an index for navigating the list of images
        self.ind: int = 0

        # prepare an alpha selection object -
        # this uses the selected uniq_points to describe a hull and decide which uniq_points are within it
        self.alpha_selection = None

        self.alpha_text = None
        self.delta_text = None
        self.clear_btn = None
        self.close_btn = None
        self.patches = []

        # add a close button to test remote closing
        self.Bind(wx.EVT_CLOSE, self.on_exit)

        # Prepare the figures and toolbars
        # we are using two figures rather than one due to a bug in imshow that prevents efficient animation
        # https://github.com/matplotlib/matplotlib/issues/18985
        # by keeping them separate, changing the view on one doesn't force a redraw of the other
        self.image_fig = Figure()
        self.image_fig.set_dpi(150)
        self.image_ax = self.image_fig.add_subplot(111)
        self.image_fig.suptitle('Left-click to select', fontsize=10)
        self.image_cv = FigureCanvas(self, -1, self.image_fig)
        self.image_art = None
        self.image_nav_toolbar = NavigationToolbar(self.image_cv)
        self.image_nav_toolbar.Realize()
        image_sizer = wx.BoxSizer(wx.VERTICAL)
        image_sizer.Add(self.image_nav_toolbar, 0, wx.ALIGN_CENTER)
        image_sizer.Add(self.image_cv, 1, wx.EXPAND)
        self.image_nav_toolbar.update()
        self.click_image = self.image_cv.mpl_connect('pick_event', self.on_click_image)

        self.lab_fig = Figure()
        self.lab_fig.set_dpi(100)
        self.lab_ax = self.lab_fig.add_subplot(111, projection='3d', facecolor='grey')
        # disable zoom so can use left click (button 1) for select
        self.lab_ax.mouse_init(rotate_btn=3, pan_btn=2, zoom_btn=0)
        self.lab_fig.suptitle("Left-click to select\nRight click to rotate", fontsize=16)
        self.lab_ax.set_zlabel('L*')
        self.lab_ax.set_xlabel('a*')
        self.lab_ax.set_ylabel('b*')
        self.lab_cv = FigureCanvas(self, -1, self.lab_fig)
        self.lab_art = None
        self.tri_art = None
        self.lab_nav_toolbar = NavigationToolbar(self.lab_cv)
        self.lab_nav_toolbar.Realize()
        self.click_lab = self.lab_cv.mpl_connect('pick_event', self.on_click_lab)

        lab_sizer = wx.BoxSizer(wx.VERTICAL)
        lab_sizer.Add(self.lab_nav_toolbar, 0, wx.ALIGN_CENTER)
        lab_sizer.Add(self.lab_cv, 1, wx.EXPAND)
        self.lab_nav_toolbar.update()

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        figure_sizer = wx.BoxSizer(wx.HORIZONTAL)
        figure_sizer.Add(image_sizer, proportion=1, flag=wx.LEFT | wx.TOP | wx.EXPAND)
        figure_sizer.Add(lab_sizer, proportion=1, flag=wx.RIGHT | wx.TOP | wx.EXPAND)
        self.sizer.Add(figure_sizer, proportion=1, flag=wx.ALIGN_CENTER)
        self.SetSizer(self.sizer)
        #self.Fit()

        self.navigation_toolbar = None
        self.toolbar = None
        self.add_toolbar()

        self.reset_selection(None)
        #self.toggle_args_vertices()
        self.load_current_image()
        logger.debug("init complete")

    def on_exit(self, event):
        logger.debug("Close called")
        if self.alpha_selection.hull is None:
            raise ValueError("Calibration not complete - please start again and select more than 4 uniq_points")

        update_arg(self.args, 'alpha', self.alpha_selection.alpha)
        update_arg(self.args, "hull_vertices", list(map(tuple, np.round(
            self.alpha_selection.hull.vertices,
            decimals=1
        ).tolist())))

        update_arg(self.args, 'delta', self.alpha_selection.delta)

        # drop the previously stored arguments from the segmentor (the new args will be loaded if we open again)
        pub.sendMessage("enable_btns")
        plt.close()
        self.Destroy()

    def add_toolbar(self):
        self.toolbar = wx.ToolBar(self, id=-1, style=wx.TB_HORIZONTAL)  # | wx.TB_TEXT)
        self.toolbar.AddTool(
            1,
            "prev",
            wx.Image(str(Path(get_data_path(), "images", "back.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            'Previous Image',
            'Load Previous Image',
            None
        )
        self.Bind(wx.EVT_TOOL, self.load_prev, id=1)
        self.toolbar.EnableTool(1, False)  # start on 0 so not enabled
        self.toolbar.AddTool(
            2,
            "next",
            wx.Image(str(Path(get_data_path(), "images", "forward.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            'Next Image',
            'Load Next Image',
            None
        )
        self.Bind(wx.EVT_TOOL, self.load_next, id=2)
        self.toolbar.EnableTool(2, len(self.images) > 1)  # enable only if more than one image
        self.toolbar.AddSeparator()
        self.toolbar.AddTool(
            3,
            "Optimise alpha",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "Alpha.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "Optimise alpha",
            'Optimise alpha',
            None
        )
        self.Bind(wx.EVT_TOOL, self.optimise_alpha, id=3)
        self.alpha_text = wx.TextCtrl(self.toolbar, 4, str(self.args.alpha), style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_TEXT_ENTER, self.set_alpha, id=4)
        self.toolbar.AddControl(self.alpha_text)
        self.toolbar.AddSeparator()
        self.toolbar.AddTool(
            5,
            "Delta",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "Delta.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.Image(str(Path(Path(__file__).parent, "bmp", "Delta.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.ITEM_NORMAL,
            "Delta",
            'Delta Input',
            None
        )
        self.toolbar.EnableTool(5, False)
        self.delta_text = wx.TextCtrl(self.toolbar, 6, str(self.args.delta),  style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_TEXT_ENTER, self.set_delta, id=6)
        self.toolbar.AddControl(self.delta_text)
        self.toolbar.AddCheckTool(
            7,
            "selected",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "selected.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Highlight selected",
            'Highlight selected pixels'
        )
        self.toolbar.ToggleTool(7, True)
        self.Bind(wx.EVT_TOOL, self.draw_image_figure, id=7)
        self.toolbar.AddCheckTool(
            8,
            "within",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "within.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Highlight within",
            'Highlight pixels within delta of alpha shape'
        )
        self.toolbar.ToggleTool(8, True)
        self.Bind(wx.EVT_TOOL, self.draw_image_figure, id=8)

        #self.toolbar.AddCheckTool(
        #    9,
        #    "boundaries",
        #    wx.Image(str(Path(Path(__file__).parent, "bmp", "boundaries.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
        #    wx.NullBitmap,
        #    "Display boundaries",
        #    'Display segment boundaries'
        #)
        #self.toolbar.ToggleTool(9, True)
        #self.Bind(wx.EVT_TOOL, self.draw_segments_figure, id=9)

        # add a clear selection button
        self.toolbar.AddSeparator()
        self.clear_btn = wx.Button(self.toolbar, 10, "Clear")
        self.toolbar.AddControl(self.clear_btn)
        self.clear_btn.Bind(wx.EVT_BUTTON, self.reset_selection)


        # add button to toggle using the hull vertices supplied as args
        self.toolbar.AddSeparator()
        self.toolbar.AddCheckTool(
            11,
            "args",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "args.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Select points supplied as arguments",
            "Select points supplied as arguments"
        )
        self.toolbar.ToggleTool(11, False)
        self.Bind(wx.EVT_TOOL, self.toggle_args_vertices, id=11)
        if self.args.hull_vertices is None:
            self.toolbar.EnableTool(11, False)

        self.toolbar.AddTool(
            20,
            "Print",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "args.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "Print animation",
            "Print animated gif of rotating hull",
            None
        )
        self.Bind(wx.EVT_TOOL, self.wait_for_animation, id=20)

        #add button to toggle using the hull vertices supplied as args
        self.toolbar.AddSeparator()
        self.toolbar.AddCheckTool(
            12,
            "all_points",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "all_points.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Display all points (all images and arguments)",
            "Display all points (all images and arguments)"
        )
        self.toolbar.ToggleTool(12, False)
        self.Bind(wx.EVT_TOOL, self.toggle_all_points, id=12)

        # add a close button
        self.toolbar.AddSeparator()
        self.close_btn = wx.Button(self.toolbar, 13, "Save and close")
        self.toolbar.AddControl(self.close_btn)
        self.close_btn.Bind(wx.EVT_BUTTON, self.on_exit)

        self.sizer.Add(self.toolbar, 0, wx.ALIGN_CENTER)
        self.toolbar.Realize()

    def load_prev(self, _):
        logger.debug("Load previous image")
        if self.ind > 0:
            self.ind -= 1
            self.load_current_image()
            self.toolbar.EnableTool(2, True)
        if self.ind == 0:
            self.toolbar.EnableTool(1, False)
        else:
            self.toolbar.EnableTool(1, True)

    def load_next(self, _):
        logger.debug("Load next image")
        if self.ind < len(self.images) - 1:
            self.ind += 1
            self.load_current_image()
            self.toolbar.EnableTool(1, True)
        if self.ind == len(self.images) - 1:
            self.toolbar.EnableTool(2, False)
        else:
            self.toolbar.EnableTool(2, True)

    def load_current_image(self):
        filepath = self.images[self.ind].filepath
        logger.debug(f"Load image: {filepath}")
        self.image_ax.set_title(str(filepath), y=-0.01, size=5)
        self.draw_image_figure()
        self.draw_lab_figure()

    def optimise_alpha(self, _):
        self.alpha_selection.update_alpha()
        self.draw_image_figure()
        self.draw_lab_figure()
        self.alpha_text.SetValue(str(self.alpha_selection.alpha))

    def set_alpha(self, _):
        value = self.alpha_text.GetValue()
        logger.debug(f"Set alpha: {value}")
        try:
            value = float(value)
        except ValueError:
            logger.debug(f"Alpha input could not be coerced to float: {value}")
            self.alpha_text.SetValue(str(self.args.alpha))
        self.alpha_selection.update_alpha(value)
        self.draw_image_figure()
        self.draw_lab_figure()

    def set_delta(self, _):  #
        value = self.delta_text.GetValue()
        logger.debug(f"Set delta: {value}")
        try:
            value = float(value)
        except ValueError:
            logger.debug(f"Delta input could not be coerced to float: {value}")
            self.delta_text.SetValue(self.args.alpha)
        self.alpha_selection.set_delta(value)
        self.draw_image_figure()

    def coord_to_pixel(self, x, y) -> int:
        x_length = self.images[self.ind].lab.shape[1]
        return (y * x_length) + x

    def pixel_to_coord(self, index) -> Tuple[int, int]:
        x_length = self.images[self.ind].lab.shape[1]
        return index % x_length, int(np.floor(index/x_length))

    def pixel_to_lab(self, pixel) -> Tuple[Path, int]:
        return tuple(self.points.pixel_to_lab.loc[(self.images[self.ind].filepath, pixel)].to_numpy())

    def lab_to_pixels(self, lab, filepath: Optional[Path] = None) -> List[Tuple[Path, int]]:
        if filepath is None:
            return [tuple(x) for x in self.points.lab_to_pixel.loc[lab].to_numpy()]
        else:
            return [tuple([filepath, *x]) for x in self.points.filepath_lab_to_pixel.loc[(filepath, *lab)].to_numpy()]

    def on_click_image(self, event):
        # Left click only
        if event.mouseevent.button == 1:
            x = int(np.around(event.mouseevent.xdata, decimals=0))
            y = int(np.around(event.mouseevent.ydata, decimals=0))
            logger.debug(f"x: {x}, y: {y}")
            image = self.images[self.ind]
            pixel = self.coord_to_pixel(x, y)
            lab = self.pixel_to_lab(pixel)
            self.alpha_selection.toggle_colour(lab)
            self.draw_image_figure()
            self.draw_hull()

    def on_click_lab(self, event):
        # Left click only (right click can be used to rotate view)
        if event.mouseevent.button == 1:
            lab_plot_index = event.ind[0]
            all_points = self.toolbar.GetToolState(11)
            if all_points:
                
                points.counts_all or points.counts_per_image  depending on *all_points or not

            image = self.images[self.ind]
            self.alpha_selection.toggle_colour(image.filepath, unique_index)
            self.draw_image_figure()
            self.draw_hull()


    def toggle_args_vertices(self, event=None):
        if self.args.hull_vertices is not None:
            show_args = self.toolbar.GetToolState(11)  # true if show args is selected in gui
            self.alpha_selection.toggle_args_vertices(show_args)
            self.draw_image_figure()
            self.draw_hull()

    def toggle_all_points(self, event=None):
        self.draw_lab_figure()


    def draw_image_figure(self, _=None):
        logger.debug("Draw image with highlighting")
        show_selected = self.toolbar.GetToolState(7)  # true if highlight selection is selected in gui
        logger.debug(f"show selected: {show_selected}")
        show_within = self.toolbar.GetToolState(8)  # true if highlight within is selected in gui
        logger.debug(f"show within: {show_within}")

        image = self.images[self.ind]
        inv = self.points.lab_inv[image.filepath]
        displayed = image.rgb.copy()

        if show_selected:
            selected = [(i, j) for i, j in self.alpha_selection.selection if i == image.filepath]
            selected_real = [i for j in selected for i in self.unique_index_to_coord_indices(*j)]
            displayed.reshape(-1, 3)[selected_real, :] = (1, 1, 1)

        if show_within:
            self.alpha_selection.update_dist(image.filepath)
            dist = self.alpha_selection.dist[image.filepath]
            within = dist <= self.alpha_selection.delta
            within_real = within[inv].reshape(-1)
            displayed.reshape(-1, 3)[within_real, :] = (1, 1, 1)  # todo make these colours variables, either as arguments or some other method

        if self.image_art is None:
            self.image_art = self.image_ax.imshow(displayed, picker=True)
        else:
            self.image_art.set_data(displayed)

        self.image_cv.draw_idle()
        self.image_cv.flush_events()

    def draw_lab_figure(self, _=None):
        logger.debug("Draw lab figure")
        #show_all = self.toolbar.GetToolState(12)
        image = self.images[self.ind]
        #if show_all:
        #    lab = np.concatenate(list(self.points.lab_unique.values()))  # todo handle these for uniqueness also including indices
        #    rgb = np.concatenate(list(self.points.rgb_unique.values()))
        #    sizes = np.concatenate(list(self.points.lab_sizes.values()))
        #else:
        lab = self.points.lab_unique[image.filepath]
        rgb = self.points.rgb_unique[image.filepath]
        sizes = self.points.lab_sizes[image.filepath]

        # get the current view angles on lab plot to reload with after redraw
        elev, azim = self.lab_ax.elev, self.lab_ax.azim

        if self.lab_art is not None:
            self.lab_art.remove()
            self.lab_art = None
        self.lab_art = self.lab_ax.scatter(
            xs=lab[:, 1],
            ys=lab[:, 2],
            zs=lab[:, 0],
            #s=10,
            s=sizes,
            edgecolor=(0,0,0),
            facecolor=rgb,
            alpha=0.9,
            lw=0,
            picker=True,
            pickradius=0.1
        )
        self.lab_ax.view_init(elev=elev, azim=azim)
        self.draw_hull()
        logger.debug("draw lab figure complete")

    def draw_hull(self):
        logger.debug("Draw hull")
        if self.tri_art is not None:
            logger.debug("remove existing alpha hull from plot")
            self.tri_art.remove()
            self.tri_art = None

        if self.alpha_selection.hull is not None:
            logger.debug("draw alpha hull on plot")
            self.tri_art = self.lab_ax.plot_trisurf(
                *zip(*self.alpha_selection.hull.vertices[:, [1, 2, 0]]),
                triangles=self.alpha_selection.hull.faces[:, [1, 2, 0]],
                color=(0, 0, 0, 1)
            )
        logger.debug("draw hull complete")
        self.lab_cv.draw_idle()
        logger.debug("draw idle complete")
        self.lab_cv.flush_events()
        logger.debug("flush complete")

    def wait_for_animation(self, _=None):
        kwargs_list = [{
            "args": self.args,
            "points": self.points,
            "hull_vertices": self.alpha_selection.hull.vertices if self.alpha_selection.hull else None,
            "alpha": self.alpha_selection.alpha,
            "counter": self.parent.figure_counter
        }]
        wait_for_results("Preparing animation", 1, plot_hull_animation, kwargs_list)

    def reset_selection(self, _=None):
        self.alpha_selection = AlphaSelection(
            self.points.lab_unique,
            set(),
            alpha=self.args.alpha,
            delta=self.args.delta
        )
        self.toolbar.ToggleTool(11, False)
        self.draw_image_figure()
        self.draw_hull()


def plot_hull_animation(args, points, hull_vertices, alpha, counter, log_queue=None):
    if log_queue is not None:
        worker_log_configurer(log_queue)

    if hull_vertices is not None:
        if len(hull_vertices) < 4:
            hull = None
        else:
            logger.debug("Calculating hull")
            if alpha is None or alpha == 0:
                logger.debug("creating convex hull")
                # the api for alphashape is a bit strange,
                # it returns a shapely polygon when alpha is 0
                # rather than a trimesh object which is returned for other values of alpha
                # so just calculate the convex hull with trimesh to ensure we get a consistent return value
                hull = PointCloud(hull_vertices).convex_hull
            else:
                logger.debug("creating alpha shape")
                hull = alphashape(hull_vertices, alpha)
                if len(hull.faces) == 0:
                    logger.debug("More points required for a complete hull with current alpha value")
                    hull = None
    else:
        hull = None

    logger.debug("Plotting animation of hull")
    fig = FigureMatplot("Hull", counter, args, cols=1)
    abl = np.concatenate(list(points.lab_unique.values()))[:, [1, 2, 0]]
    rgb = np.concatenate(list(points.rgb_unique.values()))
    sizes = np.concatenate(list(points.lab_sizes.values()))
    labels = ("a*", "b*", "L*")
    fig.plot_scatter_3d(abl, labels, rgb, sizes, hull)
    fig.animate()


class AlphaSelection:
    def __init__(
            self,
            points,
            selection: set[tuple[Path, int]],
            alpha: float,
            delta: float
    ):
        self.uniq_points = points
        # uniq_points is a dict with image filepath as key, value is ndarray of images reshaped to (-1,3)
        # filepath "args" is a special entry describing the input hull vertices supplied as arguments

        self.selection = selection  # a set of (filepath, index) for the uniq_points dataframe
        self.last_selected = None
        self.alpha = alpha  # the alpha parameter for hull construction
        self.delta = delta  # the distance from the hull surface to consider a point within the hull

        self.dist = {k: np.full(v.shape[0], np.inf) for k, v in self.uniq_points.items()}

        self.hull: Optional[Trimesh] = None

    def update_alpha(self, alpha: float = None):
        if alpha is None:
            if len(self.selection) >= 4:
                logger.debug(f"optimising alpha")
                selected = [self.uniq_points[fp][ind].values for fp, ind in self.selection]
                self.alpha = round(optimizealpha(selected), ndigits=3)
                logger.info(f"optimised alpha: {self.alpha}")
            else:
                logger.debug(f"Insufficient uniq_points selected")
        else:
            self.alpha = alpha
        self.update_hull()

    def set_delta(self, delta: float):
        logger.debug(f"Set delta: {delta}")
        self.delta = delta

    def toggle_colour(self, lab: np.ndarray):
        if lab in self.selection:
            self.selection.remove(lab)
            self.last_selected = None
        else:
            self.selection.add(lab)
            self.last_selected = lab
        self.update_hull()

    def toggle_args_vertices(self, on=True):
        if "args" in self.uniq_points:
            args_indices = {("args", i) for i in np.ndindex(self.uniq_points["args"].shape[0])}
            if on:
                self.selection = self.selection.union(args_indices)
            else:
                self.selection = self.selection - args_indices
        #logger.debug(self.selection)
        self.update_hull()

    def update_hull(self):
        selected_points = [self.uniq_points[filepath][index] for filepath, index in self.selection]
        # logger.debug(f"selected_points:{selected_points}")
        if len(selected_points) < 4:
            self.hull = None
        else:
            logger.debug("Calculating hull")
            if self.alpha is None or self.alpha == 0:
                logger.debug("creating convex hull")
                # the api for alphashape is a bit strange,
                # it returns a shapely polygon when alpha is 0
                # rather than a trimesh object which is returned for other values of alpha
                # so just calculate the convex hull with trimesh to ensure we get a consistent return value
                self.hull = PointCloud(selected_points).convex_hull
            else:
                logger.debug("creating alpha shape")
                self.hull = alphashape(np.array(selected_points), self.alpha)
                if len(self.hull.faces) == 0:
                    logger.debug("More uniq_points required for a complete hull with current alpha value")
                    self.hull = None

    def update_dist(self, filepath):
        if self.hull is None:
            logger.debug("No hull available to calculate distance")
            self.dist = {k: np.full(v.shape[0], np.inf) for k, v in self.uniq_points.items()}
            return

        logger.debug(f"updating distances from hull for {filepath}")
        # we don't update the whole array every time as it is too slow,
        # we just update the current file to support display of within
        # see https://github.com/mikedh/trimesh/issues/1116
        # todo keep an eye on this as the alphashape package is likely to change around this
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.hull.as_open3d))

        distances_array = scene.compute_signed_distance(o3d.core.Tensor.from_numpy(self.uniq_points[filepath].astype(dtype=np.float32))).numpy()
        logger.debug(distances_array.shape)
        self.dist[filepath] = distances_array.reshape(-1, 1)




