import logging
from typing import List, Set, Tuple, Optional

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
from ..figurebuilder import FigureMatplot
from ..options.update_and_verify import update_arg

from .loading import Points, LayoutMultiLoader, wait_for_result

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
            for i, layout in enumerate(layoutmultiloader.layouts):
                layouts[layouts_to_load[i]] = layout

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
        self.toggle_args_colours()
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
            "Include points supplied as arguments",
            "Include points supplied as arguments"
        )
        self.toolbar.ToggleTool(11, False)
        self.Bind(wx.EVT_TOOL, self.toggle_args_colours, id=11)
        if self.args.hull_vertices is None:
            self.toolbar.EnableTool(11, False)

        self.toolbar.AddTool(
            20,
            "Print",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "print_hull.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
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

    def pixel_to_lab(self, pixel, filepath: Optional[Path] = None) -> Tuple[str, int]:
        if filepath is None:
            filepath = self.images[self.ind].filepath
        return tuple(self.points.pixel_to_lab.loc[(str(filepath), pixel)].to_numpy())

    @wait_for_result
    def lab_to_pixels(
            self,
            lab: List[Tuple[int, int, int]] | Tuple[int, int, int],
            filepath: Optional[Path] = None
    ) -> List[Tuple[str, int]] | List[int]:
        logger.debug("Find pixels in selected colour bins and return pixel indices")
        if not isinstance(lab, list):
            lab = [lab]
        if filepath is None:
            logger.debug("Ensure only good keys")
            lab = self.points.filepath_lab_to_pixel.index.intersection(lab)  # keep only good keys
            return [tuple(x) for x in self.points.lab_to_pixel.loc[lab].to_numpy()]
        else:
            # need to handle cases where colour is in another image or args but not in the current image
            # this raises a keyerror in pandas
            logger.debug("Ensure only good keys")
            lab = self.points.filepath_lab_to_pixel.loc[str(filepath)].index.intersection(lab)  # keep only good keys
            logger.debug("Lookup keys")
            return self.points.filepath_lab_to_pixel.loc[str(filepath)].loc[lab].to_numpy().reshape(-1).tolist()

    def on_click_image(self, event):
        # Left click only
        if event.mouseevent.button == 1:
            x = int(np.around(event.mouseevent.xdata, decimals=0))
            y = int(np.around(event.mouseevent.ydata, decimals=0))
            logger.debug(f"x: {x}, y: {y}")
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
                lab = self.points.counts_all.iloc[lab_plot_index][[("lab", "L"), ("lab", "a"),("lab", "b")]].to_numpy()
            else:
                filepath = self.images[self.ind].filepath
                lab = self.points.counts_per_image.loc[str(filepath)].iloc[lab_plot_index][
                    [("lab", "L"), ("lab", "a"), ("lab", "b")]
                ].to_numpy()
            lab = tuple(lab)
            #  or points.counts_per_image
            self.alpha_selection.toggle_colour(lab)
            self.draw_image_figure()
            self.draw_hull()

    def toggle_args_colours(self, event=None):
        if self.args.hull_vertices is not None:
            show_args = self.toolbar.GetToolState(11)  # true if show args is selected in gui
            self.alpha_selection.toggle_args_colours(show_args)
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
        displayed = image.rgb.copy()

        if show_selected:
            selected_pixels = self.lab_to_pixels(list(self.alpha_selection.selection), image.filepath)
            logger.debug(f"Selected pixels: {len(selected_pixels)}")
            displayed.reshape(-1, 3)[selected_pixels, :] = (1, 1, 1)
            #last_selected_pixels = self.lab_to_pixels(self.alpha_selection.last_selected, image.filepath)
            #displayed.reshape(-1, 3)[last_selected_pixels, :] = (1, 0, 0)

        if show_within:
            self.alpha_selection.update_dist(image.filepath)
            lab_within = self.alpha_selection.get_within(image.filepath)
            if lab_within:
                logger.debug(f"Within colours: {len(lab_within)}")
                within_pixels = self.lab_to_pixels(lab_within, filepath=image.filepath)
                logger.debug(f"Within pixels: {len(within_pixels)}")
                displayed.reshape(-1, 3)[within_pixels, :] = (1, 1, 1)  # todo make these colours variables, either as arguments or some other method

        if self.image_art is None:
            self.image_art = self.image_ax.imshow(displayed, picker=True)
        else:
            self.image_art.set_data(displayed)

        self.image_cv.draw_idle()
        self.image_cv.flush_events()

    def draw_lab_figure(self, _=None):
        logger.debug("Draw lab figure")
        show_all = self.toolbar.GetToolState(12)
        image = self.images[self.ind]
        if show_all:
            lab = self.points.counts_all[[("lab", "L"), ("lab", "a"), ("lab", "b")]].to_numpy()
            rgb = self.points.counts_all[[("rgb", "r"), ("rgb", "g"), ("rgb", "b")]].to_numpy()
            counts = self.points.counts_all["count"].to_numpy()
        else:
            lab = self.points.counts_per_image.loc[str(image.filepath), [("lab", "L"), ("lab", "a"), ("lab", "b")]].to_numpy()
            rgb = self.points.counts_per_image.loc[str(image.filepath), [("rgb", "r"), ("rgb", "g"), ("rgb", "b")]].to_numpy()
            counts = self.points.counts_per_image.loc[str(image.filepath), "count"].to_numpy()

        log_counts = np.log(counts)
        scaled_log_counts = log_counts/max(log_counts)
        sizes = scaled_log_counts * 50
        # sizes[sizes <= 1] = 1  # ensure even low abundance points are still shown?

        # get the current view angles on lab plot to reload with after redraw
        elev, azim = self.lab_ax.elev, self.lab_ax.azim

        if self.lab_art is not None:
            self.lab_art.remove()
            self.lab_art = None
        self.lab_art = self.lab_ax.scatter(
            xs=lab[:, 1],
            ys=lab[:, 2],
            zs=lab[:, 0],
            s=sizes,
            color=rgb,
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

    @wait_for_result
    def wait_for_animation(self, _=None):

        if self.alpha_selection.hull is not None:
            if len(self.alpha_selection.hull.vertices) < 4:
                hull = None
            else:
                logger.debug("Calculating hull")
                if self.alpha_selection.alpha is None or self.alpha_selection.alpha == 0:
                    logger.debug("creating convex hull")
                    # the api for alphashape is a bit strange,
                    # it returns a shapely polygon when alpha is 0
                    # rather than a trimesh object which is returned for other values of alpha
                    # so just calculate the convex hull with trimesh to ensure we get a consistent return value
                    hull = PointCloud(self.alpha_selection.hull.vertices).convex_hull
                else:
                    logger.debug("creating alpha shape")
                    hull = alphashape(self.alpha_selection.hull.vertices, self.alpha_selection.alpha)
                    if len(hull.faces) == 0:
                        logger.debug("More points required for a complete hull with current alpha value")
                        hull = None
        else:
            hull = None

        logger.debug("Plotting animation of hull")
        fig = FigureMatplot("Hull", self.parent.figure_counter, self.args, cols=1)
        abl = self.points.counts_all[[("lab", "a"), ("lab", "b"), ("lab", "L")]].to_numpy()
        rgb = self.points.counts_all[[("rgb", "r"), ("rgb", "g"), ("rgb", "b")]].to_numpy()
        counts = self.points.counts_all["count"].to_numpy()
        log_counts = np.log(counts)
        scaled_log_counts = log_counts / max(log_counts)
        sizes = scaled_log_counts * 20
        labels = ("a*", "b*", "L*")
        fig.plot_scatter_3d(abl, labels, rgb, sizes, hull)
        fig.animate()

    def reset_selection(self, _=None):
        self.alpha_selection = AlphaSelection(
            self.points,
            set(),
            alpha=self.args.alpha,
            delta=self.args.delta
        )
        self.toolbar.ToggleTool(11, False)
        self.draw_image_figure()
        self.draw_hull()


class AlphaSelection:
    def __init__(
            self,
            points: Points,
            selection: Set[Tuple[int, int, int]],
            alpha: float,
            delta: float
    ):
        self.points = points

        self.selection = selection  # a set of lab tuples
        self.last_selected = None  # a single lab tuple

        self.alpha = alpha  # the alpha parameter for hull construction
        self.delta = delta  # the distance from the hull surface to consider a point within the hull

        self.hull: Optional[Trimesh] = None

    def update_alpha(self, alpha: float = None):
        if alpha is None:
            if len(self.selection) >= 4:
                logger.debug(f"optimising alpha")
                self.alpha = round(optimizealpha(list(self.selection)), ndigits=3)
                logger.info(f"optimised alpha: {self.alpha}")
            else:
                logger.debug(f"Insufficient uniq_points selected")
        else:
            self.alpha = alpha
        self.update_hull()

    def set_delta(self, delta: float):
        logger.debug(f"Set delta: {delta}")
        self.delta = delta

    def toggle_colour(self, lab: Tuple[int, int, int]):
        if lab in self.selection:
            self.selection.remove(lab)
            self.last_selected = None
        else:
            self.selection.add(lab)
            self.last_selected = lab
        self.update_hull()

    def toggle_args_colours(self, on=True):
        if "args" in self.points.pixel_to_lab.index:
            args_colours = set(map(tuple, self.points.pixel_to_lab.loc["args"].to_numpy()))
            if on:
                logger.debug(f"Include args points: {len(args_colours)}")
                self.selection = self.selection.union(args_colours)
            else:
                logger.debug(f"Remove args points: {len(args_colours)}")
                self.selection = self.selection - args_colours
        self.update_hull()

    def update_hull(self):
        # logger.debug(f"selected_points:{selected_points}")
        if len(self.selection) < 4:
            self.hull = None
        else:
            logger.debug("Calculating hull")
            if self.alpha is None or self.alpha == 0:
                logger.debug("creating convex hull")
                # the api for alphashape is a bit strange,
                # it returns a shapely polygon when alpha is 0
                # rather than a trimesh object which is returned for other values of alpha
                # so just calculate the convex hull with trimesh to ensure we get a consistent return value
                self.hull = PointCloud(list(self.selection)).convex_hull
            else:
                logger.debug("creating alpha shape")
                self.hull = alphashape(np.array(list(self.selection)), self.alpha)
                if len(self.hull.faces) == 0:
                    logger.debug("More uniq_points required for a complete hull with current alpha value")
                    self.hull = None

    def update_dist(self, filepath):
        if self.hull is None:
            logger.debug("No hull available to calculate distance")
            self.points.counts_per_image['distance'] = np.inf
            return

        logger.debug(f"updating distances from hull for {filepath}")
        # we don't update the whole array every time as it is slow,
        # we just update the current file to support display of within
        # todo keep an eye on this as the alphashape package is likely to change around this
        # see https://github.com/mikedh/trimesh/issues/1116
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.hull.as_open3d))
        colours = self.points.counts_per_image.loc[str(filepath)][[("lab", "L"), ("lab", "a"), ("lab", "b")]].to_numpy(dtype=np.float32)
        distances = scene.compute_signed_distance(
            o3d.core.Tensor.from_numpy(colours)
        ).numpy()
        self.points.counts_per_image.loc[str(filepath), 'distance'] = distances.reshape(-1, 1)

    def get_within(self, filepath):
        if self.hull is None:
            logger.debug("No hull available to calculate within")
            return list()
        all_colours = self.points.counts_per_image.loc[str(filepath)][[("lab", "L"), ("lab", "a"), ("lab", "b")]]
        logger.debug(f"Colours in file: {all_colours.shape[0]}")
        colours_within = self.points.counts_per_image.loc[str(filepath), "distance"] <= self.delta
        logger.debug(f"Colours within hull: {np.sum(colours_within)}")
        return list(map(tuple, all_colours[colours_within].to_numpy()))
