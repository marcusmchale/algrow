import logging

from typing import List, Optional
from ..image_loading import ImageLoaded

import wx
from pubsub import pub

import pandas as pd
import numpy as np
from pathlib import Path


from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigureCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar
)
from matplotlib.figure import Figure
from matplotlib import get_data_path  # we are recycling some matplotlib icons

from alphashape import alphashape, optimizealpha
from trimesh import PointCloud, proximity, Trimesh


from ..image_segmentation import Segmentor
from ..figurebuilder import FigureBuilder
from ..options import update_arg


logger = logging.getLogger(__name__)


class HullPanel(wx.Panel):

    def __init__(self, parent, segmentor: Segmentor):
        super().__init__(parent)

        self.segmentor = segmentor
        self.images = self.segmentor.images
        self.args = self.images[0].args

        # Prepare an index for navigating the list of images
        self.ind: int = 0

        # prepare an alpha selection object -
        # this uses the selected points to describe a hull and decide which points are within it
        self.alpha_selection = AlphaSelection(
            self.segmentor.lab,
            set(),
            alpha=self.args.alpha,
            delta=self.args.delta
        )
        self.alpha_text = None
        self.delta_text = None

        # add a close button to test remote closing
        self.Bind(wx.EVT_CLOSE, self.on_exit)

        # Prepare the figures and toolbars
        # we are using two figures rather than one due to a bug in imshow that prevents efficient animation
        # https://github.com/matplotlib/matplotlib/issues/18985
        # by keeping them separate, changing the view on one doesn't force a redraw of the other
        self.seg_fig = Figure()
        self.seg_fig.set_dpi(150)
        self.seg_ax = self.seg_fig.add_subplot(111)
        self.seg_fig.suptitle('Click to select segments', fontsize=16)
        self.seg_cv = FigureCanvas(self, -1, self.seg_fig)
        self.seg_art = None
        self.seg_nav_toolbar = NavigationToolbar(self.seg_cv)
        self.seg_nav_toolbar.Realize()
        seg_sizer = wx.BoxSizer(wx.VERTICAL)
        seg_sizer.Add(self.seg_nav_toolbar, 0, wx.ALIGN_CENTER)
        seg_sizer.Add(self.seg_cv, 1, wx.EXPAND)
        self.seg_nav_toolbar.update()
        self.click = self.seg_cv.mpl_connect('pick_event', self.on_click)

        self.lab_fig = Figure()
        self.lab_fig.set_dpi(100)
        self.lab_ax = self.lab_fig.add_subplot(111, projection='3d')
        self.lab_fig.suptitle("Inspect selection hull", fontsize=16)
        self.lab_ax.set_zlabel('L')
        self.lab_ax.set_xlabel('a')
        self.lab_ax.set_ylabel('b')
        self.lab_cv = FigureCanvas(self, -1, self.lab_fig)
        self.lab_art = None
        self.tri_art = None
        self.lab_nav_toolbar = NavigationToolbar(self.lab_cv)
        self.lab_nav_toolbar.Realize()

        lab_sizer = wx.BoxSizer(wx.VERTICAL)
        lab_sizer.Add(self.lab_nav_toolbar, 0, wx.ALIGN_CENTER)
        lab_sizer.Add(self.lab_cv, 1, wx.EXPAND)
        self.lab_nav_toolbar.update()

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        figure_sizer = wx.BoxSizer(wx.HORIZONTAL)
        figure_sizer.Add(seg_sizer, proportion=1, flag=wx.LEFT | wx.TOP | wx.EXPAND)
        figure_sizer.Add(lab_sizer, proportion=1, flag=wx.RIGHT | wx.TOP | wx.EXPAND)
        self.sizer.Add(figure_sizer, proportion=1, flag=wx.ALIGN_CENTER)
        self.SetSizer(self.sizer)
        #self.Fit()

        self.navigation_toolbar = None
        self.toolbar = None
        self.add_toolbar()

        # load the first image
        self.load_current_image()

    def on_exit(self, event):
        logger.debug("Close called")
        if self.alpha_selection.hull is None:
            raise ValueError("Calibration not complete - please start again and select more than 4 points")
        if self.args.debug:
            self.plot_hull(".")

        update_arg(self.args, 'alpha', self.alpha_selection.alpha)
        update_arg(self.args, "hull_vertices", list(map(tuple, np.round(
            self.alpha_selection.hull.vertices,
            decimals=1
        ).tolist())))
        update_arg(self.args, 'delta', self.alpha_selection.delta)

        pub.sendMessage("enable_btns")

        self.Destroy()
        #event.Skip()

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
            'Highlight selected segments'
        )
        self.toolbar.ToggleTool(7, True)
        self.Bind(wx.EVT_TOOL, self.draw_segments_figure, id=7)
        self.toolbar.AddCheckTool(
            8,
            "within",
            wx.Image(str(Path(Path(__file__).parent, "bmp", "within.png")), wx.BITMAP_TYPE_PNG).ConvertToBitmap(),
            wx.NullBitmap,
            "Highlight within",
            'Highlight segments within alpha shape'
        )
        self.toolbar.ToggleTool(8, True)
        self.Bind(wx.EVT_TOOL, self.draw_segments_figure, id=8)

        # add a close button
        self.toolbar.AddSeparator()
        self.close_btn = wx.Button(self.toolbar, 5, "Save and close")
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
        self.seg_ax.set_title(str(filepath))
        self.draw_segments_figure()
        self.draw_lab_figure()

    def optimise_alpha(self, _):
        self.alpha_selection.update_alpha()
        self.draw_segments_figure()
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
        self.draw_segments_figure()
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
        self.draw_segments_figure()

    def on_click(self, event):
        x = event.mouseevent.xdata.astype(int)
        y = event.mouseevent.ydata.astype(int)
        image = self.images[self.ind]
        sid = self.segmentor.image_to_segments[image].mask[y, x]
        logger.debug(f'file: {image.filepath}, segment:  {sid}, x: {x}, y: {y}')
        if sid != 0:
            self.alpha_selection.toggle_segment(image.filepath, sid)
            self.draw_segments_figure()
            self.draw_hull()

    def draw_segments_figure(self, _=None):
        logger.debug("Draw segments with highlighting")
        show_selected = self.toolbar.GetToolState(7)  # true if highlight selection is selected in gui
        logger.debug(f"show selected: {show_selected}")
        show_within = self.toolbar.GetToolState(8)  # true if highlight within is selected in gui
        logger.debug(f"show within: {show_within}")

        image = self.images[self.ind]
        filepath = image.filepath
        current_file_segments = self.segmentor.image_to_segments[image]
        displayed = current_file_segments.boundaries.copy()

        if show_selected or show_within:
            segments_mask = self.segmentor.image_to_segments[image].mask
            if show_within:
                self.alpha_selection.update_dist(filepath)
                within = self.alpha_selection.dist[(self.alpha_selection.dist >= -self.alpha_selection.delta).values].index
                within = [j for i, j in within if i == filepath]
                logger.debug(f'within: {within}')
                displayed[np.isin(segments_mask, within)] = (0, 1, 0)
            if show_selected:
                selected = [j for i, j in self.alpha_selection.selection if i == filepath]
                logger.debug(f'selected: {selected}')
                displayed[np.isin(segments_mask, selected)] = (0, 0, 1)

        if self.seg_art is None:
            self.seg_art = self.seg_ax.imshow(displayed, picker=True)
        else:
            self.seg_art.set_data(displayed)

        self.seg_cv.draw_idle()
        self.seg_cv.flush_events()

    def draw_lab_figure(self, _=None):
        current_file_segments = self.segmentor.image_to_segments[self.images[self.ind]]
        elev, azim = self.lab_ax.elev, self.lab_ax.azim  # get the current view angles on lab plot to reload with

        if self.lab_art is not None:
            self.lab_art.remove()
            self.lab_art = None
        self.lab_art = self.lab_ax.scatter(
            xs=current_file_segments.lab['a'],
            ys=current_file_segments.lab['b'],
            zs=current_file_segments.lab['L'],
            s=10,
            c=current_file_segments.rgb,
            lw=0
        )
        self.lab_ax.view_init(elev=elev, azim=azim)
        self.draw_hull()

    def draw_hull(self):
        if self.tri_art is not None:
            logger.debug("remove existing alpha hull from plot")
            self.tri_art.remove()
            self.tri_art = None

        if self.alpha_selection.hull is not None:
            logger.debug("draw alpha hull on plot")
            self.tri_art = self.lab_ax.plot_trisurf(
                *zip(*self.alpha_selection.hull.vertices[:, [1, 2, 0]]),
                triangles=self.alpha_selection.hull.faces[:, [1, 2, 0]],
                color=(0, 1, 0, 0.5)
            )

        self.lab_cv.draw_idle()
        self.lab_cv.flush_events()

    def plot_hull(self, path):
        if self.alpha_selection.hull is None:
            logger.warning("Configuration is incomplete, will not attempt to print summary figure")
            return
        logger.debug("Plotting hull in lab colourspace")
        fig = FigureBuilder(path, self.args, 'Alpha hull')
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            xs=self.segmentor.lab['a'],
            ys=self.segmentor.lab['b'],
            zs=self.segmentor.lab['L'],
            s=10,
            c=self.segmentor.rgb,
            lw=0
        )

        ax.plot_trisurf(
            *zip(
                *self.alpha_selection.hull.vertices[:, [1, 2, 0]]
            ),
            triangles=self.alpha_selection.hull.faces[:, [1, 2, 0]],
            color=(0, 1, 0, 0.5)
        )
        fig.animate()
        fig.print()


class AlphaSelection:
    def __init__(self, points: pd.DataFrame, selection: set[tuple[Path, int]], alpha: float, delta: float):
        self.points = points  # a pandas dataframe with filename and segment ID as index for points in Lab colourspace
        self.selection = selection  # a set of indices for the points dataframe
        self.alpha = alpha  # the alpha parameter for hull construction
        self.delta = delta  # the distance from the hull surface to consider a point within the hull
        self.dist: pd.DataFrame = pd.DataFrame(np.full(points.shape[0], -np.inf), index=points.index)
        self.hull: Optional[Trimesh] = None

    def update_alpha(self, alpha: float = None):
        if alpha is None:
            if len(self.selection) >= 4:
                logger.debug(f"optimising alpha")
                self.alpha = round(optimizealpha(self.points.loc[list(self.selection)].values), ndigits=3)
                logger.info(f"optimised alpha: {self.alpha}")
            else:
                logger.debug(f"Insufficient points selected")
        else:
            self.alpha = alpha
        self.update_hull()

    def set_delta(self, delta: float):
        logger.debug(f"Set delta: {delta}")
        self.delta = delta

    def toggle_segment(self, filepath, sid):
        if (filepath, sid) in self.selection:
            self.selection.remove((filepath, sid))
        else:
            self.selection.add((filepath, sid))
        self.update_hull()

    def update_hull(self):
        selected_points = self.points.loc[list(self.selection)].values
        #logger.debug(f"selected_points:{selected_points}")
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
                    logger.debug("More points required for a complete hull with current alpha value")
                    self.hull = None

    def update_dist(self, filepath):
        if self.hull is None:
            logger.debug("No hull available to calculate distance")
            return

        logger.debug(f"updating distances from hull for {filepath}")
        # we don't update the whole array every time as it is too slow,
        # we just update the current file to support display of within
        self.dist.loc[filepath] = proximity.signed_distance(self.hull, self.points.loc[filepath]).reshape(-1, 1)
