import logging
import numpy as np
from pathlib import Path
from skimage.color import lab2rgb
from matplotlib import pyplot as plt, gridspec, use, colors
from .options import options


logger = logging.getLogger(__name__)
args = options().parse_args()


class FigureBuilder:
    if args.processes > 1:
        if args.debug in ["plot", "both"]:
            logger.warning("Cannot use interactive plotting in multithreaded mode")
        use('Agg')
    counter = 0

    def __init__(self, img_path, step_name, nrows = None, ncols = None, force = None):
        self.img_path = img_path
        self.step_name = step_name
        self.nrows = nrows
        self.ncols = ncols
        self.force = force
        self.fig, self.ax = plt.subplots(nrows if nrows else 1, ncols if ncols else 1, num=self.step_name, squeeze=False, layout="constrained")
        self.plot = args.debug in ['plot', 'both'] or self.force in ['plot', 'both']
        self.save = args.debug in ['save', 'both'] or self.force in ['save', 'both']
        self.out_dir = args.out_dir
        self.row_counter = 0
        self.col_counter = 0
        FigureBuilder.counter += 1 # need to improve management of concurrent processes for this to work with multiprocessing
        # todo Consider disabling debugging except for -q flag (overlays) if multiprocessing
        # currently just not using this counter when multiprocessing, but the order of debug images is not that clear.


    def print(self):
        if self.save:
            self.fig.set_figwidth(8 * (self.ncols if self.ncols else 1))
            self.fig.set_figheight(6 * (self.nrows if self.nrows else 1))
            if args.processes > 1:
                out_path = Path(
                    self.out_dir,
                    "debug",
                    self.step_name,
                    Path(Path(self.img_path).stem).with_suffix('.png')
                )
            else:
                out_path = Path(
                    self.out_dir,
                    "debug",
                    Path(self.img_path).stem,
                    Path(" - ".join([str(FigureBuilder.counter), self.step_name])).with_suffix('.png')
                )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(str(out_path), dpi=300)
            logger.debug(f"Save figure: {self.step_name, self.img_path}")
        if self.plot:
            self.fig.set_figwidth(4 * (self.ncols if self.ncols else 1))
            self.fig.set_figheight(3 * (self.nrows if self.nrows else 1))
            self.fig.set_dpi(100)
            logger.debug(f"Show figure: {self.step_name, self.img_path}")
            plt.show()
            ## todo, might prefer to use fig.show but that requires a managed event loop
            # or all the figures are rendered at the end.
            # This is only issue if we are building one figure then move the another then go back to finish the first.
            # So far I am only building figures successively, and that works fine as is.

    def get_current_subplot(self):
        return self.ax[self.row_counter, self.col_counter]

    def finish_subplot(self):
        ncols = 1 if self.ncols is None else self.ncols
        if self.col_counter == ncols - 1:
            self.col_counter = 0
            self.row_counter += 1
        else:
            self.col_counter += 1


    def add_image(self, img, label:str = None, color_bar=False, diverging=False, midpoint=None):
        logger.debug("Add image to debug figure")
        axis = self.get_current_subplot()
        if label:
            axis.set_title(label, loc="left")
        if diverging and midpoint is not None:
            pos = axis.imshow(img, cmap='RdBu_r', norm=colors.TwoSlopeNorm(vcenter=midpoint))
        elif diverging:
            pos = axis.imshow(img, cmap='RdBu_r')
        else:
            pos = axis.imshow(img)
        if color_bar:
            self.fig.colorbar(pos, ax=axis)
            # todo trying to add the middle point to the axis ticks - failing for some reason...
            #cbar = self.fig.colorbar(pos, ax=axis)
            #if midpoint is not None:
            #    ticks = cbar.get_ticks()
            #    if midpoint not in ticks:
            #        ticks = np.insert(ticks, np.searchsorted(ticks, midpoint), midpoint)
            #    cbar.set_ticks(ticks)
        self.finish_subplot()

    def add_subplot_row(self):
        logger.debug("Add subplot row to existing figure")
        ncols = 1 if self.ncols is None else self.ncols
        if self.nrows is None:
            self.nrows = 2
        else:
            self.nrows += 1
        self.fig.canvas.draw()
        gs = gridspec.GridSpec(self.nrows, ncols, figure=self.fig)
        for i, ax in np.ndenumerate(self.ax):
            ax.set_position(gs[i].get_position(self.fig))
            ax.set_subplotspec(gs[i])
        for n in range(ncols):
            self.fig.add_subplot(gs[self.nrows-1, n])
        self.ax = np.reshape(self.fig.axes, (self.nrows, ncols))

    def plot_colours(self, target_colours, npix = 10):
        logger.debug('Output target colours plot')
        colour_plot = np.empty((0, 0, 3), int)
        for l,a,b in target_colours:
            colour_plot = np.append(colour_plot, np.tile([l, a, b], np.square(npix)).astype(float))
        colour_plot = lab2rgb(colour_plot.reshape(npix * len(target_colours), npix, 3))
        self.get_current_subplot().set_yticks(
            np.arange(len(target_colours) * npix, step=npix) + npix / 2, labels=target_colours
        )
        self.get_current_subplot().get_xaxis().set_visible(False)
        self.add_image(colour_plot)
