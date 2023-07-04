import logging
import numpy as np
from pathlib import Path
from skimage.color import lab2rgb
from matplotlib import pyplot as plt, gridspec, colors
from matplotlib import animation
from .logging import CustomAdapter

logger = logging.getLogger(__name__)


class FigureBuilder:
    counter = 0

    def __init__(self, img_path, args, step_name, nrows=1, ncols=1, force=None):
        # todo refactor force to something equivalent to loglevels
        # i.e. have different levels of debug image outputs
        self.img_path = img_path
        self.args = args
        self.logger = CustomAdapter(logger, {'image_filepath': img_path})

        self.plot = args.debug in ['plot', 'both'] or force in ['plot', 'both']
        self.save = args.debug in ['save', 'both'] or force in ['save', 'both']
        if args.processes > 1:
            if self.plot in ["plot", "both"]:
                self.logger.warning("Cannot use interactive plotting in multithreaded mode")
                self.plot = False

        self.step_name = step_name
        self.nrows = nrows
        self.ncols = ncols
        self.force = force
        self.fig = plt.figure()

        self.out_dir = args.out_dir
        self.row_counter = 1
        self.col_counter = 0
        FigureBuilder.counter += 1
        # need to improve management of concurrent processes for counters to work properly with multiprocessing
        # todo Consider disabling debugging except for -q flag (overlays) if multiprocessing

    @property
    def axes(self):
        return self.fig.axes

    @property
    def current_axis(self):
        return self.axes[self.current_ax_index]

    @property
    def current_ax_index(self):
        return (self.row_counter - 1) * self.ncols + self.col_counter - 1

    def add_subplot(self, projection = None):
        self.col_counter += 1
        if self.col_counter > self.ncols:
            self.col_counter = 1
            self.row_counter += 1
        return self.fig.add_subplot(self.nrows, self.ncols, self.current_ax_index + 1, projection=projection)

    def add_subplot_row(self):
        self.logger.debug("Add another row of subplots to existing figure")
        self.nrows += 1
        self.fig.canvas.draw()
        gs = gridspec.GridSpec(self.nrows, self.ncols, figure=self.fig)
        for i, ax in enumerate(self.axes):
            ax.set_position(gs[i].get_position(self.fig))
            ax.set_subplotspec(gs[i])

    def add_image(self, img, label:str = None, color_bar=False, diverging=False, midpoint=None, picker=None):
        self.logger.debug("Add image to figure")
        axis = self.add_subplot()
        if label:
            axis.set_title(label, loc="left")
        if diverging and midpoint is not None:
            pos = axis.imshow(img, cmap='RdBu_r', norm=colors.TwoSlopeNorm(vcenter=midpoint), picker=picker)
        elif diverging:
            pos = axis.imshow(img, cmap='RdBu_r', picker=picker)
        else:
            pos = axis.imshow(img, picker=picker)
        if color_bar:
            self.fig.colorbar(pos, ax=axis)
            # todo trying to add the midpoint to the axis ticks - failing for some reason...
            #cbar = self.segment_fig.colorbar(pos, ax=axis)
            #if midpoint is not None:
            #    ticks = cbar.get_ticks()
            #    if midpoint not in ticks:
            #        ticks = np.insert(ticks, np.searchsorted(ticks, midpoint), midpoint)
            #    cbar.set_ticks(ticks)
        return pos

    def plot_colours(self, target_colours, npix=10):
        self.logger.debug('Prepare colours plot')
        colour_plot = np.empty((0, 0, 3), int)
        for l, a, b in target_colours:
            colour_plot = np.append(colour_plot, np.tile([l, a, b], np.square(npix)).astype(float))
        colour_plot = lab2rgb(colour_plot.reshape(npix * len(target_colours), npix, 3))
        self.add_image(colour_plot)
        self.current_axis.set_yticks(
            np.arange(len(target_colours) * npix, step=npix) + npix / 2, labels=target_colours
        )
        self.current_axis.get_xaxis().set_visible(False)

    def get_out_path(self, suffix='.png'):
        out_path = Path(
            self.out_dir,
            "debug",
            Path(self.img_path).stem,
            Path(" - ".join([str(FigureBuilder.counter), self.step_name])).with_suffix(suffix)
        )
        return out_path

    def animate(self):
        if not self.args.animations & self.save:
            return

        def rotate(ii, ax):
            ax.view_init(azim=ii[0], elev=ii[1])
            return [ax]

        self.logger.debug("Making animation")
        out_path = self.get_out_path(suffix='.gif')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rot_animation = animation.FuncAnimation(
            self.fig,
            rotate,
            fargs=[self.current_axis],
            frames=np.array([np.arange(0, 720, 20), np.arange(0, 180, 5)]).T,
            interval=500
        )
        try:
            rot_animation.save(str(out_path), dpi=80, writer='imagemagick')
        except OSError:
            self.logger.debug('failed to generate animation of 3d plot - requires imagemagick')
            # todo adapt this to work on windows systems using Pillow or similar
            pass
        self.current_axis.view_init(elev=45, azim=45)

    def print(self, large = False):
        if self.save:
            if large:
                self.fig.set_figwidth(16 * self.ncols)
                self.fig.set_figheight(12 * self.nrows)
            else:
                self.fig.set_figwidth(8 * self.ncols)
                self.fig.set_figheight(6 * self.nrows)
            out_path = self.get_out_path()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(str(out_path), dpi=300)
            self.logger.debug(f"Save figure: {self.step_name, self.img_path}")
        if self.plot:
            self.fig.set_figwidth(4 * self.ncols)
            self.fig.set_figheight(3 * self.nrows)
            self.fig.set_dpi(100)
            self.logger.debug(f"Show figure: {self.step_name, self.img_path}")
            plt.show()
            ## todo, might prefer to use segment_fig.show but that requires a managed event loop
            # or all the figures are rendered at the end.
            # This is only issue if we are building one figure then move the another then go back to finish the first.
            # So far I am only building figures successively, and that works fine as is.
        plt.close()
