import logging
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt, gridspec
from .options import options

logger = logging.getLogger(__name__)
args = options().parse_args()


class FigureBuilder:
    counter = 0
    def __init__(self, img_path, step_name, nrows = None, ncols = None, force = None):
        self.img_path = img_path
        self.step_name = step_name
        self.nrows = nrows
        self.ncols = ncols
        self.force = force

        self.fig, self.ax = plt.subplots(nrows if nrows else 1, ncols if ncols else 1, num=self.step_name, squeeze=False)

        self.plot = args.debug in ['plot', 'both'] or self.force in ['plot', 'both']
        self.save = args.debug in ['save', 'both'] or self.force in ['save', 'both']
        self.out_dir = args.out_dir

        self.row_counter = 0
        self.col_counter = 0

        FigureBuilder.counter += 1

    def print(self):

        # todo :the above cause the window to resize dynamically this is annoying, find a fix
        self.fig.tight_layout()
        if self.save:
            self.fig.set_figwidth(8 * (self.ncols if self.ncols else 1))
            self.fig.set_figheight(6 * (self.nrows if self.nrows else 1))
            out_path = Path(self.out_dir, "debug", " - ".join([str(FigureBuilder.counter), self.step_name]), Path(Path(self.img_path).stem).with_suffix('.png'))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(str(out_path), dpi=300)
            logger.debug(f"Save figure: {self.step_name, self.img_path}")
        if self.plot:
            self.fig.set_figwidth(4 * (self.ncols if self.ncols else 1))
            self.fig.set_figheight(3 * (self.nrows if self.nrows else 1))
            self.fig.set_dpi(100)
            logger.debug(f"Show figure: {self.step_name, self.img_path}")
            plt.show() ## todo, here would be good to use fig.show but that requires an event loop or it all just pops up at the end
            # this is only issue if we are building one figure then move the another then go back to finish the first
            # i have worked around this so far but it is a big limitation..

    @staticmethod
    def none_or_one(arg):
        return arg is None or arg == 1

    def get_current_subplot(self):
        return self.ax[self.row_counter, self.col_counter]

    def finish_subplot(self):
        ncols = 1 if self.ncols is None else self.ncols
        if self.col_counter == ncols - 1:
            self.col_counter = 0
            self.row_counter += 1
        else:
            self.col_counter += 1


    def add_image(self, img, label:str = None, color_bar=False):
        logger.debug("Add image to debug figure")
        axis = self.get_current_subplot()
        if label:
            axis.set_title(label, loc="left")
        pos = axis.imshow(img)
        if color_bar:
            plt.colorbar(pos, ax=axis)
        self.finish_subplot()

    def add_subplot_row(self):
        logger.debug("Add subplot row to existing figure")
        ncols = 1 if self.ncols is None else self.ncols
        if self.nrows is None:
            self.nrows = 2
        else:
            self.nrows += 1
        gs = gridspec.GridSpec(self.nrows, ncols)
        for i, ax in np.ndenumerate(self.ax):
            ax.set_position(gs[i].get_position(self.fig))
            ax.set_subplotspec(gs[i])
        for n in range(ncols):
            self.fig.add_subplot(gs[self.nrows-1, n])
        self.ax = np.reshape(self.fig.axes, (self.nrows, ncols))








