import io
import logging
import argparse
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path

from skimage.color import lab2rgb
from matplotlib import gridspec,  animation, patheffects
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from scipy.cluster import hierarchy
from skimage.morphology import binary_erosion
from matplotlib.pyplot import imread


logger = logging.getLogger(__name__)


class FigureAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if self.extra['image_filepath'] is None:
            return f"[Figure: {self.extra['number']}, {self.extra['name']}] {msg}", kwargs
        else:
            return (
                f"[{self.extra['image_filepath']}] [Figure:{self.extra['number']}, {self.extra['name']}] {msg}",
                kwargs
            )


class FigureBase(ABC):

    def __init__(self, name: str, number: int, args: argparse.Namespace, cols=1, image_filepath=None):
        self.name = name
        self.number = number
        self.args = args
        self.cols = cols
        self.image_filepath = image_filepath

        self.logger = FigureAdapter(logger, {
            'image_filepath': image_filepath,
            'name': name,
            'number': number
        })

    @abstractmethod
    def plot_image(
            self,
            img: np.array,
            label: str = None,
            color_bar=False,
            picker=None,
            show_axes=True
    ):
        raise NotImplementedError

    @abstractmethod
    def add_outline(self, mask: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def add_label(self, text, coords, color, size):
        raise NotImplementedError

    @abstractmethod
    def add_circle(self, coords, radius, linewidth=5):
        raise NotImplementedError

    @abstractmethod
    def plot_dendrogram(self, dendrogram: np.ndarray, cut_height: float, label: str = None):
        raise NotImplementedError

    @abstractmethod
    def plot_colours(self, colours, npix=10):
        raise NotImplementedError

    @abstractmethod
    def plot_text(self, text):
        raise NotImplementedError

    @abstractmethod
    def plot_scatter_3d(self, points: np.ndarray, axis_labels: tuple[str, str, str], rgb: np.ndarray, hull=None):
        raise NotImplementedError

    @abstractmethod
    def animate(self):
        raise NotImplementedError

    @abstractmethod
    def print(self, large=False):
        raise NotImplementedError

    @abstractmethod
    def as_array(self, width, height):
        raise NotImplementedError


class FigureMatplot(FigureBase):

    def __init__(self, name: str, number: int, args: argparse.Namespace, cols=1, image_filepath=None):
        super().__init__(name, number, args, cols=cols, image_filepath=image_filepath)
        self.logger.debug(f"Matplotlib figure")

        self._shape = np.array([0, cols])  # rows and columns (when we add an image we add a row if needed)
        self._current_row_index = 0  # Note, matplotlib uses 1 based indexing - we are doing the same (0 is Null)
        self._current_column_index = 0
        self._fig = Figure()  # the matplotlib figure class
        self._plots = list()  # list of the axes to relocate
        self._colorbars = list()  # list of the colorbars to redraw
        self._axes_images = list()  # list of the axes_images to get/set array values

        self._add_row()

    @property
    def _current_ax_index(self):
        index = ((self._shape[0] - 1) * self._shape[1]) + self._current_column_index
        return index

    @property
    def _current_axis(self):
        return self._plots[self._current_ax_index - 1]

    @property
    def _current_axes_image(self):
        return self._axes_images[self._current_ax_index - 1]

    def _add_row(self):
        self._shape[0] += 1
        self._current_row_index += 1
        self._current_column_index = 0
        if self._shape[0] > 1:
            self._fig.canvas.draw()
            gs = gridspec.GridSpec(self._shape[0], self._shape[1], figure=self._fig, hspace=0.6)
            for i, ax in enumerate(self._plots):
                ax.set_position(gs[i].get_position(self._fig))
                ax.set_subplotspec(gs[i])
                if self._colorbars[i] is not None:
                    # remove and redraw the colorbar, can't figure out how to just move it
                    self._colorbars[i].remove()
                    self._colorbars[i] = self._fig.colorbar(self._axes_images[i], ax=ax)

    def _add_plot(self, projection=None) -> Axes:
        if self._current_column_index == self._shape[1]:
            self._add_row()
        self._current_column_index += 1
        ax = self._fig.add_subplot(  # NOTE: matplotlib uses 1 based indexes
            self._shape[0],
            self._shape[1],
            self._current_ax_index,
            projection=projection
        )
        self._plots.append(ax)
        return ax

    def plot_image(
            self,
            img: np.array,
            label: str = None,
            color_bar=False,
            picker=None,
            show_axes=True
    ):
        self.logger.debug("Add image")
        ax: Axes = self._add_plot()
        if not show_axes:
            ax.axis('off')
            ax.margins(0, 0)
        if label:
            ax.set_title(label, loc="left")
        axes_image = ax.imshow(img, picker=picker)
        if color_bar:
            cb_ax = self._fig.colorbar(axes_image, ax=ax)
            self._colorbars.append(cb_ax)
        else:
            self._colorbars.append(None)
        self._axes_images.append(axes_image)

    def plot_text(self, text):
        self.logger.debug("Add text as plot")
        ax: Axes = self._add_plot()
        ax.text(0.5, 0.5, text, ha="center", va="center")
        self._colorbars.append(None)

    def add_outline(self, mask: np.ndarray):
        mask = mask.copy()  # we mutate this here, I don't think it is used again but maybe one day so just in case
        image = self._current_axes_image.get_array()

        # better without dilation as it highlights small isolated areas
        #contour_dilate = binary_dilation(mask, footprint=np.full((5, 5), 1))
        #contour_dilate[mask] = False
        #image[contour_dilate] = (1, 1, 1)
        # erode is better as it only shows the selected areas.
        contour_erode = binary_erosion(mask, footprint=np.full((5, 5), 1))
        mask[contour_erode] = False
        image[mask] = (1, 0, 1)

        self._current_axes_image.set_array(image)

    def plot_dendrogram(self, dendrogram: np.ndarray, cut_height: float, label: str = None):
        self.logger.debug("Draw dendrogram")
        ax: Axes = self._add_plot()
        hierarchy.dendrogram(dendrogram, ax=ax, no_labels=True, count_sort=True)
        # todo consider giving labels corresponding to circle ID and plot these on the image overlay
        ax.axhline(y=cut_height, c='k')
        if label:
            ax.set_title(label)
        self._colorbars.append(None)

    def add_label(self, text, coords, color, size):
        self._current_axis.annotate(
            text,
            coords,
            color=color,
            size=size,
            ha='center',
            va='center',
            path_effects=[patheffects.withStroke(linewidth=size/2, foreground='white')]
        )

    def add_circle(self, coords, radius, linewidth=5):
        self._current_axis.add_patch(
            Circle(
                coords,
                radius,
                color="white",
                fill=False,
                linewidth=linewidth
            )
        )

    def plot_colours(self, colours, npix=10):
        self.logger.debug('Plot colours')
        colour_plot = np.empty((0, 0, 3), int)
        for L, a, b in colours:
            colour_plot = np.append(colour_plot, np.tile([L, a, b], np.square(npix)).astype(float))
        colour_plot = lab2rgb(colour_plot.reshape(npix * len(colours), npix, 3))
        self.plot_image(colour_plot)
        for i, c in enumerate(colours):
            self.add_label(str(c), (npix//2, (npix*i) + npix//2), 1-lab2rgb(c), 12)
        self._current_axis.get_xaxis().set_visible(False)
        self._current_axis.get_yaxis().set_visible(False)

    def plot_scatter_3d(self, points: np.ndarray, axis_labels: tuple[str, str, str], rgb: np.ndarray, sizes=10, hull=None):
        # todo starting to refactor to support other colourspaces
        ax = self._add_plot(projection='3d')
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        ax.scatter(xs=points[:, 0], ys=points[:, 1], zs=points[:, 2], s=sizes, c=rgb, lw=0)
        # rearranging so that L is z axis, i.e. last
        if hull is not None:
            self._current_axis.plot_trisurf(
                *zip(*hull.vertices[:, [1, 2, 0]]),
                triangles=hull.faces[:, [1, 2, 0]],
                color=(0, 1, 0, 0.5)
            )

    def animate(self):

        def rotate(ii, ax):
            ax.view_init(azim=ii[0], elev=ii[1])
            return [ax]

        self.logger.debug("Making animation")
        out_path = self._get_filepath(suffix='gif')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rot_animation = animation.FuncAnimation(
            self._fig,
            rotate,
            fargs=[self._current_axis],
            frames=np.array([np.arange(0, 720, 20), np.arange(0, 180, 5)]).T,
            interval=500
        )
        try:
            rot_animation.save(str(out_path), dpi=80, writer='imagemagick')
        except OSError:
            self.logger.debug('failed to generate animation of 3d plot - requires imagemagick')
            # todo adapt this to work on windows systems using Pillow or similar
            pass
        self._current_axis.view_init(elev=45, azim=45)

    def _get_filepath(self, suffix='png'):
        if self.image_filepath is None:
            return Path(self.args.out_dir, "Figures", Path(".".join(["_".join([str(self.number), self.name]), suffix])))
        else:
            return Path(self.args.out_dir, "Figures", "ImageAnalysis", Path(
                ".".join(["_".join([Path(self.image_filepath).stem, str(self.number), self.name]), suffix])
            ))

    def print(self, large=False, margins=True):
        self.logger.debug(f"Save figure")
        if large:
            self._fig.set_figheight(6 * self._shape[0])
            self._fig.set_figwidth(8 * self._shape[1])
        else:
            self._fig.set_figheight(3 * self._shape[0])
            self._fig.set_figwidth(4 * self._shape[1])
        out_path = self._get_filepath()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if margins:
            self._fig.savefig(str(out_path), dpi=300)
        else:
            self._fig.savefig(str(out_path), dpi=300, bbox_inches='tight', pad_inches=0)

    def as_array(self, width, height):
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._fig.set_size_inches(width/100, height/100)
        self._fig.tight_layout(pad=0)
        buf = io.BytesIO()
        #self._fig.savefig(buf, dpi=100, bbox_inches='tight', pad_inches=0)
        self._fig.savefig(buf, dpi=100)
        buf.seek(0)
        img = imread(buf)
        buf.close()
        # remove the alpha channel
        img = img[:, :, :3]
        #self.print(margins=False)
        return img


class FigureNone(FigureBase):

    def __init__(self, name: str, number: int, args: argparse.Namespace, cols=1, image_filepath=None):
        super().__init__(name, number, args, cols=cols, image_filepath=image_filepath)
        self.logger.debug("Figure will not be generated")

    def plot_image(
            self,
            img: np.array,
            label: str = None,
            color_bar=False,
            picker=None,
            show_axes=True
    ):
        pass

    def plot_text(self, text):
        pass

    def add_label(self, text, coords, color, size):
        pass

    def add_outline(self, mask: np.ndarray):
        pass

    def add_circle(self, coords, radius, linewidth=5):
        pass

    def plot_dendrogram(self, dendrogram: np.ndarray, cut_height: float, label: str = None):
        pass

    def plot_colours(self, colours, npix=10):
        pass

    def plot_scatter_3d(self, points: np.ndarray, axis_labels: tuple[str, str, str], rgb: np.ndarray, hull=None):
        pass

    def animate(self):
        pass

    def print(self, large=False):
        pass

    def as_array(self, width, height):
        pass