import argparse
import logging
from pathlib import Path

from skimage.io import imread
from skimage.util import img_as_float64 as img_as_float
# todo reconsider, 64 is actually faster with open3d so coerce to this
from skimage.color import rgb2lab
from skimage.transform import downscale_local_mean
from copy import deepcopy

from src.logging import ImageFilepathAdapter

from .options.custom_types import DebugEnum
from .figurebuilder import FigureBase, FigureMatplot, FigureNone

logger = logging.getLogger(__name__)


class ImageLoaded:

    def __init__(self, filepath: Path, args: argparse.Namespace):
        self.args = args
        self.filepath = filepath

        self.logger = ImageFilepathAdapter(logger, {'image_filepath': str(filepath)})

        self.logger.debug(f"Read image from file")
        self.rgb = img_as_float(imread(str(filepath)))
        self.logger.debug(f"Loaded RGB image data type: {self.rgb.dtype}")

        if self.rgb.shape[2] == 4:
            self.logger.info("Removing alpha channel")
            # slice off the alpha channel
            self.rgb = self.rgb[:, :, :3]
        elif self.rgb.shape[2] != 3:
            raise ValueError("This image does not appear to be RGB")

        self.figures = ImageFigureBuilder(filepath, args)

        rgb_fig = self.figures.new_figure("RGB image")
        rgb_fig.plot_image(self.rgb, "RGB image")
        rgb_fig.print()


        self.logger.debug(f"Convert to Lab")
        self.lab = rgb2lab(self.rgb)
        lab_fig = self.figures.new_figure("Lab channels")
        lab_fig.plot_image(self.lab[:, :, 0], "Lightness channel (L in Lab)", color_bar=True)
        lab_fig.plot_image(self.lab[:, :, 1], "Green-Red channel (a in Lab)", color_bar=True)
        lab_fig.plot_image(self.lab[:, :, 2], "Blue-Yellow channel (b in Lab)", color_bar=True)
        lab_fig.print()

        # downscale the image
        if self.args.downscale != 1:
            self.logger.debug(f"Downscale the RGB input image")
            self.rgb = downscale_local_mean(self.rgb, (self.args.downscale, self.args.downscale, 1))
            downscale_fig = self.figures.new_figure("Downscaled image")
            downscale_fig.plot_image(self.rgb, f"Downscale (factor={self.args.downscale})")
            downscale_fig.print()

            self.lab = downscale_local_mean(self.lab, (self.args.downscale, self.args.downscale, 1))
            lab_fig = self.figures.new_figure("Lab downscaled")
            lab_fig.plot_image(self.lab[:, :, 0], "Lightness channel (L in Lab)", color_bar=True)
            lab_fig.plot_image(self.lab[:, :, 1], "Green-Red channel (a in Lab)", color_bar=True)
            lab_fig.plot_image(self.lab[:, :, 2], "Blue-Yellow channel (b in Lab)", color_bar=True)
            lab_fig.print()

        self.logger.debug("Completed loading")

    def __hash__(self):
        return hash(self.filepath)

    def __lt__(self, other):
        return self.filepath < other.filepath

    def __le__(self, other):
        return self.filepath <= other.filepath

    def __gt__(self, other):
        return self.filepath > other.filepath

    def __ge__(self, other):
        return self.filepath <= other.filepath

    def __eq__(self, other):
        return self.filepath == other.filepath

    def __ne__(self, other):
        return self.filepath != other.filepath

    def copy(self):
        return deepcopy(self)


class ImageFigureBuilder:
    def __init__(self, image_filepath, args):
        self.counter = 0
        self.image_filepath = image_filepath
        self.args = args
        self.logger = ImageFilepathAdapter(logger, {"image_filepath": image_filepath})
        self.logger.debug("Creating figure builder object")

    def new_figure(self, name, cols=1, level="DEBUG") -> FigureBase:
        if DebugEnum[level] >= self.args.image_debug:
            self.counter += 1
            return FigureMatplot(name, self.counter, self.args, cols=cols, image_filepath=self.image_filepath)
        else:
            return FigureNone(name, self.counter, self.args, cols=cols, image_filepath=self.image_filepath)
