import argparse
import logging

from pathlib import Path
from .figurebuilder import FigureBuilder
from skimage.io import imread
from skimage.color import rgb2lab


logger = logging.getLogger(__name__)


class ImageLoaded:
    # todo consider gaussian blur on the input image when loaded
    def __init__(self, filepath: Path, args: argparse.Namespace):
        self.args = args
        self.filepath = filepath
        logger.debug(f"Load image as RGB: {self.filepath}")
        self.rgb = imread(str(filepath))
        if self.args.debug:
            fig = FigureBuilder(self.filepath, self.args, "Load image")
            fig.add_image(self.rgb, filepath.stem)
            fig.print()
        logger.debug(f"Convert RGB to Lab")
        self.lab = rgb2lab(self.rgb)
        if self.args.debug:
            fig = FigureBuilder(self.filepath, self.args, "Convert to Lab", nrows=3)
            fig.add_image(self.lab[:, :, 0], "Lightness channel (l in Lab)", color_bar=True)
            fig.add_image(self.lab[:, :, 1], "Green-Red channel (a in Lab)", color_bar=True)
            fig.add_image(self.lab[:, :, 2], "Blue-Yellow channel (b in Lab)", color_bar=True)
            fig.print()
