import argparse
import logging

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from skimage.io import imread
from skimage.color import rgb2lab
from skimage.filters import gaussian
from copy import deepcopy
from typing import List


from .options.custom_types import DebugEnum
from .figurebuilder import FigureBase, FigureMatplot, FigureNone

logger = logging.getLogger(__name__)


class ImageFilepathAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['image_filepath'], msg), kwargs


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


class ImageLoaded:

    def __init__(self, filepath: Path, args: argparse.Namespace):
        self.args = args
        self.filepath = filepath

        self.logger = ImageFilepathAdapter(logger, {'image_filepath': str(filepath)})

        self.logger.debug(f"Read image from file")
        self.rgb = imread(str(filepath))

        self.figures = ImageFigureBuilder(filepath, args)

        rgb_fig = self.figures.new_figure("RGB image")
        rgb_fig.plot_image(self.rgb, "RGB image")
        rgb_fig.print()

        # blur the rgb image to remove noise
        # todo consider alternatblurring just the L channel in lab
        self.logger.debug(f"Blur the RGB input image")
        self.rgb = gaussian(self.rgb, self.args.blur, channel_axis=-1)
        blur_fig = self.figures.new_figure("Gaussian blur RGB image")
        blur_fig.plot_image(self.rgb, f"Blur (sigma={self.args.blur})")
        blur_fig.print()

        self.logger.debug(f"Convert to Lab")
        self.lab = rgb2lab(self.rgb)
        lab_fig = self.figures.new_figure("Lab channels")
        lab_fig.plot_image(self.lab[:, :, 0], "Lightness channel (L in Lab)", color_bar=True)
        lab_fig.plot_image(self.lab[:, :, 1], "Green-Red channel (a in Lab)", color_bar=True)
        lab_fig.plot_image(self.lab[:, :, 2], "Blue-Yellow channel (b in Lab)", color_bar=True)
        lab_fig.print()

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


class ImageLoader:
    def __init__(self, paths: List[Path], args):
        self.paths = paths
        self.args = args
        self.images = list()

    def run(self):
        if self.args.processes > 1:
            self._multiprocess()
        else:
            self._process()

    def load_image(self, path):
        logger.info(f"Load image: {path}")
        return ImageLoaded(path, self.args)

    def _multiprocess(self):
        with ProcessPoolExecutor(max_workers=self.args.processes, mp_context=get_context('spawn')) as executor:
            future_to_filepath = {executor.submit(self.load_image, fp): fp for fp in self.paths}
            for future in as_completed(future_to_filepath):
                try:
                    image = future.result()
                    logger.debug(f"Loaded {image.filepath}")
                    self.images.append(image)

                except Exception as exc:
                    image.logger.info(f'Exception occurred: {exc}')
        logger.debug(f"images loaded: {len(self.images)}")
        self.images.sort()
        # multiprocessing makes a copy of args for each when using spawn, so let's write back the shared reference
        for image in self.images:
            image.args = self.args
        # Note: spawn is required for support on windows etc.
        # it is also required for the segmentation (can't recall exact reason right now)
        # anyway, sticking with this method here rather than fork due to wider support

    def _process(self):
        for fp in self.paths:
            try:
                self.images.append(self.load_image(fp))
            except Exception as exc:
                logger.info('%r generated an exception: %s' % (fp, exc))
        logger.debug(f"images processed: {len(self.images)}")
        self.images.sort()
