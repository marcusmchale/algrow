import argparse
import logging

import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path

from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2lab
from skimage.filters import gaussian
from skimage.transform import downscale_local_mean
from copy import deepcopy
from typing import List

from src.logging import worker_log_configurer, logger_thread, ImageFilepathAdapter

from .options.custom_types import DebugEnum
from .figurebuilder import FigureBase, FigureMatplot, FigureNone

logger = logging.getLogger(__name__)


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

    def load_image(self, path, queue=None):
        logger.info(f"Load image: {path}")
        if queue is not None:
            worker_log_configurer(queue)
        return ImageLoaded(path, self.args)

    def _multiprocess(self):
        queue = multiprocessing.Manager().Queue(-1)
        lp = threading.Thread(target=logger_thread, args=(queue,))
        lp.start()

        with ProcessPoolExecutor(max_workers=self.args.processes, mp_context=multiprocessing.get_context('spawn')) as executor:
            future_to_filepath = {executor.submit(self.load_image, path=fp, queue=queue): fp for fp in self.paths}
            for future in as_completed(future_to_filepath):
                try:
                    image = future.result()
                    logger.debug(f"Loaded {image.filepath}")
                    self.images.append(image)

                except Exception as exc:
                    image.logger.info(f'Exception occurred: {exc}')

        queue.put(None)
        lp.join()

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


class ImageLoaded:

    def __init__(self, filepath: Path, args: argparse.Namespace):
        self.args = args
        self.filepath = filepath

        self.logger = ImageFilepathAdapter(logger, {'image_filepath': str(filepath)})

        self.logger.debug(f"Read image from file")
        self.rgb = img_as_float(imread(str(filepath)))
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

