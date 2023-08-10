import argparse
import logging

import numpy as np
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path

from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import downscale_local_mean
from copy import deepcopy
from typing import List

from src.logging import worker_log_configurer, logger_thread, ImageFilepathAdapter

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


class ImageLoader:
    def __init__(self, paths: List[Path], args):
        self.paths = paths
        self.args = args
        self.images: List[ImageLoaded] = list()

    def run(self, progress_callback):
        log_queue = multiprocessing.Manager().Queue(-1)
        lp = threading.Thread(target=logger_thread, args=(log_queue,))
        lp.start()

        with ProcessPoolExecutor(
                max_workers=self.args.processes,
                mp_context=multiprocessing.get_context('spawn')
        ) as executor:
            futures = [executor.submit(self.load_image, path=fp, log_queue=log_queue) for fp in self.paths]

            progress_callback(message="Loading images")

            while True:
                num_completed = sum([future.done() for future in futures])
                num_total = len(futures)
                complete_percent = int(num_completed/num_total * 100)
                #logger.debug(f"complete {complete_percent}")
                if complete_percent == 100:
                    logger.debug(f"completed all")
                    break
                time.sleep(0.1)
                progress_callback(complete=complete_percent)

            for future in futures:
                try:
                    image = future.result()
                    logger.debug(f"Loaded {image.filepath}")
                    self.images.append(image)

                except Exception as exc:
                    logger.info(f'Exception occurred during image loading: {exc}')

        log_queue.put(None)
        lp.join()

        logger.debug(f"images loaded: {len(self.images)}")
        self.images.sort()
        # multiprocessing makes a copy of args for each when using spawn, so let's write back the shared reference
        for image in self.images:
            image.args = self.args
        # Note: spawn is required for support on windows etc.
        # it is also required for the segmentation (can't recall exact reason right now)
        # anyway, sticking with this method here rather than fork due to wider support

    def load_image(self, path, log_queue=None):
        logger.info(f"Load image: {path}")
        if log_queue is not None:
            worker_log_configurer(log_queue)
        return ImageLoaded(path, self.args)
