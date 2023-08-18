import logging

import wx
import multiprocessing
import threading
import time

import numpy as np
import pandas as pd

from queue import Queue

from pathlib import Path
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor

from skimage.color import lab2rgb

from ..logging import logger_thread, worker_log_configurer
from ..image_loading import ImageLoaded
from ..layout import Layout, LayoutLoader, LayoutDetector, InsufficientPlateDetection, InsufficientCircleDetection
from ..options.update_and_verify import layout_defined

logger = logging.getLogger(__name__)


def wait_for_result(func):

    def wrapped_func(q, *args, **kwargs):
        result = func(*args, **kwargs)
        q.put(result)

    def wrap(*args, **kwargs):
        dialog = wx.ProgressDialog(
            "Algrow Calibration",
            "Please wait...",
            maximum=100,
            parent=None,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE
        )
        wx.Yield()

        def progress_callback(complete: int = None):
            if complete is None:
                dialog.Pulse()
            else:
                dialog.Update(complete)
            wx.Yield()
            time.sleep(0.1)

        q = Queue()
        t = threading.Thread(target=wrapped_func, args=(q,)+args, kwargs=kwargs)
        t.start()

        while t.is_alive():
            progress_callback()
        progress_callback(100)

        return q.get()

    return wrap


def wait_for_multiprocessing(message, processes, target_function, kwargs_list=None):
    dialog = wx.ProgressDialog(
        "Algrow Calibration",
        message,
        maximum=100,
        parent=None,
        style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE
    )
    wx.Yield()

    def progress_callback(complete: int = None):
        if complete is None:
            dialog.Pulse()
        else:
            dialog.Update(complete)
        wx.Yield()
        time.sleep(0.1)

    log_queue = multiprocessing.Manager().Queue(-1)
    lp = threading.Thread(target=logger_thread, args=(log_queue,))
    lp.start()

    results = list()
    with ProcessPoolExecutor(
            max_workers=processes,
            mp_context=multiprocessing.get_context('spawn')
    ) as executor:
        if kwargs_list:
            futures = [executor.submit(target_function, log_queue=log_queue, **kwargs) for kwargs in kwargs_list]
        else:
            futures = [executor.submit(target_function, log_queue=log_queue)]
        while True:
            num_completed = sum([future.done() for future in futures])
            num_total = len(futures)
            complete_percent = int(num_completed / num_total) * 100
            if complete_percent == 100:
                logger.debug(f"completed all")
                break
            time.sleep(0.1)
            progress_callback(complete=complete_percent)

        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                logger.info(
                    f'Exception occurred during multiprocessing: {target_function} {kwargs_list} {future} {exc}'
                )

    log_queue.put(None)
    lp.join()

    return results


class ImageLoader:
    def __init__(self, paths: List[Path], args):
        self.paths = paths
        self.args = args
        self.images: List[ImageLoaded] = list()

    def run(self):
        kwargs_list = [{"path": fp} for fp in self.paths]
        self.images = wait_for_multiprocessing("Loading images", self.args.processes, self.load_image, kwargs_list)
        logger.debug(f"images loaded: {len(self.images)}")
        self.images.sort()
        # multiprocessing makes a copy of args for each when using spawn, so we need to write back a shared reference
        for image in self.images:
            image.args = self.args
        # Note: spawn is required for support on windows etc.
        # sticking with this method here rather than fork due to wider support

    def load_image(self, path, log_queue=None):  # loq queue is required by wait_for_results to log properly
        logger.info(f"Load image: {path}")
        if log_queue is not None:
            worker_log_configurer(log_queue)
        return ImageLoaded(path, self.args)


class Points:
    def __init__(self, images: List[ImageLoaded]):
        self.images = images
        self.args = self.images[0].args
        self.nearest = self.args.colour_rounding

        # Start with colour bins at some grouping (nearest)
        # Each bin has:
        #    - lab (coordinates in Lab space)
        #    - rgb (colour for plotting)
        #    - count total (n pixels all images)
        #    - count per file (n pixels from each image)
        #    - indices per file (to map to x,y)
        self.pixel_to_lab = None
        self.lab_to_pixel = None
        self.filepath_lab_to_pixel = None
        self.counts_all = None
        self.counts_per_image = None

    def calculate(self):
        results = wait_for_multiprocessing("Summarising colours", 1, self.process_images)
        self.pixel_to_lab, self.lab_to_pixel, self.filepath_lab_to_pixel, self.counts_all, self.counts_per_image = results[0]
        logger.debug("Set distance to infinity (will be computed later when have a hull)")
        self.counts_per_image['distance'] = np.inf  # a placeholder until we calculate distance from hull later

    def process_images(self, log_queue=None):
        if log_queue is not None:
            worker_log_configurer(log_queue)

        cols = [("lab", "L"), ("lab", "a"), ("lab", "b")]
        cols_index = pd.MultiIndex.from_tuples(cols)
        logger.debug("Get list of images as dataframes")
        dfs = [pd.DataFrame(image.lab.reshape(-1, 3), columns=cols_index) for image in self.images]
        # these dataframes are not references to the image arrays
        keys = [str(image.filepath) for image in self.images]
        if self.args.hull_vertices is not None:
            logger.debug("Append args points")
            dfs.append(pd.DataFrame(np.array(self.args.hull_vertices), columns=cols_index))
            keys.append("args")

        logger.debug(f"Concatenate and round to nearest {self.nearest}")
        pixel_to_lab = pd.concat(dfs, axis=0, keys=keys)
        pixel_to_lab.index = pixel_to_lab.index.set_names(["filepath", "pixel"])
        pixel_to_lab = pixel_to_lab.divide(self.nearest).round().multiply(self.nearest)

        logger.debug(f"Count pixels per colour, set and sort index")
        counts_all = pixel_to_lab.value_counts().reset_index().sort_index()
        logger.debug(f"Calculate RGB for each colour")
        counts_all[[("rgb", "r"),("rgb", "g"),("rgb", "b")]] = lab2rgb(counts_all[cols])

        logger.debug(f"Count pixels per colour per file, set and sort index")
        counts_per_image = pixel_to_lab.groupby(level="filepath").value_counts().reset_index().set_index("filepath").sort_index()
        logger.debug(f"Calculate RGB for each colour per file")
        counts_per_image[[("rgb", "r"), ("rgb", "g"), ("rgb", "b")]] = lab2rgb(counts_per_image[cols])

        logger.debug(f"More indexing for pixel to colour and colour to pixel")
        pixel_to_lab = pixel_to_lab.sort_index()
        lab_to_pixel = pixel_to_lab.reset_index().set_index(cols).sort_index()
        filepath_lab_to_pixel = pixel_to_lab.reset_index().set_index(["filepath"] + cols).sort_index()

        return pixel_to_lab, lab_to_pixel, filepath_lab_to_pixel, counts_all, counts_per_image


class LayoutMultiLoader:
    def __init__(self, images: List[ImageLoaded]):
        self.images = images
        self.args = images[0].args
        self.layouts = list()

    def run(self):
        kwargs_list = [{"image": image} for image in self.images]
        self.layouts = wait_for_multiprocessing("Loading layouts", self.args.processes, self.load_layout, kwargs_list)
        logger.debug(f"Layouts loaded: {len([l for l in self.layouts if l is not None])}")
        # multiprocessing makes a copy of args for each when using spawn, so we need to write back a shared reference
        for layout in self.layouts:
            if layout is not None:
                layout.args = self.args

    def load_layout(self, image, log_queue=None) -> Optional[Layout]:
        if log_queue is not None:
            worker_log_configurer(log_queue)
        logger.info(f"Load layout for image: {image.filepath}")
        if layout_defined(self.args):
            if self.args.whole_image:
                layout = None
            elif self.args.fixed_layout is not None:
                layout = LayoutLoader(image).get_layout()
            else:
                try:
                    layout = LayoutDetector(image).get_layout()
                except (InsufficientPlateDetection, InsufficientCircleDetection) as e:
                    logger.warning(f"Failed to detect layout {e}")
                    layout = None
            if log_queue is not None:
                worker_log_configurer(log_queue)
        else:
            layout = None
        return layout
