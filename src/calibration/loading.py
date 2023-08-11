import logging

import wx
import multiprocessing
import threading
import time

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor

from skimage.color import lab2rgb

from ..logging import logger_thread, worker_log_configurer
from ..image_loading import ImageLoaded
from ..layout import Layout, LayoutLoader, LayoutDetector
from ..options.update_and_verify import layout_defined

logger = logging.getLogger(__name__)


def wait_for_results(message, processes, target_function, kwargs_list):
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
        logger.debug(f"{kwargs_list}")
        futures = [executor.submit(target_function, log_queue=log_queue, **kwargs) for kwargs in kwargs_list]
        while True:
            num_completed = sum([future.done() for future in futures])
            num_total = len(futures)
            complete_percent = int(num_completed / num_total * 100)
            # logger.debug(f"complete {complete_percent}")
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
        self.images = wait_for_results("Loading images", self.args.processes, self.load_image, kwargs_list)
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
        self.nearest = self.args.colourspace_rounding

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
        kwargs = [{"images": self.images}]
        results = wait_for_results("Summarising colours", 1, self.process_images, kwargs)
        self.pixel_to_lab, self.lab_to_pixel, self.filepath_lab_to_pixel, self.counts_all, self.counts_per_image = results[0]

    def process_images(self, log_queue=None):
        pass
        if log_queue is not None:
            worker_log_configurer(log_queue)

        cols = [("lab", "L"), ("lab", "a"), ("lab", "b")]
        cols_index = pd.MultiIndex.from_tuples(cols)
        dfs = [pd.DataFrame(image.lab.reshape(-1, 3), columns=cols_index, index=cols) for image in self.images]
        # these dataframes are not references to the image arrays
        keys = [image.filepath for image in self.images]
        if self.args.hull_vertices is not None:
            logger.debug("Append args points")
            dfs.append(pd.DataFrame(np.array(self.args.hull_vertices), columns=cols_index))
            keys.append("args")

        df = pd.concat(dfs, axis=0, keys=keys)
        df.index = df.index.set_names(["filepath", "pixel"])
        df = df.divide(self.nearest).round().multiply(self.nearest)

        counts_all = df.value_counts()
        # counts_all.index.to_numpy()
        counts_per_image = df.groupby(level="filepath").value_counts()
        pixel_to_lab = df.sort_index()
        df = df.reset_index()
        lab_to_pixel = df.reset_index().set_index(cols).sort_index()
        filepath_lab_to_pixel = df.reset_index().set_index(["filepath"] + cols).sort_index()
        # todo look into how we can use multiple indexes on one df

        return pixel_to_lab, lab_to_pixel, filepath_lab_to_pixel, counts_all, counts_per_image


class LayoutMultiLoader:
    def __init__(self, images: List[ImageLoaded]):
        self.images = images
        self.args = images[0].args
        self.layouts = list()

    def run(self):
        kwargs_list = [{"image": image} for image in self.images]
        self.layouts = wait_for_results("Loading layouts", self.args.processes, self.load_layout, kwargs_list)
        logger.debug(f"Layouts loaded: {len([l for l in self.layouts if l is not None])}")
        # multiprocessing makes a copy of args for each when using spawn, so we need to write back a shared reference
        for layout in self.layouts:
            if layout is not None:
                layout.args = self.args
        # Note: spawn is required for support on windows etc.
        # sticking with this method here rather than fork due to wider support

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
                layout = LayoutDetector(image).get_layout()
            if log_queue is not None:
                worker_log_configurer(log_queue)
        else:
            layout = None
        return layout
