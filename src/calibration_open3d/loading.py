import logging
import multiprocessing
import time
from open3d.visualization import gui

import threading
import wx
from queue import Queue
from concurrent.futures import ProcessPoolExecutor

from pathlib import Path

from typing import List
from ..image_loading import ImageLoaded
from ..logging import worker_log_configurer, logger_thread

logger = logging.getLogger(__name__)


def wait_for_result(func):

    def wrapped_func(q, *args, **kwargs):
        result = func(*args, **kwargs)
        logger.debug("return result in queue")
        q.put(result)

    def wrap(*args, **kwargs):
        bar = gui.ProgressBar()

        q = Queue()
        t = threading.Thread(target=wrapped_func, args=(q,)+args, kwargs=kwargs)
        logger.debug("Start thread")
        t.start()

        # todo need to actually handle progress updates, this will do for now
        bar.value = 0.25
        while t.is_alive():
            bar.value = 0.5
        bar.value = 1
        logger.debug("Get result from thread")
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