"""Interactively define parameters for layout detection and area calculation: circle colour, target hull points"""
import argparse
import logging

import numpy as np

from .app import Calibrator

from ..image_loading import ImageLoader

logger = logging.getLogger(__name__)

def calibrate(args: argparse.Namespace):

    # calibrate on a subset of images only
    vars(args)['images'].sort()
    idx = np.unique(np.round(np.linspace(0, len(args.images) - 1, args.num_calibration)).astype(int))
    sample_image_paths = list(np.unique(np.array(args.images)[idx]))
    logger.info(f"Sample {min(args.num_calibration, len(sample_image_paths))} images for calibration")

    calibrator = Calibrator(sample_image_paths, args)
    calibrator.MainLoop()

