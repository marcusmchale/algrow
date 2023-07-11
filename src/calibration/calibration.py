"""Interactively define parameters for layout detection and area calculation: circle colour, target hull points"""
import argparse
import logging

import numpy as np

from ..figurebuilder import FigureBuilder
from .app import Calibrator

from ..image_loading import ImageLoaded

logger = logging.getLogger(__name__)


def calibrate(args: argparse.Namespace):
    # calibrate on a subset of images only
    idx = np.unique(np.round(np.linspace(0, len(args.images) - 1, args.num_calibration)).astype(int))
    sample_image_paths = list(np.array(args.images)[idx])
    logger.info(f"Sample {args.num_calibration} images for calibration")
    images = [ImageLoaded(i, args) for i in sample_image_paths]

    calibrator = Calibrator(images)
    calibrator.MainLoop()

