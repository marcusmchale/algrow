"""Interactively define parameters for layout detection and area calculation: circle colour, target hull points"""
import argparse
import logging

import numpy as np

from ..figurebuilder import FigureBuilder
from .app import Calibrator

from ..image_loading import ImageLoaded

logger = logging.getLogger(__name__)


def calibrate(image_filepaths, args: argparse.Namespace):
    # calibrate on a subset of images only
    idx = np.unique(np.round(np.linspace(0, len(image_filepaths) - 1, args.num_calibration)).astype(int))
    sample_image_paths = list(np.array(image_filepaths)[idx])
    logger.info(f"Sample {args.num_calibration} images for calibration")
    images = [ImageLoaded(i, args) for i in sample_image_paths]

    calibrator = Calibrator(images)
    calibrator.MainLoop()

    if args.debug:  # add debug figure with circle colour
        logger.debug(f"Plotting debug image for circle colour: {vars(args)['circle_colour']}")
        fig = FigureBuilder(".", args, "Circle colour")
        fig.plot_colours([args.circle_colour])
        fig.print()

        fig = FigureBuilder(".", args, "Target colours")
        fig.plot_colours(args.hull_vertices)
        fig.print()


