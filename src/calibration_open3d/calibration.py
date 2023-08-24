"""Interactively define parameters for layout detection and area calculation: circle colour, target hull points"""
import argparse
import logging
from open3d.visualization import gui
import numpy as np

from .app import AppWindow

logger = logging.getLogger(__name__)


def calibrate(args: argparse.Namespace):
    logger.debug("Initialise the app")
    gui.Application.instance.initialize()
    logger.debug("Get window")

    window = AppWindow(1024, 768, args, images=None)

    logger.debug("Run")
    gui.Application.instance.run()
