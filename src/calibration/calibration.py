"""Interactively define parameters for layout detection and area calculation: circle colour, target hull points"""
import argparse
import logging

from open3d.visualization import gui

from .app_o3d import AppWindow

logger = logging.getLogger(__name__)

fonts = dict()


def calibrate(args: argparse.Namespace):
    logger.debug("Initialise the app")
    app = gui.Application.instance
    app.initialize()
    fonts['large'] = app.add_font(gui.FontDescription(style=gui.FontStyle.NORMAL, point_size=20))  # font id 0
    fonts['small'] = app.add_font(gui.FontDescription(style=gui.FontStyle.NORMAL, point_size=15))  # font id 0
    logger.debug("Get window")
    AppWindow(1920, 1080, fonts, args)
    logger.debug("Run")
    gui.Application.instance.run()


