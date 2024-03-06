import argparse
import logging.config

import sys

from pathlib import Path
from open3d.visualization import gui

from .logging import LOGGING_CONFIG
from .options import options, postprocess, configuration_complete
from .area_calculation import calculate_area
from .analysis import analyse
from .gui import AppWindow

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def run():
    arg_parser = options()
    args, _ = arg_parser.parse_known_args()
    #args = arg_parser.parse_args()  # parse_args breaks pyinstaller in combination with multiprocessing
    args = postprocess(args)

    #logger = logging.getLogger('src.algrow')  # need this when using setuptools as the script is compiled in another environment
    algrow_logger = logging.getLogger('algrow')
    algrow_logger.setLevel(args.loglevel.name)

    # Ensure output directory exists
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir, 'algrow.log').touch(exist_ok=True)

    logger.info(f"Start with: {arg_parser.format_values()}")

    if not configuration_complete(args):
        logger.info("Launching AlGrow GUI")
        try:
            launch_gui(args)
        except Exception as e:
            logger.error(f"Error running GUI required for configuration: {e}")
            raise
        logger.info("Exiting AlGrow GUI")
        sys.exit()

    if args.images is not None:
        logger.info(f"Processing {len(args.images)} images")
        logger.info("Calculate area for input files")
        calculate_area(args)
        logger.info("Calculations complete")
    else:
        logger.info("No image files provided, continuing to RGR analysis")

    if args.samples is not None and args.area_file is not None:
        logger.info("Analyse area file to calculate RGR")
        analyse(args)
        logger.info("Analysis complete")
    else:
        logger.info("No sample ID file provided")


def launch_gui(args: argparse.Namespace):
    fonts = dict()
    logger.debug("Initialise the app")
    app = gui.Application.instance
    app.initialize()
    fonts['large'] = app.add_font(gui.FontDescription(style=gui.FontStyle.NORMAL, point_size=20))  # font id 0
    fonts['small'] = app.add_font(gui.FontDescription(style=gui.FontStyle.NORMAL, point_size=15))  # font id 0
    logger.debug("Get window")
    AppWindow(1920, 1080, fonts, args)
    logger.debug("Run")
    gui.Application.instance.run()
