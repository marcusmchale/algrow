import argparse
import logging.config

from pathlib import Path
from open3d.visualization import gui

from .logging import LOGGING_CONFIG
from .options import options, postprocess, configuration_complete
from .area_calculation import calculate_area
from .analysis import analyse
from .gui import AppWindow

logger = logging.getLogger(__name__)
logging.config.dictConfig(LOGGING_CONFIG)


def algrow():
    arg_parser = options()
    args = arg_parser.parse_args()
    args = postprocess(args)

    # Ensure output directory exists
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir, 'algrow.log').touch(exist_ok=True)
    logger.warning(f"Start with: {arg_parser.format_values()}")

    if not configuration_complete(args):
        logger.info("Launching configuration window")
        try:
            launch_gui(args)
        except Exception as e:
            logger.error(f"Error running GUI required for configuration: {e}")
            raise
        if not configuration_complete(args):
            logger.warning("Configuration is not complete")
        else:
            logger.info("Configuration complete")
        logger.info("exiting")
        quit()

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

