from pathlib import Path
import logging
import logging.config

from src.logging import LOGGING_CONFIG

from .options.parse_args import options, postprocess
from .options.update_and_verify import calibration_complete
from .calibration.calibration import calibrate
from .area_calculation import calculate
from .analysis import analyse


def algrow():
    arg_parser = options()
    args = arg_parser.parse_args()
    args = postprocess(args)

    # Ensure output directory exists
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir, 'algrow.log').touch(exist_ok=True)

    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info(f"Start with: {arg_parser.format_values()}")

    if args.images is not None:
        logger.info(f"Processing {len(args.images)} images")
        if not calibration_complete(args) or args.force_calibration:
            logger.info("Launching calibration window")
            calibrate(args)
            logger.info("calibration complete")
            if not calibration_complete(args):
                logger.warning("Required arguments were not provided, exiting")
                return
        logger.info("Calculate area for input files")
        calculate(args)
        logger.info("Calculations complete")
    else:
        logger.info("No image files provided, continuing to RGR analysis")

    if args.samples is not None:
        logger.info("Analyse area file to calculate RGR")
        analyse(args)
        logger.info("Analysis complete")
    else:
        logger.info("No sample ID file provided")
