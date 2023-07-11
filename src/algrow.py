import logging
from pathlib import Path

from .options import options, postprocess, calibration_args_provided
from .calibration.calibration import calibrate
from .area_calculation import calculate
from .analysis import analyse


logger = logging.getLogger(__name__)


def algrow():
    arg_parser = options()
    args = arg_parser.parse_args()
    args = postprocess(args)

    logger.info(f"Start with: {arg_parser.format_values()}")

    # Ensure output directory exists
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.images is not None:
        logger.info(f"Processing {len(args.images)} images")
        if not calibration_args_provided(args):
            logger.debug("Launching calibration window")
            calibrate(args)
        calculate(args)
    else:
        logger.info("No image files provided")

    if args.sample_id is not None:
        analyse(args)
    else:
        logger.info("No sample ID file provided")



