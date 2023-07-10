import logging
from pathlib import Path

from .options import options
from .calibration.calibration import calibrate
from .area_calculation import calculate
from .analysis import analyse


logger = logging.getLogger(__name__)


def algrow():
    argparser = options()
    args = argparser.parse_args()
    logger.info(f"Start with: {argparser.format_values()}")

    # Organise input image file(s)
    image_filepaths = []
    if args.image:
        for i in args.image:
            if Path(i).is_file():
                image_filepaths.append(Path(i))
            elif Path(i).is_dir():
                image_filepaths = image_filepaths + [p for p in Path(i).glob('**/*.jpg')]
            else:
                raise FileNotFoundError
        logger.info(f"Processing {len(image_filepaths)} images")

    # Organise output directory
    if not args.out_dir:
        if Path(args.image[0]).is_file():
            args.out_dir = Path(args.image[0]).parents[0]
        else:
            args.out_dir = Path(args.image[0])
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if image_filepaths:
        calibrate(image_filepaths, args)
        calculate(image_filepaths, args)
    else:
        logger.info("No image files provided")

    if args.sample_id:
        analyse(args)
    else:
        logger.info("No sample ID file provided")



