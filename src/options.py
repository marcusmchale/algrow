import logging
import argparse
from configargparse import ArgumentParser
from pathlib import Path
from itertools import chain

logger = logging.getLogger(__name__)


def lab(s: str):
    try:
        l, a, b = map(float, s.split(','))
        return l, a, b
    except ValueError:
        raise argparse.ArgumentTypeError(f'Each colour must be a string with 3 comma separated float values: {s}')


def update_arg(args, arg, val):
    if isinstance(val, list) and isinstance(val[0], tuple):
        val_str = f'{[",".join([str(j) for j in i]) for i in val]}'.replace("'", '"')
    elif isinstance(val, tuple):
        val_str = f"\"{','.join([str(i) for i in val])}\""
    else:
        val_str = str(val)
    if vars(args)[arg] is None:
        logger.info(f"Setting {arg}: {val_str}")
        vars(args).update({arg: val})
    else:
        if vars(args)[arg] == val:
            logger.info(f"Existing value matches the update so no change will be made {arg}: {val}")
        else:
            logger.info(f"Overwriting configured value for {arg}: {vars(args)[arg]} will be set to {val}")
            vars(args).update({arg: val})


def image_path(s: str):
    if Path(s).is_file():
        return Path(s)
    elif Path(s).is_dir():
        return [p for p in Path(s).glob('**/*.jpg')]
    else:
        raise FileNotFoundError


def postprocess(args):
    # Handle multiple directories or a mix of directories and files
    if args.images is not None:
        args.images = list(chain(*args.images))

    # Infer out dir if not specified
    if args.out_dir is None:
        if args.images is not None:
            if Path(args.images[0]).is_file():
                args.out_dir = Path(args.images[0]).parents[0]
            else:
                args.out_dir = Path(args.images[0])
        elif args.id is not None:
            args.out_dir = Path(args.id).parents[0]
        #  else:  # should return default local director as out_dir
        #      raise FileNotFoundError("No output directory specified")
    return args


def calibration_args_provided(args):
    return all([
        (args.hull_vertices is not None and len(args.hull_vertices) > 4),
        args.circle_colour is not None,
        args.scale is not None
    ])


# Parse command-line arguments
def options():
    config_dir = Path(Path(__file__).parent.parent, "conf.d")
    config_files = config_dir.glob("*.conf")
    parser = ArgumentParser(
        default_config_files=config_files,
    )
    parser.add_argument("-c", "--conf", help ="Config file path", is_config_file=True)
    parser.add_argument(
        "-i",
        "--images",
        help="Input image file or directory",
        type=image_path,
        default=None,
        action='append')
    parser.add_argument(
        "-id",
        "--sample_id",
        help="Input csv file with sample identities (block, unit, group)",
        default=None
    )
    parser.add_argument("-o", "--out_dir", help="Output directory", default=".")
    parser.add_argument("-tr", "--time_regex", help="Regex pattern to identify date time string from filename")
    parser.add_argument("-tf", "--time_format", help="String format pattern to read datetime object from regex match")
    parser.add_argument("-br", "--block_regex", help="Regex pattern to identify block string from filename")
    parser.add_argument(
        "-q",
        "--overlay",
        action='store_true',
        help="Write out overlay for quality control"
    )
    parser.add_argument(
        '-an',
        "--animations",
        help="Use imagemagick to generate gif animations of 3d plots",
        action='store_true'
    )
    parser.add_argument(
        "-p",
        "--processes",
        help="Number of processes to launch (images to concurrently process)",
        default=1,
        type=int
    )
    parser.add_argument(
        "-d",
        "--debug",
        help=(
            "Plots intermediate images for debugging/tuning"
        ),
        choices=["save", "plot", "both"],
        default=None
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        help=(
            "Set log-level: INFO or DEBUG"
        ),
        choices=["INFO", "DEBUG"],
        default=None
    )
    parser.add_argument(
        '-nsp', "--num_superpixels",
        help="The number of superpixels to find in image",
        type=int,
        default=1000
    )
    parser.add_argument(
        '-spc', "--superpixel_compactness",
        help="Superpixel compactness, higher is more regular (square)",
        type=float,
        default=10
    )
    parser.add_argument(
        '-sig', "--sigma",
        help="Smoothing kernel applied before superpixel clustering",
        type=float,
        default=1
    )
    parser.add_argument(
        "-cc", "--circle_colour",
        help="Circle colour in Lab colourspace",
        type=lab,
        default=None
    )
    parser.add_argument(
        "-hv", "--hull_vertices",
        help="Points in Lab colourspace that define the alpha hull, at least 4 points are required",
        type=lab,
        default=None,
        action='append'
    )
    parser.add_argument(
        "-al", "--alpha",
        help="Alpha value used to construct a hull (or hulls) around the provided hull vertices",
        type=float,
        default=0
    )
    parser.add_argument(
        "-de", "--delta",
        help="Maximum distance outside of target polygon to consider as target",
        type=float,
        default=10
    )
    parser.add_argument(
        "-nc", "--num_calibration",
        help="Number of images to use for calibration",
        type=int,
        default=3
    )
    parser.add_argument(
        "-r",
        "--remove",
        help="Set remove size (px)",
        default=100,
        type=int
    )
    parser.add_argument(
        "-f",
        "--fill",
        help="Set fill size (px)",
        default=100,
        type=int
    )

    parser.add_argument("-fs", "--fit_start", help="Start (day) for RGR calculation", type=int, default=0)
    parser.add_argument("-fe", "--fit_end", help="End (day) for RGR calculation", type=int, default=float('inf'))
    parser.add_argument(
        "-ao",
        "--area_file",
        help="Disc area filename for analysis (must be in the output directory)",
        type=str,
        default="area.csv"
    )

    parser.add_argument(
        "-sc",
        "--scale",
        help="pixels/unit distance for area calculation (if unit distance is mm then area will be reported in mm²)",
        default=None,
        type=float
    )
    parser.add_argument(
        "-cd",
        "--circle_diameter",
        help="Diameter of surrounding circles in pixels",
        default=180,
        type=float
    )
    parser.add_argument(
        "-ce",
        "--circle_expansion",
        help="Optional expansion factor for circles (increases radius to search, circles must not overlap)",
        default=1,
        type=float
    )
    parser.add_argument(
        "-pcs",
        "--plate_circle_separation",
        help="Distance between edges of circles within a plate (px)",
        default=50,
        type=float
    )
    parser.add_argument(
        "-pw",
        "--plate_width",
        help="Length of shortest edge of plate (px)",
        default=500,
        type=float
    )
    parser.add_argument(
        "-cpp",
        "--circles_per_plate",
        help="In plate clustering, the number of circles per plate",
        default=6,
        type=int
    )
    parser.add_argument(
        "-npi",
        "--n_plates",
        help="In plate layout, the number of plates per image",
        default=8,
        type=int
    )
    parser.add_argument(
        "-ccf",
        "--circles_cols_first",
        help="In circle ID layout, increment by columns first",
        action='store_true'
    )
    parser.add_argument(
        "-clr",
        "--circles_right_left",
        help="In circle ID layout, increment right to left",
        action='store_true',
    )
    parser.add_argument(
        "-cbt",
        "--circles_bottom_top",
        help="In circle ID layout, increment from bottom to top)",
        action='store_true'
    )
    parser.add_argument(
        "-pcf",
        "--plates_cols_first",
        help="In plate ID layout, increment by columns first",
        action='store_true',
    )
    parser.add_argument(
        "-prl",
        "--plates_right_left",
        help="In plate ID layout, increment from right to left",
        action='store_true',
    )
    parser.add_argument(
        "-pbt",
        "--plates_bottom_top",
        help="In plate ID layout, increment from bottom to top",
        action='store_true'
    )
    return parser
