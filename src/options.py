import logging
import argparse
from configargparse import ArgumentParser
from pathlib import Path
from itertools import chain

logger = logging.getLogger(__name__)


def lab(s: str | tuple[float|int]):
    if isinstance(s, tuple):
        if len(s) > 3:
            raise argparse.ArgumentTypeError(f'Colour must be a tuple with 3 float or int values: {s}')
        else:
            try:
                for i in s:
                    float(i)
            except ValueError:
                raise argparse.ArgumentTypeError(f'Colour must be a tuple with 3 float or int values: {s}')
        return s
    elif isinstance(s, str):
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

    # coerce to known type:
    try:
        val = arg_types[arg](val)
    except ValueError:
        logger.debug("Issue with updating arg")
        raise

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
        args.scale is not None,
        args.circle_diameter is not None,
        args.plate_circle_separation is not None,
        args.plate_width is not None,
        args.circles_per_plate is not None,
        args.n_plates is not None,
        args.circles_cols_first is not None,
        args.circles_right_left is not None,
        args.circles_bottom_top is not None,
        args.plates_cols_first is not None,
        args.plates_bottom_top is not None,
        args.plates_right_left is not None
    ])


def layout_args_provided(args):
    return all([
        args.circle_colour is not None,
        args.circle_diameter is not None,
        args.plate_circle_separation is not None,
        args.plate_width is not None,
        args.circles_per_plate is not None,
        args.n_plates is not None,
        args.circles_cols_first is not None,
        args.circles_right_left is not None,
        args.circles_bottom_top is not None,
        args.plates_cols_first is not None,
        args.plates_bottom_top is not None,
        args.plates_right_left is not None
    ])


arg_types = {
    "conf": str,
    "images": image_path,
    "sample_id": str,
    "out_dir": str,
    "time_regex": str,
    "time_format": str,
    "block_regex": str,
    "overlay": bool,
    "animations": bool,
    "processes": int,
    "debug": str,
    "loglevel": str,
    "num_superpixels": int,
    "superpixel_compactness": float,
    "sigma": float,
    "circle_colour": lab,
    "hull_vertices": lab,
    "alpha": float,
    "delta": float,
    "num_calibration": int,
    "remove": int,
    "fill": int,
    "area_file": str,
    "fit_start": float,
    "fit_end": float,
    "scale": float,
    "circle_diameter": float,
    "circle_expansion": float,
    "plate_circle_separation": float,
    "plate_width": float,
    "circles_per_plate": int,
    "n_plates": int,
    "plates_cols_first": bool,
    "plates_right_left": bool,
    "plates_bottom_top": bool,
    "circles_cols_first": bool,
    "circles_right_left": bool,
    "circles_bottom_top": bool
}


# Parse command-line arguments
def options():
    config_dir = Path(Path(__file__).parent.parent, "conf.d")
    config_files = config_dir.glob("*.conf")
    parser = ArgumentParser(
        default_config_files=config_files,
    )
    parser.add_argument("-c", "--conf", help="Config file path", is_config_file=True, type=arg_types["conf"])
    parser.add_argument(
        "-i",
        "--images",
        help="Input image file or directory",
        type=arg_types["images"],
        default=None,
        action='append')
    parser.add_argument(
        "-id",
        "--sample_id",
        help="Input csv file with sample identities (block, unit, group)",
        type=arg_types["sample_id"],
        default=None
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        help="Output directory",
        default=".",
        type=arg_types["out_dir"]
    )
    parser.add_argument(
        "-tr",
        "--time_regex",
        help="Regex pattern to identify date time string from filename",
        type=arg_types["time_regex"]
    )
    parser.add_argument(
        "-tf",
        "--time_format",
        help="String format pattern to read datetime object from regex match",
        type=arg_types["time_format"]
    )
    parser.add_argument(
        "-br",
        "--block_regex",
        help="Regex pattern to identify block string from filename",
        type=arg_types["block_regex"]
    )
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
        type=arg_types["processes"]
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
        type=arg_types["num_superpixels"],
        default=1000
    )
    parser.add_argument(
        '-spc', "--superpixel_compactness",
        help="Superpixel compactness, higher is more regular (square)",
        type=arg_types["superpixel_compactness"],
        default=10
    )
    parser.add_argument(
        '-sig', "--sigma",
        help="Smoothing kernel applied before superpixel clustering",
        type=arg_types["sigma"],
        default=1
    )
    parser.add_argument(
        "-cc", "--circle_colour",
        help="Circle colour in Lab colourspace",
        type=arg_types["circle_colour"],
        default=None
    )
    parser.add_argument(
        "-hv", "--hull_vertices",
        help="Points in Lab colourspace that define the alpha hull, at least 4 points are required",
        type=arg_types["hull_vertices"],
        default=None,
        action='append'
    )
    parser.add_argument(
        "-al", "--alpha",
        help="Alpha value used to construct a hull (or hulls) around the provided hull vertices",
        type=arg_types["alpha"],
        default=0
    )
    parser.add_argument(
        "-de", "--delta",
        help="Maximum distance outside of target polygon to consider as target",
        type=arg_types["delta"],
        default=0
    )
    parser.add_argument(
        "-nc", "--num_calibration",
        help="Number of images to use for calibration",
        type=arg_types["num_calibration"],
        default=3
    )
    parser.add_argument(
        "-r",
        "--remove",
        help="Set remove size (px)",
        default=100,
        type=arg_types["remove"]
    )
    parser.add_argument(
        "-f",
        "--fill",
        help="Set fill size (px)",
        default=100,
        type=arg_types["fill"]
    )

    parser.add_argument(
        "-fs",
        "--fit_start",
        help="Start (day) for RGR calculation",
        type=arg_types["fit_start"],
        default=float(0)
    )
    parser.add_argument(
        "-fe",
        "--fit_end",
        help="End (day) for RGR calculation",
        type=arg_types["fit_end"],
        default=float('inf')
    )
    parser.add_argument(
        "-ao",
        "--area_file",
        help="Disc area filename for analysis (must be in the output directory)",
        type=arg_types["area_file"],
        default="area.csv"
    )

    parser.add_argument(
        "-sc",
        "--scale",
        help="pixels/unit distance for area calculation (if unit distance is mm then area will be reported in mm²)",
        default=None,
        type=arg_types["scale"]
    )
    parser.add_argument(
        "-cd",
        "--circle_diameter",
        help="Diameter of surrounding circles in pixels",
        default=None,
        type=arg_types["circle_diameter"]
    )
    parser.add_argument(
        "-ce",
        "--circle_expansion",
        help="Optional expansion factor for circles (increases radius to search, circles must not overlap)",
        default=1,
        type=arg_types["circle_expansion"]
    )
    parser.add_argument(
        "-pcs",
        "--plate_circle_separation",
        help="Distance between edges of circles within a plate (px)",
        default=None,
        type=arg_types["plate_circle_separation"]
    )
    parser.add_argument(
        "-pw",
        "--plate_width",
        help="Length of shortest edge of plate (px)",
        default=None,
        type=arg_types["plate_width"]
    )
    parser.add_argument(
        "-cpp",
        "--circles_per_plate",
        help="In plate clustering, the number of circles per plate",
        default=None,
        type=arg_types["circles_per_plate"]
    )
    parser.add_argument(
        "-npi",
        "--n_plates",
        help="In plate layout, the number of plates per image",
        default=None,
        type=arg_types["n_plates"]
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
    return parser
