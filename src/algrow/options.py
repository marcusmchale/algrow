import logging
import argparse

from configargparse import ArgumentParser
from itertools import chain
from enum import IntEnum
from pathlib import Path
from typing import Optional, Union, List

logger = logging.getLogger(__name__)


# Types and constants
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp"]


class DebugEnum(IntEnum):
    DEBUG = 0
    INFO = 1
    WARN = 2


def debug_level(s: str):
    try:
        return DebugEnum[s.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f'Log level must be one of: {[d.name for d in DebugEnum]}')


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


def image_path(s: Union[None, str, List[str]]) -> Union[None, List[Path]]:
    if s is None:
        return None
    elif isinstance(s, list):
        return [Path(i) for i in s]
    elif Path(s).is_file():
        return [Path(s)]
    elif Path(s).is_dir():
        paths = [[p for p in Path(s).glob(f'**/*.{extension}')] for extension in IMAGE_EXTENSIONS]
        return [p for sublist in paths for p in sublist]
    else:
        raise FileNotFoundError(s)


def filepath(s: Optional[str]):
    if s is None:
        return None
    else:
        return Path(s)


arg_types = {
    "conf": str,
    "images": image_path,
    "samples": filepath,
    "out_dir": str,
    "filename_regex": str,
    "animations": bool,
    "processes": int,
    "detect_layout": bool,
    "image_debug": debug_level,
    "loglevel": debug_level,
    "voxel_size": float,
    "superpixels": int,
    "slic_iter": int,
    "compactness": float,
    "sigma": float,
    "circle_colour": lab,
    "hull_vertices": lab,
    "alpha": float,
    "delta": float,
    "downscale": int,
    "remove": int,
    "fill": int,
    "area_file": filepath,
    "fit_start": float,
    "fit_end": float,
    "scale": float,
    "fixed_layout": filepath,
    "circle_diameter": float,
    "circle_variability": float,
    "circle_expansion": float,
    "circle_separation": float,
    "circle_separation_tolerance": float,
    "plate_width": float,
    "circles": int,
    "circles_per_plate": int,
    "plates": int,
    "plates_cols_first": bool,
    "plates_right_left": bool,
    "plates_bottom_top": bool,
    "circles_cols_first": bool,
    "circles_right_left": bool,
    "circles_bottom_top": bool,
    "denoise": bool,
    "canny_sigma": float,
    "canny_low": float,
    "canny_high": float
}


# Parse command-line arguments
def options(filepath=None):
    #config_dir = Path(Path(__file__).parent.parent, "conf.d")
    #config_files = config_dir.glob("*.conf")
    #config_files = [str(i) for i in config_files]
    config_files = list()
    if filepath is not None:
        config_files.append(filepath)
    parser = ArgumentParser(
        default_config_files=[str(i) for i in config_files],
    )
    parser.add_argument("--conf", help="Config file path", is_config_file=True, type=arg_types["conf"])
    parser.add_argument(
        "-i", "--images",
        help="Input image file or directory",
        type=arg_types["images"],
        default=None,
        action='append')
    parser.add_argument(
        "-s", "--samples",
        help="Input csv file with sample identities in columns (block, unit, group)",
        type=arg_types["samples"],
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
        "-p", "--processes",
        help="Number of processes to launch (images to concurrently process)",
        default=1,
        type=arg_types["processes"]
    )
    parser.add_argument(
        "-l", "--detect_layout",
        help="Run without layout definition to calculate target area for the entire image",
        action='store_true'
    )
    parser.add_argument(
        "--fixed_layout",
        help="Path to a plate layout definition file (generated during calibration by setting fixed layout)",
        default=None,
        type=arg_types["fixed_layout"]
    )
    parser.add_argument(
        "-d", "--image_debug",
        help="Level of image debugging",
        type=arg_types['image_debug'],
        default="INFO"
    )
    parser.add_argument(
        "--filename_regex",
        help="Regex pattern to capture named groups from filename (year, month, day, hour, minute, second, block)",
        type=arg_types["filename_regex"],
        default=".*(?P<year>[0-9]{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12][0-9]|3[01])_(?P<hour>[01][0-9]|2[0-4])h(?P<minute>[0-5][0-9])m_(?P<block>[a-z|A-Z|0-9]*).*"
    )
    parser.add_argument(
        "--circle_colour",
        help="Circle colour in Lab colourspace",
        type=arg_types["circle_colour"],
        default=None
    )
    parser.add_argument(
        "--hull_vertices",
        help="Points in Lab colourspace that define the alpha hull, at least 4 points are required",
        type=arg_types["hull_vertices"],
        default=None,
        action='append'
    )
    parser.add_argument(
        "--alpha",
        help="Alpha value used to construct a hull (or hulls) around the provided hull vertices",
        type=arg_types["alpha"],
        default=0.0
    )
    parser.add_argument(
        "--delta",
        help="Maximum distance outside of target polygon to consider as target",
        type=arg_types["delta"],
        default=1.0
    )
    parser.add_argument(
        "--remove",
        help="Set remove size (px)",
        default=0,
        type=arg_types["remove"]
    )
    parser.add_argument(
        "--fill",
        help="Set fill size (px)",
        default=0,
        type=arg_types["fill"]
    )
    parser.add_argument(
        "--area_file",
        help="Disc area filename for analysis",
        type=arg_types["area_file"],
        default=None
    )
    parser.add_argument(
        "--circle_diameter",
        help="Diameter of surrounding circles in pixels",
        default=None,
        type=arg_types["circle_diameter"]
    )
    parser.add_argument(
        "--circle_variability",
        help="Variability of circle diameter for detection",
        default=0.0,
        type=arg_types["circle_variability"]
    )
    parser.add_argument(
        "--circle_expansion",
        help="Optional expansion factor for circles (increases radius to search, circles must not overlap)",
        default=0.1,
        type=arg_types["circle_expansion"]
    )
    parser.add_argument(
        "--circle_separation",
        help="Distance between edges of circles within a plate (px)",
        default=None,
        type=arg_types["circle_separation"]
    )
    parser.add_argument(
        "--circle_separation_tolerance",
        help="How much tolerance to allow for distances between circles within a plate",
        default=0.1,
        type=arg_types["circle_separation_tolerance"]
    )
    parser.add_argument(
        "--canny_sigma",
        help="Standard deviation of the Gaussian filter in Canny edge detection",
        default=3,
        type=arg_types["canny_sigma"]
    )
    parser.add_argument(
        "--canny_low",
        help="Lower bound for edge strength, passed to skimage canny edge detection",
        default=None,
        type=arg_types["canny_low"]
    )
    parser.add_argument(
        "--canny_high",
        help="Upper bound for edge strength, passed to skimage canny edge detection",
        default=None,
        type=arg_types["canny_high"]
    )
    parser.add_argument(
        "--plate_width",
        help="Length of shortest edge of plate (px)",
        default=None,
        type=arg_types["plate_width"]
    )
    parser.add_argument(
        "--circles",
        help="The total number of circles to detect",
        default=None,
        type=arg_types["circles"]
    )
    parser.add_argument(
        "--plates",
        help="In plate layout, the number of plates per image",
        default=None,
        type=arg_types["plates"]
    )
    parser.add_argument(
        "--circles_per_plate",
        help="In plate clustering, the number of circles per plate (multiple may be specified)",
        default=None,
        type=arg_types["circles_per_plate"],
        action='append'
    )
    parser.add_argument(
        "--plates_cols_first",
        help="In plate ID layout, increment by columns first",
        action='store_true',
    )
    parser.add_argument(
        "--plates_right_left",
        help="In plate ID layout, increment from right to left",
        action='store_true',
    )
    parser.add_argument(
        "--plates_bottom_top",
        help="In plate ID layout, increment from bottom to top",
        action='store_true'
    )
    parser.add_argument(
        "--circles_cols_first",
        help="In circle ID layout, increment by columns first",
        action='store_true'
    )
    parser.add_argument(
        "--circles_right_left",
        help="In circle ID layout, increment right to left",
        action='store_true',
    )
    parser.add_argument(
        "--circles_bottom_top",
        help="In circle ID layout, increment from bottom to top)",
        action='store_true'
    )
    parser.add_argument(
        "--scale",
        help="pixels/unit distance for area calculation (if unit distance is mm then area will be reported in mm²)",
        default=None,
        type=arg_types["scale"]
    )
    parser.add_argument(
        "--fit_start",
        help="Start (day) for RGR calculation",
        type=arg_types["fit_start"],
        default=float(0)
    )
    parser.add_argument(
        "--fit_end",
        help="End (day) for RGR calculation",
        type=arg_types["fit_end"],
        default=float('inf')
    )
    # Debugging arguments
    parser.add_argument(
        "--loglevel",
        help="Log-level",
        type=arg_types['loglevel'],
        default="INFO"
    )
    # GUI only options
    parser.add_argument(
        "--voxel_size",
        help="Size of voxel in colourspace projection to display a point during hull calibration",
        type=arg_types["voxel_size"],
        default=1
    )
    # Poorly supported arguments
    parser.add_argument(  # todo parameter isn't handled well, need to scale all, e.g. diameter, scale etc. on both ends
        "--downscale",
        help="Downscale by this factor",
        type=arg_types["downscale"],
        default=1
    )
    parser.add_argument(
        "--denoise",
        help="Denoise using bilateral filter",
        action='store_true'
    )
    return parser


# Processing that should happen after parsing input arguments
def postprocess(args):
    # Handle multiple directories or a mix of directories and files
    if args.images is not None:
        if isinstance(args.images, list):
            args.images = list(chain(*args.images))
        else:
            args.images = list(chain(args.images))
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


class TemporaryArgsAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['temporary_args'], msg), kwargs


def update_arg(args, arg, val, temporary=False):
    # We need to distinguish updates to a temporary copy of args vs the main args
    if temporary:
        adapter = TemporaryArgsAdapter(logger, {'temporary_args': "Temporary"})
        # only want to log this at debug level
    else:
        adapter = logger

    # flatten tuples and lists thereof used for e.g. colours into strings
    if isinstance(val, list):
        if not val:
            val_str = ""
            val = None  # empty list coerced back to None
        else:
            if isinstance(val[0], tuple):
                val_str = f'{[",".join([str(j) for j in i]) for i in val]}'.replace("'", '"')
            else:
                val_str = f'{",".join([str(i) for i in val])}'.replace("'", '"')

            # coerce to known type:
            try:
                if arg_types[arg] == image_path:  # this arg parser returns a list of paths
                    val = arg_types[arg](val)
                else:
                    val = [arg_types[arg](v) for v in val]
            except ValueError:
                adapter.debug("Issue with updating arg, could not be coerced to defined type")
                raise
    else:
        if isinstance(val, tuple):
            val_str = f"\"{','.join([str(i) for i in val])}\""
        else:
            val_str = str(val)
            # coerce to known type:
        try:
            val = arg_types[arg](val)
        except ValueError:
            adapter.debug("Issue with updating arg, could not be coerced to defined type")
            raise

    # now update
    if vars(args)[arg] is None:
        if temporary:
            adapter.debug(f"Setting {arg}: {val_str}")
        else:
            adapter.info(f"Setting {arg}: {val_str}")
        vars(args).update({arg: val})
        # adapter.debug(f"{arg}:{vars(args)[arg]}")
    else:
        if vars(args)[arg] == val:
            adapter.debug(f"Existing value matches the update so no change will be made {arg}: {val}")
        else:
            if temporary:
                adapter.debug(f"Overwriting configured value for {arg}: {vars(args)[arg]} will be set to {val}")
            else:
                adapter.info(f"Overwriting configured value for {arg}: {vars(args)[arg]} will be set to {val}")
            vars(args).update({arg: val})


def minimum_calibration(args):
    return all([
        args.images is not None and len(args.images) >= 1,
        (args.hull_vertices is not None and len(args.hull_vertices) >= 4)
    ])


def layout_defined(args):
    return args.fixed_layout is not None or all([
        args.circle_colour is not None,
        args.circle_diameter is not None and args.circle_diameter > 0,
        args.circle_variability is not None and args.circle_variability >= 0,
        args.circle_separation is not None and args.circle_separation >= 0,
        args.plate_width is not None and args.plate_width > 0,
        args.circles is not None,
        args.circles_per_plate is not None,
        args.plates is not None and args.plates > 0,
        args.circle_expansion is not None,
        args.circle_separation_tolerance is not None,
        args.circles_cols_first is not None,
        args.circles_right_left is not None,
        args.circles_bottom_top is not None,
        args.plates_cols_first is not None,
        args.plates_bottom_top is not None,
        args.plates_right_left is not None
    ])


def configuration_complete(args):
    if args.fixed_layout or not args.detect_layout:
        return minimum_calibration(args)
    else:
        return all([
            (args.hull_vertices is not None and len(args.hull_vertices) >= 4),  # todo this can fail when alpha is set
            layout_defined(args)
        ])
