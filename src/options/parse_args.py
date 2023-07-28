from configargparse import ArgumentParser
from pathlib import Path
from itertools import chain

from .update_and_verify import arg_types


# Parse command-line arguments
def options():
    config_dir = Path(Path(__file__).parent.parent.parent, "conf.d")
    config_files = config_dir.glob("*.conf")
    parser = ArgumentParser(
        default_config_files=[str(i) for i in config_files],
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
        default="algrow_output",
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
        "--image_debug",
        help="Level of image debugging",
        type=arg_types['image_debug'],
        default="INFO"
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        help="Log-level",
        type=arg_types['loglevel'],
        default="INFO"
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
        default=5
    )
    parser.add_argument(
        "-nc", "--num_calibration",
        help="Number of images to use for calibration",
        type=arg_types["num_calibration"],
        default=3
    )
    parser.add_argument(
        "-b",
        "--blur",
        help="Gaussian blur applied to image during loading",
        type=arg_types["blur"],
        default=1
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
        "-w",
        "--whole_image",
        help="Run without layout definition to calculate target area for the entire image",
        action='store_true'
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
        "-pce",
        "--plate_cut_expansion",
        help="How much tolerance to allow for distances between circles within a plate",
        default=1.1,
        type=arg_types["plate_cut_expansion"]
    )
    parser.add_argument(
        "-pw",
        "--plate_width",
        help="Length of shortest edge of plate (px)",
        default=None,
        type=arg_types["plate_width"]
    )
    parser.add_argument(
        "-cp",
        "--circles_per_plate",
        help="In plate clustering, the number of circles per plate",
        default=None,
        type=arg_types["circles_per_plate"]
    )
    parser.add_argument(
        "-np",
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


# Processing that should happen after parsing input arguments
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
