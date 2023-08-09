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
        "-i", "--images",
        help="Input image file or directory",
        type=arg_types["images"],
        default=None,
        action='append')
    parser.add_argument(
        "-s", "--samples",
        help="Input csv file with sample identities (block, unit, group)",
        type=arg_types["samples"],
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
        "-l",
        "--fixed_layout",
        help="Path to a plate layout definition file (generated during calibration by setting fixed layout)",
        default=None,
        type=arg_types["fixed_layout"]
    )
    parser.add_argument(
        "-w", "--whole_image",
        help="Run without layout definition to calculate target area for the entire image",
        action='store_true'
    )
    parser.add_argument(
        "-p", "--processes",
        help="Number of processes to launch (images to concurrently process)",
        default=1,
        type=arg_types["processes"]
    )
    parser.add_argument(
        "-d", "--image_debug",
        help="Level of image debugging",
        type=arg_types['image_debug'],
        default="INFO"
    )
    parser.add_argument(
        "--loglevel",
        help="Log-level",
        type=arg_types['loglevel'],
        default="INFO"
    )
    parser.add_argument(
        "--time_regex",
        help="Regex pattern to identify date time string from filename",
        type=arg_types["time_regex"]
    )
    parser.add_argument(
        "--time_format",
        help="String format pattern to read datetime object from regex match",
        type=arg_types["time_format"]
    )
    parser.add_argument(
        "--block_regex",
        help="Regex pattern to identify block string from filename",
        type=arg_types["block_regex"]
    )
    parser.add_argument(
        "--animations",
        help="Use imagemagick to generate gif animations of 3d plots",
        action='store_true'
    )
    parser.add_argument(
        "--superpixels",
        help="The number of superpixels to find in image",
        type=arg_types["superpixels"],
        default=None
    )
    parser.add_argument(
        "--slic_iter",
        help="The maximum number of iterations for SLIC clustering",
        type=arg_types["slic_iter"],
        default=10
    )
    parser.add_argument(
        "--compactness",
        help=(
            "Superpixel compactness, higher values is more more weight to distance than colour."
            " Positive values will be fixed, negative values will run run slic_zero"
            " (a locally adaptive compactness parameter)"
            " with the absolute of the provided value being the starting point"
        ),
        type=arg_types["compactness"],
        default=-.1
    )
    parser.add_argument(
        "--sigma",
        help="Smoothing kernel applied before superpixel clustering",
        type=arg_types["sigma"],
        default=1
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
        default=0
    )
    parser.add_argument(
        "--delta",
        help="Maximum distance outside of target polygon to consider as target",
        type=arg_types["delta"],
        default=5
    )
    parser.add_argument(
        "--force_calibration",
        help="Force calibration window to load, even if all parameters are defined",
        action='store_true'
    )
    parser.add_argument(
        "--num_calibration",
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
        "--downscale",
        help="Downscale by this factor",
        type=arg_types["downscale"],
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
        "--area_file",
        help="Disc area filename for analysis (must be in the output directory)",
        type=arg_types["area_file"],
        default="area.csv"
    )

    parser.add_argument(
        "--circle_diameter",
        help="Diameter of surrounding circles in pixels",
        default=None,
        type=arg_types["circle_diameter"]
    )
    parser.add_argument(
        "--circle_expansion",
        help="Optional expansion factor for circles (increases radius to search, circles must not overlap)",
        default=1,
        type=arg_types["circle_expansion"]
    )
    parser.add_argument(
        "--plate_circle_separation",
        help="Distance between edges of circles within a plate (px)",
        default=None,
        type=arg_types["plate_circle_separation"]
    )
    parser.add_argument(
        "--plate_cut_expansion",
        help="How much tolerance to allow for distances between circles within a plate",
        default=1.1,
        type=arg_types["plate_cut_expansion"]
    )
    parser.add_argument(
        "--plate_width",
        help="Length of shortest edge of plate (px)",
        default=None,
        type=arg_types["plate_width"]
    )
    parser.add_argument(
        "--circles_per_plate",
        help="In plate clustering, the number of circles per plate",
        default=None,
        type=arg_types["circles_per_plate"]
    )
    parser.add_argument(
        "--n_plates",
        help="In plate layout, the number of plates per image",
        default=None,
        type=arg_types["n_plates"]
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
