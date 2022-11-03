from configargparse import ArgumentParser
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Parse command-line arguments
def options():
    config_dir = Path(Path(__file__).parent.parent, "conf.d")
    config_files = config_dir.glob("*.conf")
    parser = ArgumentParser(
        default_config_files=config_files,
    )
    parser.add_argument("-i", "--image", help="Input image file or directory", default=None)
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
        "-p",
        "--processes",
        help="Number of processes to launch (images to concurrently process)",
        default=1,
        type=int
    )
    parser.add_argument(
        "-D",
        "--image_debug",
        help=(
            "Writes out intermediate images for debugging/tuning: "
            "'print' to file or "
            "'plot' to device (requires matplotlib) "
        ),
        choices=["plot", "print"],
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
        "-tl", "--target_l",
        help="typical 'l' (lightness) value for target in Lab colourspace",
        type=float,
        default=None
    )
    parser.add_argument(
        "-ta", "--target_a",
        help="typical 'a' (green-magenta) value for target in Lab colourspace",
        type=float,
        default=None
    )
    parser.add_argument(
        "-tb", "--target_b",
        help="typical 'b' (blue-yellow) value for target in Lab colourspace",
        type=float,
        default=None
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
        "-cc",
        "--circle_channel",
        help="The channel from Lab colour-space to use for circle detection",
        choices=["a", "b"],
        default="b",
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
    parser.add_argument(
        "-k",
        "--kernel",
        help="Kernel for median blur, used in circle detection",
        default=20,
        type=int
    )
    parser.add_argument(
        "-sc",
        "--scale",
        help="pixels/unit distance for area calculation (if unit distance is mm then area will be reported in mm²)",
        default=5.625,  # 180px = 32mm
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
        "-ct",
        "--circle_diameter_tolerance",
        help="Circle diameter tolerance, tune for circle detection",
        default=0.05,
        type=float
    )
    parser.add_argument(
        "-ph",
        "--param2",
        help="Accumulator threshold for HoughCircles, if not defined then performs a search for optimum value",
        default=None,
        type=int
    )
    parser.add_argument(
        "-ch",
        "--cut_height_tolerance",
        help="Cut height tolerance, tune for plate clustering",
        default=0.35,
        type=float
    )
    parser.add_argument(
        "-cpp",
        "--circles_per_plate",
        help="In plate clustering, the number of circles per plate",
        default=6
    )
    parser.add_argument(
        "-npi",
        "--n_plates",
        help="In plate layout, the number of plates per image",
        default=8
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
    args = parser.parse_args()
    if not args.out_dir:
        if Path(args.image).is_file():
            args.out_dir = Path(args.image).parents[0]
        else:
            args.out_dir = Path(args.image)
    if args.kernel % 2 == 0:
        args.kernel = args.kernel + 1
    logger.info(parser.format_values())
    return args
