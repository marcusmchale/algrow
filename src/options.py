import argparse

from configargparse import ArgumentParser
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def lab(s: str):
    try:
        l, a, b = map(float, s.split(','))
        return l, a, b
    except:
        raise argparse.ArgumentTypeError(f'Colour must be a tuple of 3 float values : {s}')

# Parse command-line arguments
def options():
    config_dir = Path(Path(__file__).parent.parent, "conf.d")
    config_files = config_dir.glob("*.conf")
    parser = ArgumentParser(
        default_config_files=config_files,
    )
    parser.add_argument("-c", "--conf", help ="Config file path", is_config_file=True)
    parser.add_argument("-i", "--image", help="Input image file or directory", default=None, action='append')
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
        help = "Use imagemagick to generate gif animations of 3d plots",
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
        choices = ["save", "plot", "both"],
        default = None
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
        type=int,
        default=20
    )
    parser.add_argument(
        "-cc", "--circle_colour",
        help="Circle colour in Lab colourspace",
        type=lab,
        default=None
    )
    parser.add_argument(
        "-tc", "--target_colour",
        help="Target colour in Lab colourspace as comma separated integers. Multiple targets are accepted",
        type=lab,
        default=None,
        action='append'
    )
    parser.add_argument(
        "-ntc", "--non_target_colour",
        help="Non-target colour in Lab colourspace as comma separated integers. Multiple targets are accepted",
        type=lab,
        default=None,
        action='append'
    )
    parser.add_argument(
        "-td", "--target_dist",
        help="Maximum distance from target colour to consider a superpixel as the start of a target cluster",
        type=int,
        default=8
    )
    parser.add_argument(
        "-ntd", "--non_target_dist",
        help="Maximum distance from non-target colour to consider a superpixel as non-target",
        type=int,
        default=8
    )
    parser.add_argument(
        "-gd", "--graph_dist",
        help="Maximum colour distance between adjacent superpixels in a target cluster - set to 0 to avoid graph construction",
        type=int,
        default=8
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
