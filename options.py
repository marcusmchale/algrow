from argparse import ArgumentParser
from pathlib import Path
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argparse import Namespace


# Parse command-line arguments
def options():
    parser = ArgumentParser(description="Imaging processing")
    parser.add_argument("-i", "--image", help="Input image file or directory", required=True)
    parser.add_argument("-o", "--out_dir", help="Output directory for image files.", default=None)
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
        "--debug",
        help=(
            "Writes out intermediate images: "
            "'print' to file or "
            "'plot' to device (requires matplotlib) "
        ),
        choices=["plot", "print"],
        default=None
    )
    parser.add_argument(
        "-s",
        "--saturation",
        help="Set threshold value for saturation channel in HSV",
        default=150,
        type=int
    )
    parser.add_argument(
        "-v",
        "--value",
        help="Set threshold value for value channel in HSV",
        default=50,
        type=int
    )
    parser.add_argument(
        "-a",
        "--green_red",
        help="Set threshold value for green-red channel (A in LAB)",
        default=130,
        type=int
    )
    parser.add_argument(
        "-b",
        "--blue_yellow",
        help="Set threshold value for blue-yellow channel (B in LAB)",
        default=140,
        type=int
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
        help="pixels/mm",
        default=5.625,  # 180px = 32mm
        type=float
    )
    parser.add_argument(
        "-cd",
        "--circle_diameter",
        help="Diameter of blue circles in mm",
        default=32,
        type=float
    )
    parser.add_argument(
        "-ct",
        "--circle_radius_tolerance",
        help="Circle radius tolerance, tune for circle detection",
        default=0.05,
        type=float
    )
    parser.add_argument(
        "-ch",
        "--cut_height_tolerance",
        help="Cut height tolerance, tune for plate clustering",
        default=0.3,
        type=float
    )
    args: Namespace = parser.parse_args()
    if not args.out_dir:
        if Path(args.image).is_file():
            args.out_dir = Path(args.image).parents[0]
        else:
            args.out_dir = Path(args.image)
    if args.kernel % 2 == 0:
        args.kernel = args.kernel + 1
    return args
