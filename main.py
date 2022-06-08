import argparse
import os.path
from argparse import Namespace

from plantcv import plantcv as pcv


# Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", default=None)
    parser.add_argument(
        "-a",
        "--action",
        help="Get mask from thresholding or print channels to determine appropriate threshold values",
        choices=["threshold", "channels"],
        default="threshold"
    )
    parser.add_argument(
        "-m",
        "--mask",
        help="Write out image mask, overlay or both",
        choices=["mask", "overlay", "both"],
        default="both"
    )
    parser.add_argument(
        "-s",
        "--saturation",
        help="Set threshold value for saturation channel in HSV",
        default=100,
        type=float
    )
    parser.add_argument("-b", "--blue", help="Set threshold value for blue channel in RGB", default=150, type=float)
    parser.add_argument("-f", "--fill", help="Set fill size", default=1000, type=float)
    parser.add_argument(
        "-D",
        "--debug",
        help="Writes out intermediate images: None, 'print' (to file) or 'plot' (to device)",
        default=None
    )
    args: Namespace = parser.parse_args()
    if not args.outdir:
        args.outdir = os.path.dirname(args.image)
    return args


# Extract black and white image of mask
def get_mask():
    # Get options
    args = options()

    saturation = args.saturation
    blue = args.blue
    fill = args.fill

    # Read image
    img, path, filename = pcv.readimage(args.image)
    # Convert RGB to HSV and extract the Saturation channel
    s = pcv.rgb2gray_hsv(img, 's')
    # Threshold the Saturation images
    s_thresh = pcv.threshold.binary(s, saturation, 255, 'light')
    # Convert RGB to LAB and extract the Blue channel
    b = pcv.rgb2gray_lab(img, 'b')
    # Threshold the blue image
    b_thresh = pcv.threshold.binary(b, blue, 255, 'light')
    # device, b_cnt = pcv.binary_threshold(b, 127, 255, 'light', device, debug)
    # Join the segmented saturation and blue-yellow images
    bs = pcv.logical_and(s_thresh, b_thresh)
    # Fill small objects
    b_fill = pcv.fill(bs, fill)
    # Apply Mask (for vis images, mask_color=white)
    overlay = pcv.visualize.overlay_two_imgs(img, b_fill, alpha=0.5)
    # output file
    if args.mask in ["both", "masked"]:
        overlay_file = os.path.join(args.outdir, "masked_" + filename)
        pcv.print_image(overlay, overlay_file)
    if args.mask in ["both", "mask"]:
        mask_file = os.path.join(args.outdir, "mask_" + filename)
        pcv.print_image(pcv.invert(b_fill), mask_file)


# Extract black and white image of mask
def get_channels():
    args = options()
    # Read image
    img, path, filename = pcv.readimage(args.image)
    # Convert RGB to HSV and extract the Saturation channel
    s = pcv.rgb2gray_hsv(img, 's')
    saturation_hsv_file = args.outdir + "/" + "saturation_" + filename
    pcv.print_image(s, saturation_hsv_file)
    # Convert RGB to LAB and extract the Blue channel
    b = pcv.rgb2gray_lab(img, 'b')
    blue_hsv_file = args.outdir + "/" + "blue_" + filename
    pcv.print_image(b, blue_hsv_file)


def main():
    args = options()
    pcv.params.debug = args.debug

    if args.action == "threshold":
        get_mask()
    else:
        get_channels()


main()

