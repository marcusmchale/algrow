import argparse
import os.path

from plantcv import plantcv as pcv
import numpy as np
import cv2 as cv
from scipy.spatial import distance
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt

# Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", default=None)
    parser.add_argument(
        "-c",
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
    parser.add_argument(
        "-v",
        "--value",
        help="Set threshold value for value channel in HSV",
        default=10,
        type=float
    )
    parser.add_argument(
        "-b",
        "--blue_yellow",
        help="Set threshold value for blue-yellow channel in LAB",
        default=100,
        type=float
    )
    parser.add_argument("-f", "--fill", help="Set fill size", default=1000, type=float)
    parser.add_argument(
        "-D",
        "--debug",
        help="Writes out intermediate images: None, 'print' (to file) or 'plot' (to device)",
        choices=[None, "print", "plot"],
        default=None
    )
    args: argparse.Namespace = parser.parse_args()
    if not args.outdir:
        args.outdir = os.path.dirname(args.image)
    return args


# Extract black and white image of mask
def get_mask():
    # Get options
    args = options()
    saturation = args.saturation
    value = args.value
    blue_yellow = args.blue_yellow
    fill = args.fill

    # Read image
    img, path, filename = pcv.readimage(args.image)

    # Find blue circles to define areas

    # Convert RGB to LAB and extract the Blue-Yellow channel
    b = pcv.rgb2gray_lab(img, 'b')
    # Threshold the blue-yellow image
    #b = pcv.threshold.binary(b, blue_yellow, 255, 'light')
    b = pcv.median_blur(b, 5)
    #pcv.plot_image(b_thresh)
    circles = cv.HoughCircles(b, cv.HOUGH_GRADIENT, dp=1, minDist=170, param1=20, param2=30, minRadius=70, maxRadius=120)
    # 170 px is the diameter of a blue circle
    circles = np.squeeze(np.uint16(np.around(circles)))
    centres = np.delete(circles, 2, axis=1)


    # First clustering to find the plates
    distances = distance.pdist(centres)
    dendrogram = hierarchy.linkage(centres)
    fig = plt.figure()
    dn = hierarchy.dendrogram(dendrogram)
    plt.show()
    clusters = hierarchy.cut_tree(dendrogram, height=230)
    # the cut height is dependent on the scale (pixels between centre of blue rings)
    # consider asking for input of scale to allow more variable inputs
    # we will need for the later quant steps anyway
    unique, counts = np.array(np.unique(clusters, return_counts=True))
    target_clusters = unique[[i for i, j in enumerate(counts.flat) if j == 6]]
    target_circles = circles[[i for i, j in enumerate(clusters.flat) if j in target_clusters]]
    for i in target_circles:
        # draw the outer circle
        cv.circle(b, (i[0], i[1]), i[2], (255, 0, 0), 5)
        # draw the center of the circle
        cv.circle(b, (i[0], i[1]), 2, (0, 0, 255), 5)
    pcv.plot_image(b)

    class Plate:
        def __init__(self, cluster, cluster_circles):
            self.cluster: int = cluster
            self.circles = cluster_circles
            self.centroid = tuple(np.uint16(self.circles[:, 0:2].mean(axis=0)))
            x_range = np.max(self.circles[:, 0]) - np.min(self.circles[:, 0])
            y_range = np.max(self.circles[:, 1]) - np.min(self.circles[:, 1])
            self.vertical = y_range > x_range
            self.number = None

    plates = [
            Plate(
                t,
                circles[[i for i, j in enumerate(clusters.flat) if j == t]],
            ) for t in target_clusters
    ]
    plates.sort(key=lambda x: (not x.vertical, x.centroid[0]))
    for i, p in enumerate(plates):
        p.number = i + 1
        cv.putText(b, str(p.number), p.centroid, 0, 5, (0, 0, 255), 5)
        if p.vertical:
            # first split the plate into left and right
            p.circles = p.circles[p.circles[:,0].argsort()][::-1]
            right = p.circles[0:3]
            right = right[right[:, 1].argsort()][::-1]
            left = p.circles[3:6]
            left = left[left[:, 1].argsort()][::-1]
            p.circles = np.concatenate((left, right))
        else:
            # first split the plate into left, middle and right
            p.circles = p.circles[p.circles[:,0].argsort()][::-1]
            right = p.circles[0:2]
            right = right[right[:, 1].argsort()][::-1]
            middle = p.circles[2:4]
            middle = middle[middle[:, 1].argsort()][::-1]
            left = p.circles[4:6]
            left = left[left[:, 1].argsort()][::-1]
            p.circles = np.concatenate((left, middle, right))
        for j, c in enumerate(p.circles):
            cv.putText(b, str(j+1+6*i), c[0:2], 0, 3, (0, 0, 255), 5)
    pcv.plot_image(b)
    import pdb; pdb.set_trace()

        # Add a cluster label to the image











    import pdb; pdb.set_trace()



    # Convert RGB to HSV and extract the saturation channel and value channel
    s = pcv.rgb2gray_hsv(img, 's')
    v = pcv.rgb2gray_hsv(img, 'v')


    # Threshold the saturation image
    # this gets most of the coloured objects (blue rings and leaf disks)
    s_thresh = pcv.threshold.binary(s, saturation, 255, 'light')
    # Threshold the value image
    # this catches the low exposure bits that are often folded or dark lamina disks
    v_thresh = pcv.threshold.binary(v, value, 10, 'dark')


    # Join v and s with an or statement
    vs = pcv.logical_or(s_thresh, v_thresh)
    vs = pcv.threshold.binary(vs, 0,255, 'light')
    # Join vs and b with and statement
    vsb = pcv.logical_and(vs, b_thresh)
    vsb_fill = pcv.fill(vsb, fill)
    vsb_fill2 = pcv.fill_holes(vsb_fill)
    masked = pcv.apply_mask(img, vsb, 'white')
    overlay = pcv.visualize.overlay_two_imgs(img, vsb_fill2, alpha=0.5)
    # output file
    if args.mask in ["both", "masked"]:
        overlay_file = os.path.join(args.outdir, "masked_" + filename)
        pcv.print_image(overlay, overlay_file)
    if args.mask in ["both", "mask"]:
        mask_file = os.path.join(args.outdir, "mask_" + filename)
        pcv.print_image(pcv.invert(vsb_fill2), mask_file)


# Extract black and white image of mask
def get_channels():
    args = options()
    img, path, filename = pcv.readimage(args.image)
    filename = os.path.basename(args.image)
    pcv.print_image(pcv.rgb2gray_hsv(img, 's'), args.outdir + "/" + filename + "_saturation.jpg")
    pcv.print_image(pcv.rgb2gray_hsv(img, 'v'), args.outdir + "/" + filename + "_value.jpg")
    pcv.print_image(pcv.rgb2gray_hsv(img, 'h'), args.outdir + "/" + filename + "_hue.jpg")
    pcv.print_image(pcv.rgb2gray_lab(img, 'a'), args.outdir + "/" + filename + "_green_magenta.jpg")
    pcv.print_image(pcv.rgb2gray_lab(img, 'b'), args.outdir + "/" + filename + "_blue_yellow.jpg")


def main():
    args = options()
    pcv.params.debug = args.debug

    if args.action == "threshold":
        get_mask()
    else:
        get_channels()


main()

