import argparse
from pathlib import Path
import csv
from plantcv import plantcv as pcv
import numpy as np
import cv2 as cv
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt
from copy import copy
from concurrent.futures import ProcessPoolExecutor, as_completed


# Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing")
    parser.add_argument("-i", "--image", help="Input image file or directory", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", default=None)
    parser.add_argument(
        "-c",
        "--action",
        help="Get mask from thresholding or print channels to determine appropriate threshold values",
        choices=["area", "channels"],
        default="area"
    )
    parser.add_argument(
        "-m",
        "--mask",
        help="Write out image mask, overlay or both",
        choices=[None, "mask", "overlay", "both"],
        default=None
    )
    parser.add_argument(
        "-s",
        "--saturation",
        help="Set threshold value for saturation channel in HSV",
        default=100,
        type=int
    )
    parser.add_argument(
        "-v",
        "--value",
        help="Set threshold value for value channel in HSV",
        default=10,
        type=int
    )
    parser.add_argument(
        "-b",
        "--blue_yellow",
        help="Set threshold value for blue-yellow channel in LAB",
        default=130,
        type=int
    )
    parser.add_argument("-f", "--fill", help="Set fill size", default=1000, type=int)
    parser.add_argument(
        "-d",
        "--cut_height",
        help="Dendrogram cut height for circle clustering to plates",
        default=230,
        type=int
    )
    parser.add_argument(
        "-D",
        "--debug",
        help=(
            "Writes out intermediate images: "
            "None, "
            "'print' (to file) , "
            "'plot' (to device), "
            "'log' (print comments)"
        ),
        choices=[None, "print", "plot", "log"],
        default=None
    )
    parser.add_argument(
        "-p",
        "--processes",
        help="Number of processes to launch (images to concurrently process)",
        default=1,
        type=int
    )
    args: argparse.Namespace = parser.parse_args()
    if not args.outdir:
        if Path(args.image).is_file():
            args.outdir = Path(args.image).parents[0]
        else:
            args.outdir = Path(args.image)
    return args


# Extract black and white image of mask
def get_area(image):
    # Get options
    args = options()
    saturation = args.saturation
    value = args.value
    blue_yellow = args.blue_yellow
    fill = args.fill
    cut_height = args.cut_height

    # Read image
    img, path, filename = pcv.readimage(str(image))
    basename = Path(image).stem
    # Find blue circles to define areas

    # Convert RGB to LAB and extract the Blue-Yellow channel
    b = pcv.rgb2gray_lab(img, 'b')
    # Convert RGB to HSV and extract the saturation channel and value channel
    s = pcv.rgb2gray_hsv(img, 's')
    v = pcv.rgb2gray_hsv(img, 'v')

    # Threshold the saturation image
    # this gets most of the coloured objects (blue rings and leaf disks)
    s_thresh = pcv.threshold.binary(s, saturation, 255, 'light')
    # Threshold the value image
    # this catches the low exposure bits that are often folded or dark lamina disks
    v_thresh = pcv.threshold.binary(v, value, 10, 'dark')
    # and we still need to threshold the b image
    b_thresh = pcv.threshold.binary(b, blue_yellow, 255, 'light')
    # Join s and b with and statement
    sb = pcv.logical_and(s_thresh, b_thresh)
    # Join v and sb with or statement (all low value and all saturated yellowish)
    vsb = pcv.logical_or(sb, v_thresh)
    vsb = pcv.threshold.binary(vsb, 0, 255, 'light')
    vsb_fill = pcv.fill(vsb, fill)
    mask = pcv.fill_holes(vsb_fill)
    overlay = pcv.visualize.overlay_two_imgs(img, mask, alpha=0.5)
    # output file
    if args.mask in ["both", "masked"]:
        overlay_file = Path(args.outdir, "masked_" + filename)
        pcv.print_image(overlay, str(overlay_file))
    if args.mask in ["both", "mask"]:
        mask_file = Path(args.outdir, "mask_" + filename)
        pcv.print_image(pcv.invert(mask), str(mask_file))

    # Find circles for numbering
    b_blur = pcv.median_blur(b, 20)  # raise this to  get rid of spurious circles
    # pcv.plot_image(b_thresh)
    circles = cv.HoughCircles(b_blur, cv.HOUGH_GRADIENT, dp=1, minDist=170, param1=20, param2=30, minRadius=70, maxRadius=120)
    # 170 px is the diameter of a blue circle
    # todo consider adjusting this according to input scale
    try:
        circles = np.squeeze(np.uint16(np.around(circles)))
    except TypeError:
        if args.debug:
            print(filename + ': No circles found')
        return [(None, None, None)]
    if circles.shape[0] < 48:
        if args.debug:
            print(filename + ': Insufficient circles found, expect 48')
        return [(None, None, None)]
    centres = np.delete(circles, 2, axis=1)

    # First clustering to find the plates
    dendrogram = hierarchy.linkage(centres)

    if args.debug:
        if args.debug == 'log':
            pass
        else:
            fig = plt.figure()
            dn = hierarchy.dendrogram(dendrogram)
            plt.axhline(y=cut_height, c='k')
            if args.debug == 'plot':
                plt.show()
            elif args.debug == 'print':
                plt.savefig(Path(args.outdir, basename + "dendrogram.png"))

    clusters = hierarchy.cut_tree(dendrogram, height=cut_height)
    # the cut height is dependent on the scale (pixels between centre of blue rings)
    # todo consider asking for input of scale to allow more variable inputs
    # we will need for the later quant steps anyway
    unique, counts = np.array(np.unique(clusters, return_counts=True))
    target_clusters = unique[[i for i, j in enumerate(counts.flat) if j >= 6]]
    if len(target_clusters) < 8:
        return [(None, None, None)]
    if args.debug:
        print(filename + ': ' + str(len(target_clusters)) + ' plates found')
    target_circles = circles[[i for i, j in enumerate(clusters.flat) if j in target_clusters]]

    img_labeled = copy(img)
    for i in target_circles:
        # draw the outer circle
        cv.circle(img_labeled, (i[0], i[1]), i[2], (255, 0, 0), 5)
        # draw the center of the circle
        cv.circle(img_labeled, (i[0], i[1]), 2, (0, 0, 255), 5)
    if args.debug:
        if args.debug == 'plot':
            pcv.plot_image(img_labeled)
        elif args.debug == 'print':
            pcv.print_image(img_labeled, Path(args.outdir, basename + "circles.png"))

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
    result = []
    for i, p in enumerate(plates):
        p.number = i + 1
        cv.putText(img_labeled, str(p.number), p.centroid, 0, 5, (255, 0, 255), 5)
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
            circle_number = j+1+6*i
            cv.putText(img_labeled, str(circle_number), c[0:2], 0, 3, (0, 255, 255), 5)
            # count pixels that are in the mask inside the circle
            circle_image = np.zeros_like(mask)
            cv.circle(circle_image, (c[0],c[1]), c[2], (255,255,255), -1)
            if args.debug:
                if args.debug == 'plot':
                    pcv.plot_image(circle_image)
                elif args.debug == 'print':
                    pcv.print_image(circle_image, Path(args.outdir, basename + "circle" + circle_number + ".png"))
            circle_mask = cv.bitwise_and(mask, circle_image)
            pixels = cv.countNonZero(circle_mask)
            result.append((p.number, circle_number, pixels))
    if args.debug:
        if args.debug == 'plot':
            pcv.plot_image(img_labeled)
        elif args.debug == 'print':
            pcv.print_image(img_labeled, Path(args.outdir, basename + "circles_labeled.png"))
    return str(image), result



# Extract black and white image of mask
def get_channels():
    args = options()
    img, path, filename = pcv.readimage(args.image)
    filename = Path(args.image)
    pcv.print_image(pcv.rgb2gray_hsv(img, 's'), args.outdir + "/" + filename + "_saturation.jpg")
    pcv.print_image(pcv.rgb2gray_hsv(img, 'v'), args.outdir + "/" + filename + "_value.jpg")
    pcv.print_image(pcv.rgb2gray_hsv(img, 'h'), args.outdir + "/" + filename + "_hue.jpg")
    pcv.print_image(pcv.rgb2gray_lab(img, 'a'), args.outdir + "/" + filename + "_green_magenta.jpg")
    pcv.print_image(pcv.rgb2gray_lab(img, 'b'), args.outdir + "/" + filename + "_blue_yellow.jpg")


def main():
    args = options()
    pcv.params.debug = args.debug

    # currently hard coded scale to, allow adjustment
    # 30 mm = 170 px
    # ~0.1765 mm = 1px
    # ~0.031 mm^2 / px
    scale = 0.031  # mm^2 / px

    if args.action == "channels":
        get_channels()
    elif args.action == "area":
        header = ['filename', 'plate', 'well', 'pixels', 'mm²']
        if Path(args.image).is_file():
            p = {Path(args.image)}
        elif Path(args.image).is_dir():
            p = set(Path(args.image).glob('**/*.jpg'))
        else:
            raise FileNotFoundError
        out_path = Path(args.outdir, "area.csv")
        if out_path.is_file():  # if the file exists then check for any already processed images
            with open(out_path) as csv_file:
                reader = csv.reader(csv_file)
                next(reader)
                files_done = {Path(row[0]) for row in reader}
            p = p - files_done
            if args.debug:
                print('Some images already included in result file, skipping these:', [str(f) for f in files_done])

        results = dict()
        with ProcessPoolExecutor(max_workers = args.processes) as executor:

            with open(out_path, 'a+') as csv_file:
                writer = csv.writer(csv_file)
                if out_path.stat().st_size == 0:  # True if empty
                    writer.writerow(header)

                future_to_file = {executor.submit(get_area, fp): fp for fp in p}
                for future in as_completed(future_to_file):
                    fp = future_to_file[future]
                    try:
                        result = future.result()
                        for r in result[1]:
                            writer.writerow([result[0], r[0], r[1], r[2], None if r[2] is None else r[2] * scale])
                    except Exception as exc:
                        print('%r generated an exception: %s' % (str(fp), exc))
                    else:
                        if args.debug:
                            print('%r processed' % (str(fp)))


if __name__ == '__main__':
    main()

