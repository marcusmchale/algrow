import logging

from pathlib import Path

import numpy as np
import cv2 as cv2
from scipy.cluster import hierarchy
from matplotlib import pyplot, rcParams
from copy import copy

#  rcParams['toolbar'] = 'None'


class ImageContentException(Exception):
    pass


class Plate:
    def __init__(self, cluster_id, circles):
        self.cluster_id: int = cluster_id
        self.circles = circles
        self.centroid = tuple(np.uint16(self.circles[:, 0:2].mean(axis=0)))
        x_range = np.max(self.circles[:, 0]) - np.min(self.circles[:, 0])
        y_range = np.max(self.circles[:, 1]) - np.min(self.circles[:, 1])
        self.vertical = y_range > x_range
        self.plate_id = None


class ImageProcessor:
    def __init__(self, filepath, args):
        self.args = args
        self.filepath = Path(filepath)
        logging.debug(f"Load image as RGB: {self.filepath}")
        self.rgb = cv2.imread(str(self.filepath))
        self.debug_image(self.rgb, f"Raw: {self.filepath}")
        if args.debug:
            assert self.rgb.dtype == 'uint8'
        logging.debug(f"Convert image to LAB colour space, split and keep Blue-Yellow")
        l, a, b = cv2.split(cv2.cvtColor(self.rgb, cv2.COLOR_RGB2Lab))
        self.b = b
        self.debug_image(b, f"Blue-Yellow channel")

    def debug_image(self, img, label: str, prefix="debug", extension=".jpg"):
        if self.args.debug:
            if self.args.debug == 'print':
                prefix = "_".join([i for i in (prefix, label) if i])
                filepath = self.filepath.with_stem(f'{prefix}_{self.filepath.stem}').with_suffix(extension)
                cv2.imwrite(Path(self.args.out_dir, filepath), img)
            elif self.args.debug == 'plot':
                rescale = 0.2
                width = int(img.shape[1] * rescale)
                height = int(img.shape[0] * rescale)
                dim = (width, height)
                small_img = cv2.resize(img, dim)
                cv2.imshow(label, small_img)
                cv2.waitKey()
                cv2.destroyWindow(label)

    def debug_plot(self, plot, label: str, prefix="debug", extension=".jpg"):
        if self.args.debug:
            if self.args.debug == 'print':
                prefix = "_".join([i for i in (prefix, label) if i])
                filepath = self.filepath.with_stem(f'{prefix}_{self.filepath.stem}').with_suffix(extension)
                plot.savefig(Path(self.args.out_dir, filepath))
            elif self.args.debug == 'plot':
                plot.show()

    def find_circles(self):
        args = self.args
        logging.debug("Find the blue rings in image")
        logging.debug("Blur blue-yellow greyscale image")
        b_blur = cv2.medianBlur(self.b, args.kernel)
        self.debug_image(b_blur, "Blue-Yellow median blur")
        logging.debug("Find circles in blurred image")
        circle_radius_px = int((args.scale * args.circle_diameter)/2)
        circle_tolerance = 0.2
        circles = cv2.HoughCircles(
            b_blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(circle_radius_px*2),
            param1=20,
            param2=30,
            minRadius=int(circle_radius_px * (1-circle_tolerance)),
            maxRadius=int(circle_radius_px * (1+circle_tolerance))
        )
        try:
            circles = np.squeeze(np.uint16(np.around(circles)))
        except TypeError:
            raise ImageContentException('No circles found')
        if circles.shape[0] < 48:
            raise ImageContentException(f'Insufficient circles found, expect 48 and found {str(circles.shape[0])}')
        return circles

    def find_plates(self):
        args = self.args
        circles = self.find_circles()
        logging.debug("Find clusters of circles to identify plates")
        logging.debug("Get centres of circles")
        centres = np.delete(circles, 2, axis=1)
        cut_height = int(args.scale * args.circle_diameter * (1 + args.cut_height_tolerance))
        logging.debug("Create dendrogram of centre distances (linkage method)")
        dendrogram = hierarchy.linkage(centres)
        if args.debug:
            logging.debug("Output dendrogram and treecut height for plate clustering")
            logging.debug("Create empty figure")
            fig = pyplot.figure()
            logging.debug("Create dendrogram")
            hierarchy.dendrogram(dendrogram)
            logging.debug("Add cut-height line")
            pyplot.axhline(y=cut_height, c='k')
            self.debug_plot(pyplot, "dendrogram")
        logging.debug("Cut the dendrogram and select clusters containing 6 centre points only")
        clusters = hierarchy.cut_tree(dendrogram, height=cut_height)
        unique, counts = np.array(np.unique(clusters, return_counts=True))
        target_clusters = unique[[i for i, j in enumerate(counts.flat) if j >= 6]]
        if len(target_clusters) < 8:
            raise ImageContentException(
                f'Insufficient plates found, expect 8 and {str(len(target_clusters))} found:'
            )
        circles = circles[[i for i, j in enumerate(clusters.flat) if j in target_clusters]]
        plates = [
            Plate(
                cluster_id,
                circles[[i for i, j in enumerate(clusters.flat) if j == cluster_id]],
            ) for cluster_id in target_clusters
        ]
        return plates

    @staticmethod
    def sort_plates(plates):
        plates.sort(key=lambda x: (not x.vertical, x.centroid[0]))
        for i, p in enumerate(plates):
            p.plate_id = i + 1
            if p.vertical:
                # first split the plate into left and right
                p.circles = p.circles[p.circles[:, 0].argsort()][::-1]
                right = p.circles[0:3]
                right = right[right[:, 1].argsort()][::-1]
                left = p.circles[3:6]
                left = left[left[:, 1].argsort()][::-1]
                p.circles = np.concatenate((left, right))
            else:
                # first split the plate into left, middle and right
                p.circles = p.circles[p.circles[:, 0].argsort()][::-1]
                right = p.circles[0:2]
                right = right[right[:, 1].argsort()][::-1]
                middle = p.circles[2:4]
                middle = middle[middle[:, 1].argsort()][::-1]
                left = p.circles[4:6]
                left = left[left[:, 1].argsort()][::-1]
                p.circles = np.concatenate((left, middle, right))
        return plates

    def get_plates(self):
        return self.sort_plates(self.find_plates())

    def get_area(self):
        args = self.args
        logging.debug("Start thresholding to calculate area")
        logging.debug("Convert RGB to HSV and extract the hue, saturation and value channels")
        h, s, v = cv2.split(cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV))
        self.debug_image(h, "Hue")
        self.debug_image(s, "Saturation")
        self.debug_image(v, "Value")
        logging.debug("Threshold the saturation channel to select coloured objects")
        ret, s_thresh = cv2.threshold(s, args.saturation, 255, cv2.THRESH_BINARY)

        self.debug_image(s_thresh, f"Saturation threshold: {args.saturation}")
        logging.debug("Threshold the value image to select low exposure pixels")
        # These are often folded lamina disks within blue circles but includes e.g. marbles and pump outside)
        ret, v_thresh = cv2.threshold(v, args.value, 255, cv2.THRESH_BINARY_INV)
        self.debug_image(v_thresh, f"Value threshold: {args.saturation}")
        logging.debug("Join the thresholded images with logical or to create a mask")
        # i.e. keeping anything coloured or very dark

        vs = v_thresh | s_thresh  #cv2.bitwise_or(v_thresh, s_thresh)

        logging.debug("Threshold blue-yellow channel that was previously extracted to find blue rings")
        ret, b_thresh = cv2.threshold(self.b, args.blue_yellow, 255, cv2.THRESH_BINARY)
        logging.debug("Mask the blue rings in the existing mask (logical and)")
        vsb = cv2.bitwise_and(vs, b_thresh)
        if len(np.shape(vsb)) != 2 or len(np.unique(vsb)) != 2:
            raise ImageContentException("Image is not binary")
            # todo might need to convert to grayscale again e.g. pcv.threshold.binary(vsb, 0, 255, 'light')
        #  vsb_fill = pcv.fill(vsb, args.fill) #todo if want fill can use skimage.morphology remove_small_objects
        #  mask = pcv.fill_holes(vsb_fill)
        threshold_mask = vsb

        plates = self.get_plates()
        result = []
        empty_mask = np.zeros_like(threshold_mask)
        overlay_mask = copy(empty_mask)

        for p in plates:
            logging.debug(f"Processing plate {p.plate_id}")
            for j, c in enumerate(p.circles):
                circle_number = j+1+6*(p.plate_id-1)
                logging.debug(f"Processing circle {circle_number}")
                logging.debug("Draw circle mask")
                circle_mask = copy(empty_mask)
                cv2.circle(circle_mask, (c[0], c[1]), c[2], (255, 255, 255), -1)
                local_mask = circle_mask & threshold_mask
                pixels = cv2.countNonZero(local_mask)
                result.append((p.plate_id, circle_number, pixels, local_mask))
                if args.overlay:
                    overlay_mask = overlay_mask | local_mask

        if args.overlay:
            logging.debug("Prepare annotated overlay for QC")
            overlay = copy(self.rgb)
            overlay = cv2.addWeighted(overlay, 0.5, overlay_mask, 0.5, 0.2, 0)
            for p in plates:
                logging.debug(f"Annotate overlay with plate ID: {p.plate_id}")
                cv2.putText(overlay, str(p.plate_id), p.centroid, 0, 5, (255, 0, 255), 5)
                for j, c in enumerate(p.circles):
                    circle_number = j + 1 + 6 * p.plate_id
                    # draw the outer circle
                    cv2.circle(overlay, (c[0], c[1]), c[2], (255, 0, 0), 5)
                    # draw the center of the circle
                    cv2.circle(overlay, (c[0], c[1]), 2, (0, 0, 255), 5)
                    cv2.putText(overlay, str(circle_number), c[0:2], 0, 3, (0, 255, 255), 5)

            overlay_path = Path(args.outdir, "overlay", self.filepath.name)
            overlay_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(overlay_path, overlay)
        return str(self.filepath), result
