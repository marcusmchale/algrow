import logging

from pathlib import Path

import numpy as np
import cv2 as cv2
from scipy.cluster import hierarchy
from matplotlib import pyplot
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.util import img_as_ubyte, img_as_bool
from copy import copy



class ImageContentException(Exception):
    pass


def area_worker(filepath, args):
    return ImageProcessor(filepath, args).get_area()


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
        logging.debug(f"Convert RGB to Lab and split")
        l, a, b = cv2.split(cv2.cvtColor(self.rgb, cv2.COLOR_RGB2Lab))
        self.a = a
        self.debug_image(a, f"Green-Red channel (a in Lab)")
        self.b = b
        self.debug_image(b, f"Blue-Yellow channel (b in Lab)")
        logging.debug("Convert RGB to HSV and split")
        h, s, v = cv2.split(cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV))
        self.s = s
        self.debug_image(s, "Saturation (S in HSV)")
        self.v = v
        self.debug_image(v, "Value (V in HSV)")

    def debug_image(self, img, label: str, prefix="debug", extension=".jpg"):
        if self.args.debug:
            img = img_as_ubyte(img)
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

    def find_circles(self, image, param2=30):
        args = self.args
        logging.debug("Find the rings in image")
        circle_radius_px = int((args.scale * args.circle_diameter)/2)
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(circle_radius_px*2),
            param1=20,
            param2=param2,
            minRadius=int(circle_radius_px * (1-args.circle_radius_tolerance)),
            maxRadius=int(circle_radius_px * (1+args.circle_radius_tolerance))
        )
        return circles

    def find_plates(self):
        args = self.args
        logging.debug("Blur blue-yellow greyscale image")
        b_blur = cv2.medianBlur(self.b, args.kernel)
        self.debug_image(b_blur, "Blue-Yellow median blur")
        logging.debug(
            "Finding circles, increasing the value for param2 until we find all the plates"
        )
        for param2 in range(50, 1, -5):
            param2 = int(param2)
            circles = self.find_circles(b_blur, param2)
            if circles is None:
                logging.debug(f"No circles found with param2 = {param2}")
                continue
            circles = np.squeeze(np.uint16(np.around(circles)))
            if circles.shape[0] < 48:
                logging.debug(f'{str(circles.shape[0])} circles found with param2 = {param2}')
                continue
            logging.debug(f"found {str(circles.shape[0])} circles with param2 = {param2}, proceeding to clustering")
            if args.debug:
                circle_debug = copy(b_blur)
                for c in circles:
                    cv2.circle(circle_debug, (c[0], c[1]), c[2], (255, 0, 0), 5)
                self.debug_image(circle_debug, f"Circles found with param2 = {param2}")
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
            logging.debug(f"Found {len(target_clusters)} clusters with param2 = {param2}")
            if len(target_clusters) < 8:
                continue
            elif len(target_clusters) > 8:
                raise ImageContentException("More than 8 clusters of 6 circles found")
            logging.debug("Collect circles from target clusters into plates")
            logging.debug("")
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
        plates = self.find_plates()
        if plates is None:
            raise ImageContentException("The plate layout could not be detected")
        return self.sort_plates(plates)

    def get_circle_mask(self, plates):
        logging.debug("Draw the circle mask")
        circle_mask = np.full_like(self.b, 255)
        #  radius = int((args.scale * args.circle_diameter) / 2)  # todo see radius note below
        for p in plates:
            for j, c in enumerate(p.circles):
                x = c[0]
                y = c[1]
                radius = c[2]  # todo consider the value of drawing a constant radius
                thickness = 20  # todo consider how thick line to avoid edge
                colour = (0, 0, 0)
                cv2.circle(circle_mask, (x, y), radius, colour, thickness)
        self.debug_image(circle_mask, "Circle mask")
        return circle_mask

    def get_image_mask(self):
        args = self.args

        logging.debug("Threshold the saturation channel (select all colour rich areas")
        ret, s_thresh = cv2.threshold(self.s, args.saturation, 255, cv2.THRESH_BINARY)
        self.debug_image(s_thresh, f"Saturation threshold: {args.saturation}")

        logging.debug("Threshold the value channel to select low exposure pixels")
        # These are often folded lamina disks within blue circles but includes e.g. marbles and pump outside)
        ret, v_thresh = cv2.threshold(self.v, args.value, 255, cv2.THRESH_BINARY_INV)
        self.debug_image(v_thresh, f"Value threshold: {args.value}")

        logging.debug("Join the value or saturation thresholds to create colour mask")
        # i.e. keeping anything coloured or very dark
        colour_mask = cv2.bitwise_or(v_thresh, s_thresh)
        self.debug_image(colour_mask, f"Value or Saturation joined (colour mask)")

        logging.debug("Threshold the green-red channel (a in Lab) to select green tissues")
        ret, a_thresh = cv2.threshold(self.a, args.green_red, 255, cv2.THRESH_BINARY_INV)
        self.debug_image(a_thresh, f"Green-Red (a in Lab) threshold: {args.green_red}")

        logging.debug("Threshold the blue-yellow channel (b in Lab) to select green tissues")
        ret, b_thresh = cv2.threshold(self.b, args.blue_yellow, 255, cv2.THRESH_BINARY_INV)
        self.debug_image(b_thresh, f"Blue-Yellow (b in Lab) threshold: {args.blue_yellow}")

        logging.debug("Join the a and b channels to create a green mask")
        green_mask = cv2.bitwise_and(a_thresh, b_thresh)
        self.debug_image(green_mask, f"a and b join (green mask)")

        logging.debug("Join the colour mask and the green mask to create the image mask")
        image_mask = cv2.bitwise_and(colour_mask, green_mask)
        self.debug_image(image_mask, f"Colour and green joined (image mask)")

        return image_mask

    def get_mask(self, plates):
        args = self.args
        circle_mask = self.get_circle_mask(plates)
        image_mask = self.get_image_mask()

        logging.debug("Mask the area identified as circles in the green mask")
        mask = cv2.bitwise_and(circle_mask, image_mask)
        self.debug_image(mask, "Colour mask and blue ring mask joined")

        logging.debug("Remove small objects in the mask")
        clean_mask = remove_small_objects(img_as_bool(mask), args.remove)
        self.debug_image(clean_mask, "Cleaned mask (removed small objects)")

        logging.debug("Remove small holes in the mask")
        filled_mask = remove_small_holes(clean_mask, args.fill)
        self.debug_image(clean_mask, "Filled mask (removed small holes)")

        return img_as_ubyte(filled_mask)

    def get_area(self):
        args = self.args

        plates = self.get_plates()
        mask = self.get_mask(plates)
        
        result = []
        empty_mask = np.zeros_like(mask)
        overlay_mask = copy(empty_mask)

        for p in plates:
            logging.debug(f"Processing plate {p.plate_id}")
            for j, c in enumerate(p.circles):
                circle_number = j+1+6*(p.plate_id-1)
                logging.debug(f"Processing circle {circle_number}")
                circle_mask = copy(empty_mask)
                cv2.circle(circle_mask, (c[0], c[1]), c[2], (255, 255, 255), -1)
                #  self.debug_image(circle_mask, f"Circle mask: {circle_number}")
                local_mask = cv2.bitwise_and(circle_mask, mask)
                #  self.debug_image(local_mask, "Local mask")
                pixels = cv2.countNonZero(local_mask)
                result.append((p.plate_id, circle_number, pixels))
                if args.overlay:
                    overlay_mask = cv2.bitwise_or(overlay_mask, local_mask)
                    #  self.debug_image(overlay_mask, "Overlay mask")

        if args.overlay:
            logging.debug("Prepare annotated overlay for QC")
            overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2RGB)
            overlay = cv2.addWeighted(self.rgb, 0.5, overlay_mask, 0.5, 0.2, 0)
            self.debug_image(overlay, "Overlay (unlabeled)")
            for p in plates:
                logging.debug(f"Annotate overlay with plate ID: {p.plate_id}")
                cv2.putText(overlay, str(p.plate_id), p.centroid, 0, 5, (255, 0, 255), 5)
                for j, c in enumerate(p.circles):
                    circle_number = j + 1 + 6 * (p.plate_id - 1)
                    # draw the outer circle
                    cv2.circle(overlay, (c[0], c[1]), c[2], (255, 0, 0), 5)
                    # draw the center of the circle
                    cv2.circle(overlay, (c[0], c[1]), 2, (0, 0, 255), 5)
                    cv2.putText(overlay, str(circle_number), c[0:2], 0, 3, (0, 255, 255), 5)

            self.debug_image(overlay, "Overlay (annotated)")
            overlay_path = Path(args.out_dir, "overlay", self.filepath.name)
            overlay_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay)
        return str(self.filepath), result
