import logging
import numpy as np
import cv2 as cv2
from pathlib import Path
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.util import img_as_ubyte, img_as_bool
from copy import copy
from .layout import Layout
from .debugger import Debugger


def area_worker(filepath, args):
    return ImageProcessor(filepath, args).get_area()


class ImageProcessor:
    def __init__(self, filepath, args):
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.filepath = Path(filepath)
        self.logger.debug(f"Load image as RGB: {self.filepath}")
        self.rgb = cv2.imread(str(self.filepath))

        self.image_debugger = Debugger(args, self.filepath)
        self.image_debugger.render_image(self.rgb, f"Raw: {self.filepath}")
        assert self.rgb.dtype == 'uint8'
        self.logger.debug(f"Convert RGB to Lab and split")
        l, a, b = cv2.split(cv2.cvtColor(self.rgb, cv2.COLOR_RGB2Lab))
        self.a = a
        self.image_debugger.render_image(a, f"Green-Red channel (a in Lab)")
        self.b = b
        self.image_debugger.render_image(b, f"Blue-Yellow channel (b in Lab)")
        self.logger.debug("Convert RGB to HSV and split")
        h, s, v = cv2.split(cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV))
        self.s = s
        self.image_debugger.render_image(s, "Saturation (S in HSV)")
        self.v = v
        self.image_debugger.render_image(v, "Value (V in HSV)")

    def get_circle_mask(self, plates):
        self.logger.debug("Draw the circle mask")
        circle_mask = np.full_like(self.b, 255)
        #  radius = int((args.scale * args.circle_diameter) / 2)  # todo see radius note below
        for p in plates:
            for j, c in enumerate(p.circles):
                x = c[0]
                y = c[1]
                radius = c[2]  # todo consider the option of drawing a constant radius
                thickness = 20  # todo consider how thick line should be, good to have boundary of detection?
                colour = (0, 0, 0)
                cv2.circle(circle_mask, (x, y), radius, colour, thickness)
        self.image_debugger.render_image(circle_mask, "Circle mask")
        return circle_mask

    def get_image_mask(self):
        args = self.args

        self.logger.debug("Threshold the saturation channel (select all colour rich areas")
        ret, s_thresh = cv2.threshold(self.s, args.saturation, 255, cv2.THRESH_BINARY)
        self.image_debugger.render_image(s_thresh, f"Saturation threshold: {args.saturation}")

        self.logger.debug("Threshold the value channel to select low exposure pixels")
        # These are often folded lamina disks within blue circles but includes e.g. marbles and pump outside)
        ret, v_thresh = cv2.threshold(self.v, args.value, 255, cv2.THRESH_BINARY_INV)
        self.image_debugger.render_image(v_thresh, f"Value threshold: {args.value}")

        self.logger.debug("Join the value or saturation thresholds to create colour mask")
        # i.e. keeping anything coloured or very dark
        colour_mask = cv2.bitwise_or(v_thresh, s_thresh)
        self.image_debugger.render_image(colour_mask, f"Value or Saturation joined (colour mask)")

        self.logger.debug("Threshold the green-red channel (a in Lab) to select green tissues")
        ret, a_thresh = cv2.threshold(self.a, args.green_red, 255, cv2.THRESH_BINARY_INV)
        self.image_debugger.render_image(a_thresh, f"Green-Red (a in Lab) threshold: {args.green_red}")

        self.logger.debug("Threshold the blue-yellow channel (b in Lab) to select green tissues")
        ret, b_thresh = cv2.threshold(self.b, args.blue_yellow, 255, cv2.THRESH_BINARY_INV)
        self.image_debugger.render_image(b_thresh, f"Blue-Yellow (b in Lab) threshold: {args.blue_yellow}")

        self.logger.debug("Join the a and b channels to create a green mask")
        green_mask = cv2.bitwise_and(a_thresh, b_thresh)
        self.image_debugger.render_image(green_mask, f"a and b join (green mask)")

        self.logger.debug("Join the colour mask and the green mask to create the image mask")
        image_mask = cv2.bitwise_and(colour_mask, green_mask)
        self.image_debugger.render_image(image_mask, f"Colour and green joined (image mask)")

        return image_mask

    def get_mask(self, plates):
        args = self.args
        circle_mask = self.get_circle_mask(plates)
        image_mask = self.get_image_mask()

        self.logger.debug("Mask the area identified as circles in the green mask")
        mask = cv2.bitwise_and(circle_mask, image_mask)
        self.image_debugger.render_image(mask, "Colour mask and blue ring mask joined")

        self.logger.debug("Remove small objects in the mask")
        clean_mask = remove_small_objects(img_as_bool(mask), args.remove)
        self.image_debugger.render_image(clean_mask, "Cleaned mask (removed small objects)")

        self.logger.debug("Remove small holes in the mask")
        filled_mask = remove_small_holes(clean_mask, args.fill)
        self.image_debugger.render_image(clean_mask, "Filled mask (removed small holes)")

        return img_as_ubyte(filled_mask)

    def get_area(self):
        args = self.args
        plates = Layout(args, self.b, self.image_debugger).get_plates_sorted()
        mask = self.get_mask(plates)
        result = []
        empty_mask = np.zeros_like(mask)
        overlay_mask = copy(empty_mask)
        for p in plates:
            self.logger.debug(f"Processing plate {p.id}")
            for j, c in enumerate(p.circles):
                circle_number = j+1+6*(p.id-1)
                self.logger.debug(f"Processing circle {circle_number}")
                circle_mask = copy(empty_mask)
                cv2.circle(circle_mask, (c[0], c[1]), c[2], (255, 255, 255), -1)
                local_mask = cv2.bitwise_and(circle_mask, mask)
                pixels = cv2.countNonZero(local_mask)
                result.append((p.id, circle_number, pixels))
                if args.overlay:
                    overlay_mask = cv2.bitwise_or(overlay_mask, local_mask)
        if args.overlay:
            self.logger.debug("Prepare annotated overlay for QC")
            overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2RGB)
            overlay = cv2.addWeighted(self.rgb, 0.5, overlay_mask, 0.5, 0.2, 0)
            self.image_debugger.render_image(overlay, "Overlay (unlabeled)")
            for p in plates:
                self.logger.debug(f"Annotate overlay with plate ID: {p.id}")
                cv2.putText(overlay, str(p.id), p.centroid, 0, 5, (255, 0, 255), 5)
                for j, c in enumerate(p.circles):
                    circle_number = j + 1 + 6 * (p.id - 1)
                    # draw the outer circle
                    cv2.circle(overlay, (c[0], c[1]), c[2], (255, 0, 0), 5)
                    # draw the center of the circle
                    cv2.circle(overlay, (c[0], c[1]), 2, (0, 0, 255), 5)
                    cv2.putText(overlay, str(circle_number), c[0:2], 0, 3, (0, 255, 255), 5)

            self.image_debugger.render_image(overlay, "Overlay (annotated)")
            overlay_path = Path(args.out_dir, "overlay", self.filepath.name)
            overlay_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay)
        return str(self.filepath), result
