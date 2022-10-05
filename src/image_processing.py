import logging
import numpy as np
import cv2
from pathlib import Path
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.util import img_as_ubyte
from skimage.measure import regionprops
from skimage.segmentation import slic
from skimage.color import label2rgb
from copy import copy
from re import search
from datetime import datetime
from .layout import Layout
from .debugger import Debugger
from colour import delta_E

#from colormath.color_objects import LabColor
#from colormath.color_diff import delta_e_cie2000 as delta_e

def area_worker(filepath, args, mean_colour_target):
    result = ImageProcessor(filepath, args, mean_colour_target).get_area()
    filename = result["filename"]
    block_match = search(args.block_regex, str(filename))
    if block_match:
        block = block_match.group(1)
    else:
        block = None
    time_match = search(args.time_regex, str(filename))
    if time_match:
        time = datetime(*[int(time_match[i]) for i in range(1, 6)]).isoformat(sep=" ")
    else:
        time = None
    for record in result["units"]:
        plate = record[0]
        unit = record[1]
        pixels = record[2]
        area = None if pixels is None else round(pixels / (args.scale ** 2), 2)
        yield [filename, block, plate, unit, time, pixels, area]


class ImageProcessor:
    def __init__(self, filepath, args, mean_colour_target):
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.filepath = Path(filepath)
        self.logger.debug(f"Load image as RGB: {self.filepath}")
        self.rgb = cv2.imread(str(self.filepath))

        self.image_debugger = Debugger(args, self.filepath)
        self.image_debugger.render_image(self.rgb, f"Raw: {self.filepath}")
        self.logger.debug(f"Convert RGB to Lab and split")
        self.lab = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2Lab)
        l, a, b = cv2.split(self.lab)
        self.l = l
        self.image_debugger.render_image(l, f"Lightness channel (l in Lab)")
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
        self.mean_colour_target = mean_colour_target

    def get_circle_mask(self, plates):
        self.logger.debug("Draw the circle mask")
        circle_mask = np.zeros_like(self.b)
        #  radius = int((args.scale * args.circle_diameter) / 2)  # todo see radius note below
        for p in plates:
            for j, c in enumerate(p.circles):
                x = c[0]
                y = c[1]
                radius = c[2]  # todo consider the option of drawing a constant radius
                colour = 255
                cv2.circle(circle_mask, (x, y), radius, colour, -1)
        self.image_debugger.render_image(circle_mask, "Circle mask")
        # todo instead of drawing a mask here, build the image labels so can use to intercept with ID rather than drawing twice
        # something like https://stackoverflow.com/questions/34902477/drawing-circles-on-image-with-matplotlib-and-numpy
        return circle_mask

    def get_target_mask(self, circle_mask, n_segments):
        mask = circle_mask.astype('bool')
        segments = slic(self.rgb, convert2lab=True, mask=mask, n_segments=n_segments, enforce_connectivity=False)
        if self.args.image_debug:  # testing here so don't bother creating the labeled image
            self.image_debugger.render_image(label2rgb(segments, self.rgb, kind='avg'), "Labels (average)")
            self.image_debugger.render_image(label2rgb(segments, self.rgb), "Labels (false colour)")
        # find the colour closest to the selected target colour to extract
        colour_deltas = []
        regions_l = regionprops(segments, intensity_image=self.l)
        regions_a = regionprops(segments, intensity_image=self.a)
        regions_b = regionprops(segments, intensity_image=self.b)
        for i in range(n_segments):
            mean_lab = np.array([regions_l[i].mean_intensity, regions_a[i].mean_intensity, regions_b[i].mean_intensity])
            colour_deltas.append(delta_E(mean_lab, np.array(self.mean_colour_target)))
        val, idx = min((val, idx) for (idx, val) in enumerate(colour_deltas))
        target_label = regions_l[idx].label
        # create mask from this region
        target_mask = segments == target_label
        self.image_debugger.render_image(target_mask, "Target mask (SLIC)")
        return target_mask

    def get_mask(self, plates):
        args = self.args
        circle_mask = self.get_circle_mask(plates)
        target_mask = self.get_target_mask(circle_mask, n_segments=5)
        #  todo consider passing n_segments back up to user as option for more complex images

        self.logger.debug("Remove small objects in the mask")
        clean_mask = remove_small_objects(target_mask, args.remove)
        self.image_debugger.render_image(clean_mask, "Cleaned mask (removed small objects)")

        self.logger.debug("Remove small holes in the mask")
        filled_mask = remove_small_holes(clean_mask, args.fill)
        self.image_debugger.render_image(clean_mask, "Filled mask (removed small holes)")

        return img_as_ubyte(filled_mask)  # todo why does this pass back ubyte?
        # name suggests it should be returning a boolean mask

    def get_area(self):
        args = self.args
        plates = Layout(args, self.b, self.image_debugger).get_plates_sorted()
        mask = self.get_mask(plates)
        result = {
            "filename": self.filepath,
            "units": []
        }
        empty_mask = np.zeros_like(mask)
        overlay_mask = copy(empty_mask)
        for p in plates:
            self.logger.debug(f"Processing plate {p.id}")
            for j, c in enumerate(p.circles):
                unit = j+1+6*(p.id-1)
                self.logger.debug(f"Processing circle {unit}")
                circle_mask = copy(empty_mask)
                cv2.circle(circle_mask, (c[0], c[1]), c[2], 255, -1)
                local_mask = cv2.bitwise_and(circle_mask, mask)
                pixels = cv2.countNonZero(local_mask)
                result["units"].append((p.id, unit, pixels))
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
                    unit = j + 1 + 6 * (p.id - 1)
                    # draw the outer circle
                    cv2.circle(overlay, (c[0], c[1]), c[2], (255, 0, 0), 5)
                    # draw the center of the circle
                    cv2.circle(overlay, (c[0], c[1]), 2, (0, 0, 255), 5)
                    cv2.putText(overlay, str(unit), c[0:2], 0, 3, (0, 255, 255), 5)

            self.image_debugger.render_image(overlay, "Overlay (annotated)")
            overlay_path = Path(args.out_dir, "overlay", self.filepath.name)
            overlay_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay)
        return result  # todo consider replacing cv with skimage here too
