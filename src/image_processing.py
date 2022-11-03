import logging
import numpy as np
from pathlib import Path
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import slic
from skimage.color import label2rgb, rgb2lab, gray2rgb, delta_e
from skimage import draw
from skimage.future import graph
from skimage.io import imread, imsave
from re import search
from datetime import datetime
from PIL import Image, ImageDraw
from .layout import Layout
from .debugger import Debugger
from .options import options

logger = logging.getLogger(__name__)


def area_worker(filepath, args):
    result = ImageProcessor(filepath, args).get_area()
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

    def format_result(raw_result):
        for record in raw_result["units"]:
            plate = record[0]
            unit = record[1]
            pixels = record[2]
            area = None if pixels is None else round(pixels / (args.scale ** 2), 2)
            yield [filename, block, plate, unit, time, pixels, area]

    return list(format_result(result))


class ImageProcessor:
    def __init__(self, filepath, args):
        self.filepath = Path(filepath)
        self.args = args
        logger.debug(f"Load image as RGB: {self.filepath}")
        self.rgb = imread(str(self.filepath))
        self.image_debugger = Debugger(self.filepath)
        self.image_debugger.render_image(self.rgb, f"Raw: {self.filepath}")
        logger.debug(f"Convert RGB to Lab")
        self.lab = rgb2lab(self.rgb)
        self.image_debugger.render_image(self.lab[:, :, 0], f"Lightness channel (l in Lab)")
        self.image_debugger.render_image(self.lab[:, :, 1], f"Green-Red channel (a in Lab)")
        self.image_debugger.render_image(self.lab[:, :, 2], f"Blue-Yellow channel (b in Lab)")
        self.empty_mask = np.zeros_like(self.rgb[:, :, 0]).astype("bool")

    @property
    def a(self):
        return self.lab[:, :, 1]

    @property
    def b(self):
        return self.lab[:, :, 2]

    def get_circle_mask(self, circle):
        #  radius = int((args.scale * args.circle_diameter) / 2)  # todo see radius note below
        x = circle[0]
        y = circle[1]
        radius = circle[2]  # todo consider the option of drawing a constant radius
        circle_mask = self.empty_mask.copy()
        yy, xx = draw.disk((y, x), radius, shape=circle_mask.shape)
        circle_mask[yy, xx] = True
        # self.image_debugger.render_image(circle_mask, "Circle mask")
        return circle_mask.astype('bool')

    def get_circles_mask(self, circles):
        logger.debug("get the circles mask")
        circles_mask = np.zeros_like(self.b).astype("bool")
        for circle in circles:
            circles_mask = circles_mask | self.get_circle_mask(circle)
        self.image_debugger.render_image(circles_mask, "Circles mask")
        return circles_mask

    def get_target_mask(self, circles, max_delta_e):
        circles_mask = self.get_circles_mask(circles)
        logger.debug(f"cluster the region of interest into segments")
        segments = slic(
            self.rgb,
            # broken behaviour when using existing transformation and convert2lab=false
            # just allowing slic to do own conversion
            # todo work out what this conversion is doing differently
            mask=circles_mask,
            n_segments=50,
            compactness=10,
            #slic_zero=True,
            convert2lab=True,
            enforce_connectivity=False
        )
        if self.args.image_debug:
            self.image_debugger.render_image(label2rgb(segments, self.rgb, kind='avg'), "Labels (average)")
            self.image_debugger.render_image(label2rgb(segments, self.rgb), "Labels (false colour)")
        # find the colour closest to the selected target colour (from picker) to extract
        logger.debug(f"Find segments with colour within {max_delta_e} of target")
        colour_deltas = []
        target_colour = np.array([self.args.target_l, self.args.target_a, self.args.target_b])
        for i in range(len(np.unique(segments))):
            #segment_a = self.a[segments == i]
            #segment_b = self.b[segments == i]
            #colour_deltas.append(segment_a - self.args.target_a)
            #colour_deltas.append(np.sqrt(np.add(np.square(segment_a - self.args.target_a), np.square(segment_b - self.args.target_b))).mean())
            segment = self.lab[segments == i]
            colour_deltas.append(delta_e.deltaE_cie76(target_colour, segment).mean())
        target_regions = [idx for (idx, val) in enumerate(colour_deltas) if val < max_delta_e]
        # create mask from this region
        target_mask = np.isin(segments, target_regions)
        logger.debug("Remove small objects in the mask")
        clean_mask = remove_small_objects(target_mask, self.args.remove)
        self.image_debugger.render_image(clean_mask, "Cleaned mask (removed small objects)")
        logger.debug("Remove small holes in the mask")
        filled_mask = remove_small_holes(clean_mask, self.args.fill)
        self.image_debugger.render_image(filled_mask, "Filled mask (removed small holes)")
        return target_mask

    def get_area(self):
        if self.args.circle_channel == 'a':
            channel = self.a
        else:
            channel = self.b

        plates = Layout(channel, self.image_debugger).get_plates_sorted()
        result = {
            "filename": self.filepath,
            "units": []
        }

        all_circles = [c for p in plates for c in p.circles]
        target_mask = self.get_target_mask(all_circles, max_delta_e=20)

        if self.args.overlay:
            logger.debug("Prepare annotated overlay for QC")
            overlay_mask = np.zeros_like(self.a, dtype="bool")
            alpha = 0.2
            blended = (alpha * self.rgb) + ((1-alpha) * gray2rgb(target_mask))
            blended = np.asarray(blended, dtype="uint8")
            annotated_image = Image.fromarray(blended)
            draw_tool = ImageDraw.Draw(annotated_image)

        for p in plates:
            logger.debug(f"Processing plate {p.id}")
            if self.args.overlay:
                logger.debug(f"Annotate overlay with plate ID: {p.id}")
                draw_tool.text(p.centroid, str(p.id), (255, 0, 255))
            for j, c in enumerate(p.circles):
                unit = j+1+6*(p.id-1)
                logger.debug(f"Processing circle {unit}")
                circle_mask = self.get_circle_mask(c)
                circle_target = circle_mask & target_mask
                # todo pass these variables up as configurable options
                pixels = np.count_nonzero(circle_target)
                result["units"].append((p.id, unit, pixels))
                if self.args.overlay:
                    logger.debug(f"Join target to overlay mask: {p.id}")
                    unit = j + 1 + 6 * (p.id - 1)
                    # draw the outer circle
                    x = c[0]
                    y = c[1]
                    r = c[2]
                    #xx, yy = draw.circle_perimeter_aa(x, y, r, shape=circle_mask.shape)
                    draw_tool.text((x,y), str(unit), (0,255,255))
                    draw_tool.ellipse((x-r, y-r, x+r, y+r), fill=(255, 0, 0, 0))
        if self.args.overlay:
            self.args.image_debug = "plot"
            import pdb; pdb.set_trace()
            self.image_debugger.render_image(np.array(annotated_image), "Overlay (unlabeled)")
            overlay_path = Path(self.args.out_dir, "overlay", self.filepath.name)
            overlay_path.parent.mkdir(exist_ok=True)
            imsave(str(overlay_path), annotated_image)
        return result  # todo consider replacing cv with skimage here too
