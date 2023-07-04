import logging
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects, binary_dilation

from trimesh import proximity

from re import search
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from .layout import LayoutDetector
from .figurebuilder import FigureBuilder
from .logging import CustomAdapter
from .image_loading import ImageLoaded


logger = logging.getLogger(__name__)


def area_worker(filepath, alpha_hull, args):
    adapter = CustomAdapter(logger, {'image_filepath': str(filepath)})

    adapter.debug(f"Processing file: {filepath}")
    result = ImageProcessor(filepath, alpha_hull, args).get_area()
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
    def __init__(self, filepath, alpha_hull, args):
        self.image = ImageLoaded(filepath, args)
        self.alpha_hull = alpha_hull
        self.args = args
        self.logger = CustomAdapter(logger, {'image_filepath': str(filepath)})

    def get_area(self):
        self.logger.debug("Circle expansion")
        layout = LayoutDetector(self.image).get_layout()

        # break up the masked image into chunks of points as the signed_distance calculation gives OOM otherwise
        masked_lab = self.image.lab[layout.mask]
        ind = np.linspace(0, len(masked_lab), num=int(len(masked_lab)/1e5), dtype=int, endpoint=False)[1:]

        self.logger.debug("Calculate distance from alpha hull for all pixels")
        distances = list()
        for c in np.split(masked_lab, ind):
            dist_c = proximity.signed_distance(self.alpha_hull, c)
            distances.append(dist_c)
        distances_array = np.concatenate(distances)

        distance_image = np.empty(self.image.lab.shape[0:2])
        distance_image[layout.mask] = np.negative(distances_array)
        distance_image[~layout.mask] = 0  # set masked region as 0

        if self.args.debug:
            fig = FigureBuilder(self.image.filepath, self.args, 'Alpha hull distance')
            fig.add_image(distance_image, color_bar=True)
            fig.print(large=True)

        target_mask = (distance_image < self.args.delta) & layout.mask

        fig = FigureBuilder(
            self.image.filepath,
            self.args,
            "Fill mask",
            nrows=len([i for i in [self.args.remove, self.args.fill] if i])+1
        ) if self.args.debug else None
        if fig:
            self.logger.debug("Raw mask from distance threshold")
            fig.add_image(target_mask, "Raw mask")

        if self.args.remove:
            self.logger.debug("Remove small objects in the mask")
            target_mask = remove_small_objects(target_mask, self.args.remove)
            if fig:
                fig.add_image(target_mask, "Removed small objects")

        if self.args.fill:
            self.logger.debug("Fill small holes in the mask")
            target_mask = remove_small_holes(target_mask, self.args.fill)
            if fig:
                fig.add_image(target_mask, "Filled small holes")
        if fig:
            fig.print()

        if self.args.overlay or self.args.debug:
            self.logger.debug("Prepare annotated overlay for QC")
            blended = self.image.rgb.copy()
            contour = binary_dilation(target_mask, footprint=np.full((5,5), 1))
            contour[target_mask] = False
            blended[contour] = (255, 0, 255)
            # the below would lower the intensity of the not target area in the image, not necessary
            #  blended[~target_mask] = np.divide(blended[~target_mask], 2)
            annotated_image = Image.fromarray(blended)
            draw_tool = ImageDraw.Draw(annotated_image)
        else:
            draw_tool = None
            annotated_image = None

        height = self.image.rgb.shape[0]
        font_file = font_manager.findfont(font_manager.FontProperties())
        large_font = ImageFont.truetype(font_file, size=int(height/50), encoding="unic")
        small_font = ImageFont.truetype(font_file, size=int(height/80), encoding="unic")

        result = {
            "filename": self.image.filepath,
            "units": []
        }

        for p in layout.plates:
            self.logger.debug(f"Processing plate {p.id}")
            for j, c in enumerate(p.circles):
                unit = j+1+6*(p.id-1)
                circle_mask = layout.get_circle_mask(c)
                circle_target = circle_mask & target_mask
                pixels = np.count_nonzero(circle_target)
                result["units"].append((p.id, unit, pixels))
                if self.args.overlay or self.args.debug:
                    unit = j + 1 + 6 * (p.id - 1)
                    # draw the outer circle
                    x = c[0]
                    y = c[1]
                    r = c[2]
                    draw_tool.text((x, y), str(unit), "blue", small_font)
                    draw_tool.ellipse((x-r, y-r, x+r, y+r), outline=(255, 255, 0), fill=None, width=5)
            if self.args.overlay or self.args.debug:
                draw_tool.text(p.centroid, str(p.id), "red", large_font)
        if self.args.overlay or self.args.debug:
            fig = FigureBuilder(self.image.filepath, self.args, "Overlay", force="save")
            fig.add_image(annotated_image)
            fig.print()
        return result

