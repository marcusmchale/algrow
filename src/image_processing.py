import logging
import numpy as np
from pathlib import Path
from skimage.morphology import remove_small_holes, remove_small_objects, binary_dilation
from skimage.segmentation import slic
from skimage.color import label2rgb, rgb2lab, deltaE_ciede2000
from skimage import draw
from skimage.io import imread, imsave
from skimage.future import graph
from networkx import node_connected_component, set_edge_attributes, get_edge_attributes
import matplotlib.pyplot as plt
from re import search
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from .layout import Layout
from .debugger import Debugger
from collections import defaultdict

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
        self.image_debugger = Debugger(self.filepath, args)
        self.image_debugger.render_image(self.rgb, f"Raw: {self.filepath}")
        logger.debug(f"Convert RGB to Lab")
        self.lab = rgb2lab(self.rgb)
        self.image_debugger.render_image(self.lab[:, :, 0], f"Lightness channel (l in Lab)")
        self.image_debugger.render_image(self.lab[:, :, 1], f"Green-Red channel (a in Lab)")
        self.image_debugger.render_image(self.lab[:, :, 2], f"Blue-Yellow channel (b in Lab)")
        self.empty_mask = np.zeros_like(self.rgb[:, :, 0]).astype("bool")
        self.circle_id_map = defaultdict(set)

    @property
    def l(self):
        return self.lab[:, :, 0]

    @property
    def a(self):
        return self.lab[:, :, 1]

    @property
    def b(self):
        return self.lab[:, :, 2]

    def get_circle_mask(self, circle):
        x = circle[0]
        y = circle[1]
        radius = circle[2]
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

    def plot_adjacency_in_lab(self, rag, segments, prefix=None):
        if self.args.image_debug:
            # todo move this sort of plotting to the image debugger

            set_edge_attributes(rag, get_edge_attributes(rag, "delta_e"), name="weight")
            lc = graph.show_rag(segments, rag, self.rgb, border_color='white')
            plt.colorbar(lc)
            plt.title(f"{prefix} :Lab deltaE scaled network")

            plt.show()

    def build_graph(self, segments, connectivity=1, graph_dist=8, kl=2):

        logger.debug("Build adjacency graph")

        rag = graph.RAG(segments, connectivity=connectivity)

        for n in rag:
            rag.nodes[n].update({
                "labels": [n],
                "pixel count": 0,
                "total color": np.array([0, 0, 0], dtype=np.float64),
                "target color distance": None
            })

        for index in np.ndindex(segments.shape):
            current = segments[index]
            rag.nodes[current]['pixel count'] += 1
            rag.nodes[current]['total color'] += self.lab[index]

        # add the mean color and distance from any target colour to select the closest node in roi
        target_lab = np.array(self.args.target_colour)

        logger.debug("add distance from target colours to each node")

        if self.args.image_debug:
            dist_image = segments.copy()

        for n in rag:
            rag.nodes[n]['mean color'] = (rag.nodes[n]['total color'] / rag.nodes[n]['pixel count'])
            #target_distance = np.linalg.norm(rag.nodes[n]['mean color'] - target_lab)
            target_distance = min(
                deltaE_ciede2000(
                    np.tile(rag.nodes[n]['mean color'], target_lab.shape[0]).reshape(target_lab.shape[0],3),
                    target_lab,
                    kL=kl
                )
            )
            rag.nodes[n]["target color distance"] = target_distance
            if self.args.image_debug:
                dist_image[segments == n] = target_distance

        if self.args.image_debug:
            #todo add debug image with scale set from 0 to target_dist
            self.image_debugger.render_image(dist_image, "Superpixel distance from any target colour")

        logger.debug("add distance in colour as weights")
        for x, y, d in list(rag.edges(data=True)):
            d['delta_e'] = deltaE_ciede2000(rag.nodes[x]['mean color'], rag.nodes[y]['mean color'], kL=kl)

        self.plot_adjacency_in_lab(rag, segments, prefix="Full network")

        logger.debug("remove edges above {graph_dist} delta_e")
        for x, y, d in list(rag.edges(data=True)):
            if d['delta_e'] > graph_dist:
                rag.remove_edge(x, y)
        self.plot_adjacency_in_lab(rag, segments, prefix="Pruned by deltaE")

        logger.debug("remove edges connected to 0 node (background)")
        background_edges = list(rag.edges(0))
        rag.remove_edges_from(background_edges)
        self.plot_adjacency_in_lab(rag, segments, prefix="Background removed")

        return rag

    def get_segments(self, circles, n_segments=1500, compactness=10):
        #todo pass n_segments up as option
        circles_mask = self.get_circles_mask(circles)
        segments = slic(
            self.rgb,
            # this function currently has broken behaviour when using the existing transformation and convert2lab=false
            # just allowing this function to redo conversion from rgb
            # todo work out what this is doing differently when using the already converted lab image
            mask=circles_mask,
            n_segments=n_segments,
            compactness=compactness,
            convert2lab=True,
            enforce_connectivity=True
        )
        # The slic output includes segments that span circles which breaks graph building.
        # Clean it up by iterating through circles and relabel segments if found in another circle
        # todo need to work out why these edges are being made in the first place,
        #  should only be connections between circles through 0
        circles_per_segment = defaultdict(int)
        segment_counter = np.max(segments)
        for circle in circles:
            circle_mask = self.get_circle_mask(circle)
            circle_segments = segments.copy()
            circle_segments[~circle_mask] = -1 # to differentiate the background in circle from background outside
            circle_segment_ids = set(np.unique(circle_segments))
            circle_segment_ids.remove(-1)
            self.circle_id_map[tuple(circle)] = circle_segment_ids
            for i in list(circle_segment_ids):
                circles_per_segment[i] += 1
                if circles_per_segment[i] > 1 or i == 0 :  # add a new segment for this ID in this circle, always relabel if part of background but inside circle
                    segment_counter += 1
                    segments[circle_segments == i] = segment_counter
                    self.circle_id_map[tuple(circle)].remove(i)
                    self.circle_id_map[tuple(circle)].add(segment_counter)

        if self.args.image_debug:
            # self.image_debugger.render_image(label2rgb(segments, self.rgb, kind='avg'), "Labels (average)")
            self.image_debugger.render_image(label2rgb(segments, self.rgb), "Labels (false colour)")

        return segments

    def get_target_mask(self, circles, target_dist=8):
        logger.debug(f"cluster the region of interest into segments")
        segments = self.get_segments(circles)

        # Create a regional adjacency graph with weights based on distance of a and b in Lab colourspace
        rag = self.build_graph(segments, graph_dist=self.args.graph_dist)
        starting_segments = set()
        target_segments = set()

        for circle in circles:
            for n in self.circle_id_map[tuple(circle)]:
                if rag.nodes[n]["target color distance"] <= target_dist:
                    starting_segments.add(n)
                    target_segments.update(node_connected_component(rag, n))
            # todo consider getting area directly from regionprops ... but we are doing it with fill etc.
        # create mask from this region
        starting_mask = np.isin(segments, np.array(list(starting_segments)))

        self.image_debugger.render_image(starting_mask, "Starting node mask (those superpixels within target_dist of a target colour)")
        target_mask = np.isin(segments, np.array(list(target_segments)))
        self.image_debugger.render_image(target_mask, "Target node mask (include connected to starting nodes by less than graph_dist)")

        logger.debug("Remove small objects in the mask")
        clean_mask = remove_small_objects(target_mask, self.args.remove)
        self.image_debugger.render_image(clean_mask, "Cleaned mask (removed small objects)")
        logger.debug("Remove small holes in the mask")
        filled_mask = remove_small_holes(clean_mask, self.args.fill)
        self.image_debugger.render_image(filled_mask, "Filled mask (removed small holes)")

        if self.args.image_debug:
            target_segments_image = segments.copy()
            target_segments_image[np.isin(target_segments_image, np.array(list(target_segments)), invert=True)] = 0
            self.image_debugger.render_image(label2rgb(target_segments_image, self.rgb), "Labels (false colour)")

        return target_mask

    def get_area(self):
        if self.args.circle_channel == 'a':
            channel = self.a
        else:
            channel = self.b

        plates = Layout(channel, self.image_debugger, self.args).get_plates_sorted()
        result = {
            "filename": self.filepath,
            "units": []
        }

        all_circles = [c for p in plates for c in p.circles]
        target_mask = self.get_target_mask(
            all_circles,
            target_dist=self.args.target_dist
        )

        if self.args.overlay:
            logger.debug("Prepare annotated overlay for QC")
            blended = self.rgb.copy()
            contour = binary_dilation(target_mask, footprint=np.full((5,5), 1))
            contour[target_mask] = False
            blended[contour] = (255, 0, 255)
            # the below would lower the intensity of the not target area in the image, not necessary
            #  blended[~target_mask] = np.divide(blended[~target_mask], 2)
            annotated_image = Image.fromarray(blended)
            draw_tool = ImageDraw.Draw(annotated_image)
        height = self.rgb.shape[0]
        font_file = font_manager.findfont(font_manager.FontProperties())
        large_font = ImageFont.truetype(font_file, size=int(height/50), encoding="unic")
        small_font = ImageFont.truetype(font_file, size=int(height/80), encoding="unic")
        for p in plates:
            logger.debug(f"Processing plate {p.id}")
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
                    draw_tool.text((x,y), str(unit), "blue", small_font)
                    draw_tool.ellipse((x-r, y-r, x+r, y+r), outline=(255, 255, 0), fill=None, width=5)
            if self.args.overlay:
                logger.debug(f"Annotate overlay with plate ID: {p.id}")
                draw_tool.text(p.centroid, str(p.id), "red", large_font)
        if self.args.overlay:
            self.image_debugger.render_image(np.array(annotated_image), "Overlay (unlabeled)")
            overlay_path = Path(self.args.out_dir, "overlay", self.filepath.name)
            overlay_path.parent.mkdir(exist_ok=True)
            imsave(str(overlay_path), np.array(annotated_image))
        return result  # todo consider replacing cv with skimage here too
