import logging
import numpy as np
from pathlib import Path
from skimage.morphology import remove_small_holes, remove_small_objects, binary_dilation
from skimage.segmentation import slic
from skimage.color import label2rgb, rgb2lab, deltaE_ciede2000, deltaE_cie76
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

    def plot_adjacency_in_lab(self, rag, segments, prefix=None):
        if self.args.image_debug:
            # todo move this sort of plotting to the image debugger
            #set_edge_attributes(rag, get_edge_attributes(rag, "euclidian_dist"), name="weight")
            #lc = graph.show_rag(segments, rag, self.rgb)
            #plt.colorbar(lc)
            #plt.title(f"{prefix}: Lab euclidian distance network")
            #
            #set_edge_attributes(rag, get_edge_attributes(rag, "l_dist"), name="weight")
            #lc = graph.show_rag(segments, rag, self.rgb)
            #plt.colorbar(lc)
            #plt.title(f"{prefix}: Lab 'l_dist' network")
            #
            #set_edge_attributes(rag, get_edge_attributes(rag, "a_dist"), name="weight")
            #lc = graph.show_rag(segments, rag, self.rgb)
            #plt.colorbar(lc)
            #plt.title(f"{prefix}: Lab 'a_dist' network")
            #
            #set_edge_attributes(rag, get_edge_attributes(rag, "b_dist"), name="weight")
            #lc = graph.show_rag(segments, rag, self.rgb)
            #plt.colorbar(lc)
            #plt.title(f"{prefix} :Lab 'b_dist' network")

            set_edge_attributes(rag, get_edge_attributes(rag, "delta_e"), name="weight")
            lc = graph.show_rag(segments, rag, self.rgb)
            plt.colorbar(lc)
            plt.title(f"{prefix} :Lab deltaE network")

            set_edge_attributes(rag, get_edge_attributes(rag, "delta_e_scaled"), name="weight")
            lc = graph.show_rag(segments, rag, self.rgb)
            plt.colorbar(lc)
            plt.title(f"{prefix} :Lab deltaE scaled network")

            plt.show()


    def build_graph(self, segments, connectivity=1, max_dist_l=20, max_dist_a=20, max_dist_b=40):
        # todo pass max_dist parameters up to options()
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

        # add the mean color and distance from target to select the closest node in roi
        target_lab = np.array([self.args.target_l, self.args.target_a, self.args.target_b])
        logger.debug("add distance from target colour to each node")
        for n in rag:
            rag.nodes[n]['mean color'] = (rag.nodes[n]['total color'] / rag.nodes[n]['pixel count'])
            #target_distance = np.linalg.norm(rag.nodes[n]['mean color'] - target_lab)
            target_distance = deltaE_ciede2000(rag.nodes[n]['mean color'], target_lab)
            rag.nodes[n]["target color distance"] = target_distance

        logger.debug("add distance in colour as weights")
        # remove rels between nodes connected by more than max_weight
        for x, y, d in list(rag.edges(data=True)):
            l_dist = np.linalg.norm(rag.nodes[x]['mean color'][0] - rag.nodes[y]['mean color'][0])
            a_dist = np.linalg.norm(rag.nodes[x]['mean color'][1] - rag.nodes[y]['mean color'][1])
            b_dist = np.linalg.norm(rag.nodes[x]['mean color'][2] - rag.nodes[y]['mean color'][2])
            d['l_dist'] = l_dist
            d['a_dist'] = a_dist
            d['b_dist'] = b_dist
            d['euclidian_dist'] = np.linalg.norm(rag.nodes[x]['mean color'] - rag.nodes[y]['mean color'])
            d['delta_e'] = deltaE_ciede2000(rag.nodes[x]['mean color'], rag.nodes[y]['mean color'])
            d['delta_e_scaled'] = deltaE_ciede2000(rag.nodes[x]['mean color'], rag.nodes[y]['mean color'], kL=2)
            #d['delta_e'] = deltaE_cie76(rag.nodes[x]['mean color'], rag.nodes[y]['mean color'])

        self.plot_adjacency_in_lab(rag, segments, prefix="Full network")

        for x, y, d in list(rag.edges(data=True)):
            if d['delta_e_scaled'] > 10:
                rag.remove_edge(x, y)

        self.plot_adjacency_in_lab(rag, segments, prefix="Pruned by deltaE")

        return rag

    def get_target_mask(self, circles, n_segments=1000):
        circles_mask = self.get_circles_mask(circles)
        logger.debug(f"cluster the region of interest into segments")
        segments = slic(
            self.rgb,
            # this function currently has broken behaviour when using the existing transformation and convert2lab=false
            # just allowing this function to redo conversion from rgb
            # todo work out what this is doing differently when using the already converted lab image
            mask=circles_mask,
            n_segments=n_segments,
            compactness=10,
            convert2lab=True,
            enforce_connectivity=True
        )

        if self.args.image_debug:
            # self.image_debugger.render_image(label2rgb(segments, self.rgb, kind='avg'), "Labels (average)")
            self.image_debugger.render_image(label2rgb(segments, self.rgb), "Labels (false colour)")

        # Create a regional adjacency graph with weights based on distance of a and b in Lab colourspace
        rag = self.build_graph(segments)

        target_segments = set()
        found_circle_segments_ids = set()  # keep this as we don't want to allow connections between circles

        from collections import defaultdict
        repeated_circle_segment_ids = defaultdict(int)
        # todo need to work out why these edges are being made in the first place, should only be to 0#
        # It seems like slic is creating superpixels that span circles!
        # this must be something strange about the slic mask implementation
        # that does not respect the enforce_connectivity rule
        # todo consider relabelling these manually before building the graph?
        import pdb;
        pdb.set_trace()
        # now within each circle
        for circle in circles:
            circle_mask = self.get_circle_mask(circle)
            circle_segments = segments.copy()
            circle_segments[~circle_mask] = 0
            #circle_segment_ids = np.delete(np.unique(circle_segments), 0)
            circle_segment_ids = np.unique(circle_segments)
            # all except background
            if set(circle_segment_ids).intersection(found_circle_segments_ids):
                for i in set(circle_segment_ids).intersection(found_circle_segments_ids):
                    repeated_circle_segment_ids[i] += 1
            found_circle_segments_ids.update(circle_segment_ids)

            closest_target_distance = np.inf
            starting_node = None
            # todo consider a global threshold for starting node tolerance,
            #  currently just taking the closest but should be taking the closest within some max distance threshold
            #  or we get obvious background

            circle_target_segments = set()

            for n in circle_segment_ids:
                if rag.nodes[n]['mean color'][0] < self.args.dark_target:
                    circle_target_segments.add(n)
                if rag.nodes[n]["target color distance"] < closest_target_distance:
                    starting_node = n
                    closest_target_distance = rag.nodes[n]["target color distance"]

            if self.args.image_debug:
                # highlight the starting segment
                starting_node_mask = np.equal(segments, starting_node)
                starting_node_overlay = self.rgb.copy()
                starting_node_overlay[starting_node_mask] = (255, 0, 255)
                # self.image_debugger.render_image(starting_node_overlay, f"Starting node for circle")

            # now start at the closest node and get connected nodes
            circle_target_segments.update(node_connected_component(rag, starting_node))
            circle_target_segments = circle_target_segments.union(circle_segment_ids)
            target_segments.update(circle_target_segments)
            # todo consider getting area directly from regionprops ... but we are doing it with fill etc.

        import pdb; pdb.set_trace()
        for n in rag.nodes:
            if n in found_circle_segments_ids:
                pass
        self.plot_adjacency_in_lab(rag, segments, prefix="Pruned by deltaE")

        # create mask from this region
        target_mask = np.isin(segments, np.array(list(target_segments)))
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
        target_mask = self.get_target_mask(all_circles)

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
