import logging
import numpy as np
from pathlib import Path
from skimage.morphology import remove_small_holes, remove_small_objects, binary_dilation
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.color import label2rgb, rgb2lab, deltaE_ciede2000
from skimage import draw
from skimage.io import imread, imsave
from skimage import graph
from networkx import node_connected_component, set_edge_attributes, get_edge_attributes
import matplotlib.pyplot as plt
from re import search
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from .layout import Layout
from .figurebuilder import FigureBuilder
from collections import defaultdict

logger = logging.getLogger(__name__)

class OverlappingCircles(Exception):
    pass


def area_worker(filepath, args):
    logger.debug(f"Processing file: {filepath}")
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

        if args.debug:
            fig = FigureBuilder(self.filepath, "Target colours")
            fig.plot_colours(vars(args)['target_colour'])
            fig.print()
            target_colours_string = f'{[",".join([str(j) for j in i]) for i in vars(args)["target_colour"]]}'.replace("'", '"')

            fig = FigureBuilder(self.filepath, "Non-target colours")
            fig.plot_colours(vars(args)['non_target_colour'])
            fig.print()
            non_target_colours_string = f'{[",".join([str(j) for j in i]) for i in vars(args)["non_target_colour"]]}'.replace("'", '"')

            fig = FigureBuilder(filepath, "Circle colour")
            fig.plot_colours([vars(args)['circle_colour']])
            fig.print()
            circle_colour_string = f"\"{','.join([str(i) for i in vars(args)['circle_colour']])}\""

            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(args.out_dir, "colours.txt"), 'w') as text_file:
                text_file.write("target_colours:")
                text_file.write(target_colours_string)
                text_file.write("\n")
                text_file.write("non_target_colours:")
                text_file.write(non_target_colours_string)
                text_file.write("\n")
                text_file.write("circle_colour:")
                text_file.write(circle_colour_string)
                text_file.write("\n")

        logger.debug(f"Load image as RGB: {self.filepath}")
        self.rgb = imread(str(self.filepath))
        if args.debug:
            fig = FigureBuilder(self.filepath, "Load image")
            fig.add_image(self.rgb, Path(self.filepath).stem)
            fig.print()
        logger.debug(f"Convert RGB to Lab")
        self.lab = rgb2lab(self.rgb)
        if args.debug:
            fig = FigureBuilder(self.filepath, "Convert to Lab", nrows = 3)
            fig.add_image(self.lab[:, :, 0], "Lightness channel (l in Lab)", color_bar=True)
            fig.add_image(self.lab[:, :, 1], "Green-Red channel (a in Lab)", color_bar=True)
            fig.add_image(self.lab[:, :, 2], "Blue-Yellow channel (b in Lab)", color_bar=True)
            fig.print()
        self.empty_mask = np.zeros_like(self.rgb[:, :, 0]).astype("bool")
        #self.circle_id_map = defaultdict(set)

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
        return circle_mask.astype('bool')

    def get_circles_mask(self, circles):
        logger.debug("get the circles mask")
        circles_mask = np.zeros_like(self.b).astype("bool")
        for circle in circles:
            circle_mask = self.get_circle_mask(circle)
            if np.logical_and(circles_mask, circle_mask).any():
                raise OverlappingCircles("Circles overlapping - consider trying again with a lower circle_expansion factor")
            circles_mask = circles_mask | circle_mask
        if self.args.debug:
            fig = FigureBuilder(self.filepath, "Circles mask")
            fig.add_image(circles_mask)
            fig.print()
        return circles_mask

    def subplot_adjacency(self, axis, rag, segments, prefix=None):
        set_edge_attributes(rag, get_edge_attributes(rag, "delta_e"), name="weight")
        lc = graph.show_rag(segments, rag, self.rgb, border_color='white', ax=axis, edge_width=0.5)
        plt.colorbar(lc, ax=axis)
        axis.set_title(prefix)
        for n in rag.nodes:
            axis.text(*reversed(rag.nodes[n]['centroid']), rag.nodes[n]['labels'][0], fontsize=3, color='red')


    def build_graph(self, segments, segment_colours):
        # build the graph and add labels and mean colours to nodes with distances as weights
        logger.debug("Build adjacency graph")
        rag = graph.RAG(segments)

        for n in rag.nodes():
            rag.nodes[n].update({
                "labels": [n],
                "mean colour": segment_colours[n] if n > 0 else (0, 0, 0)
            })
        logger.debug("add distance in colour as weights")
        for x, y, d in list(rag.edges(data=True)):
            d['delta_e'] = deltaE_ciede2000(rag.nodes[x]['mean colour'], rag.nodes[y]['mean colour'], kL=2)

        logger.debug("remove edges connected to 0 node (background)")
        background_edges = list(rag.edges(0))
        rag.remove_edges_from(background_edges)


        fig = FigureBuilder(self.filepath, "Regional Adjacency Graph", nrows = 2) if self.args.debug else None
        if fig:
            self.subplot_adjacency(fig.get_current_subplot(), rag, segments, prefix="Weighted graph")
            fig.finish_subplot()

        logger.debug(f"remove edges above {self.args.graph_dist} delta_e")
        for x, y, d in list(rag.edges(data=True)):
            if d['delta_e'] > self.args.graph_dist:
                rag.remove_edge(x, y)

        if fig:
            self.subplot_adjacency(fig.get_current_subplot(), rag, segments, prefix="Pruned by delta E")
            fig.finish_subplot()
            fig.print()

        return rag

    def get_segments(self, circles, n_segments=1000, compactness=10):
        circles_mask = self.get_circles_mask(circles)
        logger.debug("Perform SLIC for superpixel identification")
        segments = slic(
            self.rgb,
            # this slic implementation has broken behaviour when using the existing transformation and convert2lab=false
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
            #self.circle_id_map[tuple(circle)] = circle_segment_ids
            for i in list(circle_segment_ids):
                circles_per_segment[i] += 1
                if circles_per_segment[i] > 1 or i == 0 :  # add a new segment for this ID in this circle, always relabel if part of background but inside circle
                    segment_counter += 1
                    ## caution: this behaviour breaks get_target_mask if circles actually overlap
                    segments[circle_segments == i] = segment_counter
                    #self.circle_id_map[tuple(circle)].remove(i)

                    #self.circle_id_map[tuple(circle)].add(segment_counter)


        if self.args.debug:
            fig = FigureBuilder(self.filepath, "Superpixel labels", nrows=2)
            fig.add_image(label2rgb(segments, self.rgb, kind='avg'), "Labels (average)")
            fig.add_image(label2rgb(segments, self.rgb), "Labels (false colour)")
            fig.print()

        return segments

    def get_mean_colours_and_distances(self, segments, kl=2, target=True):
        if target:
            target_lab = np.array(self.args.target_colour)
            label = "Delta e from any target colour"
        else:
            target_lab = np.array(self.args.non_target_colour)
            label = "Delta e from non-target colour"
        segment_colours = {r.label: r.mean_intensity for r in regionprops(segments, self.lab)}  # does not include 0 region
        target_distances = {x: min([deltaE_ciede2000(segment_colours[x], target, kL=kl) for target in target_lab]) for x in segment_colours.keys()}
        if self.args.debug:
            fig = FigureBuilder(self.filepath, f"Superpixel distance from {'target' if target else 'non-target'}")
            dist_image = segments.copy()
            for k in target_distances.keys():
                dist_image[segments == k] = target_distances[k]
            fig.add_image(dist_image, label, color_bar = True, diverging=True, midpoint = self.args.target_dist if target else self.args.non_target_dist)
            fig.print()

        return segment_colours, target_distances


    def get_target_mask_without_graph(self, segments, target_segments, non_target_segments):
        target_mask = np.logical_and(
            np.isin(segments, target_segments), np.isin(segments, non_target_segments, invert=True)
        )
        if self.args.debug:
            fig = FigureBuilder(
                self.filepath,
                "Target mask",
                nrows=3
            )
            fig.add_image(np.isin(segments, target_segments), "target")
            fig.add_image(np.isin(segments, non_target_segments), "non-target")
            fig.add_image(target_mask, "target mask")
        return target_mask

    # kl is a weighting applied to deltaE that makes it more closely resemble human colour differentiation
    # this has improved segmentation in my experience but may not always be desirable
    def get_target_mask_with_graph(self, circles, segments, segment_colours, target_segments, non_target_segments):
        # First merge non-target segments with background
        segment_colours = {k: segment_colours[k] for k in segment_colours.keys() if k not in non_target_segments}
        segments[np.isin(segments, non_target_segments)] = 0

        # Create a regional adjacency graph with weights based on distance of a and b in Lab colourspace
        rag = self.build_graph(segments, segment_colours)
        starting_segments = set()
        collected_segments = set()

        ## Now only look within circles...is this necessary? why not just apply to the whole image since we masked the background that isn't in the image already...
        #for circle in circles:
        #    for n in self.circle_id_map[tuple(circle)]:
        #        ## caution: this can fail if circles overlap (see logic in get_segments)
        #        if n in rag.nodes and n in target_segments:
        #            starting_segments.add(n)
        #            collected_segments.update(node_connected_component(rag, n))
        for n in target_segments:
            if n in rag.nodes():  # if node is both target and background then it will be removed from the graph
                starting_segments.add(n)
                collected_segments.update(node_connected_component(rag, n))

        target_mask = np.isin(segments, np.array(list(collected_segments)))
        if self.args.debug:
            fig = FigureBuilder(self.filepath, "Target mask", nrows=2)
            # create mask from this region
            starting_mask = np.isin(segments, np.array(list(starting_segments)))
            fig.add_image(starting_mask, "Starting node mask")
            fig.add_image(target_mask, "Target mask")
            fig.print()

        return target_mask

    def get_target_mask(self, circles, target_dist, non_target_dist):
        segments = self.get_segments(
            circles,
            n_segments=self.args.num_superpixels,
            compactness=self.args.superpixel_compactness
        )
        _, non_target_distances = self.get_mean_colours_and_distances(segments, target=False)
        segment_colours, target_distances = self.get_mean_colours_and_distances(segments, target=True)
        # add 1 to get 1 based indexing which matches the segment ID
        target_segments = [k for k in target_distances.keys() if target_distances[k] <= target_dist]
        non_target_segments = [k for k in target_distances.keys() if non_target_distances[k] <= non_target_dist]
        if self.args.graph_dist == 0:
            target_mask = self.get_target_mask_without_graph(segments, target_segments, non_target_segments)
        else:
            target_mask = self.get_target_mask_with_graph(
                circles,
                segments,
                segment_colours,
                target_segments,
                non_target_segments
            )

        fig = FigureBuilder(
                self.filepath,
                "Fill mask",
                nrows=len([i for i in [self.args.remove, self.args.fill] if i])
        ) if self.args.debug else None
        if self.args.remove:
            logger.debug("Remove small objects in the mask")
            target_mask = remove_small_objects(target_mask, self.args.remove)
            if fig:
                fig.add_image(target_mask, "Removed small objects")
        if self.args.fill:
            logger.debug("Fill small holes in the mask")
            target_mask = remove_small_holes(target_mask, self.args.fill)
            if fig:
                fig.add_image(target_mask, "Filled small holes")
                fig.print()
        return target_mask

    def get_area(self):

        plates = Layout(self.lab, self.filepath, self.args).get_plates_sorted()
        result = {
            "filename": self.filepath,
            "units": []
        }

        all_circles = [c for p in plates for c in p.circles]
        target_mask = self.get_target_mask(
            all_circles,
            target_dist=self.args.target_dist,
            non_target_dist=self.args.non_target_dist
        )

        if self.args.overlay or self.args.debug:
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
                if self.args.overlay or self.args.debug:
                    logger.debug(f"Join target to overlay mask: {p.id}")
                    unit = j + 1 + 6 * (p.id - 1)
                    # draw the outer circle
                    x = c[0]
                    y = c[1]
                    r = c[2]
                    #xx, yy = draw.circle_perimeter_aa(x, y, r, shape=circle_mask.shape)
                    draw_tool.text((x,y), str(unit), "blue", small_font)
                    draw_tool.ellipse((x-r, y-r, x+r, y+r), outline=(255, 255, 0), fill=None, width=5)
            if self.args.overlay or self.args.debug:
                logger.debug(f"Annotate overlay with plate ID: {p.id}")
                draw_tool.text(p.centroid, str(p.id), "red", large_font)
        if self.args.overlay or self.args.debug:
            fig = FigureBuilder(self.filepath, "Overlay", force="save")
            fig.add_image(annotated_image)
            fig.print()
        return result  # todo consider replacing cv with skimage here too
