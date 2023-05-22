import logging
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.morphology import remove_small_holes, remove_small_objects, binary_dilation
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.color import label2rgb, rgb2lab, lab2rgb, deltaE_cie76
from skimage import draw
from skimage.io import imread
from skimage import graph
from networkx import node_connected_component
from re import search
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from .layout import LayoutDetector
from .figurebuilder import FigureBuilder
from .picker import Picker
from collections import defaultdict


logger = logging.getLogger(__name__)

class OverlappingCircles(Exception):
    pass


def median_intensity(mask, intensity_image):
    return np.median(intensity_image[mask], axis=0)

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
                raise OverlappingCircles("Circles overlapping - try again with a lower circle_expansion factor")
            circles_mask = circles_mask | circle_mask
        if self.args.debug:
            fig = FigureBuilder(self.filepath, "Circles mask")
            fig.add_image(circles_mask)
            fig.print()
        return circles_mask


    def build_graph(self, segments, segment_colours):
        # build the graph and add labels and mean colours to nodes with distances as weights
        logger.debug("Build adjacency graph")
        rag = graph.RAG(segments)

        for n in rag.nodes():
            rag.nodes[n].update({
                "labels": [n],
                "mean colour": segment_colours.loc[n].values if n > 0 else (0, 0, 0)
            })
        logger.debug("add distance in colour as weights")
        for x, y, d in list(rag.edges(data=True)):
            d['delta_e'] = deltaE_cie76(rag.nodes[x]['mean colour'], rag.nodes[y]['mean colour'])

        logger.debug("remove edges connected to 0 node (background)")
        background_edges = list(rag.edges(0))
        rag.remove_edges_from(background_edges)


        fig = FigureBuilder(self.filepath, "Regional Adjacency Graph", nrows = 2) if self.args.debug else None
        if fig:
            fig.plot_adjacency(rag, segments, self.rgb, prefix='Weighted graph')

        logger.debug(f"remove edges above {self.args.graph_dist} delta_e")
        for x, y, d in list(rag.edges(data=True)):
            if d['delta_e'] > self.args.graph_dist:
                rag.remove_edge(x, y)

        if fig:
            fig.plot_adjacency(rag, segments, self.rgb, prefix='Pruned by delta E')
            fig.print()

        return rag

    def get_segments(self, layout):
        n_segments = self.args.num_superpixels
        compactness = self.args.superpixel_compactness

        circles_mask = self.get_circles_mask(layout.circles)

        logger.debug("Perform SLIC for superpixel identification")
        segments = slic(
            self.rgb,
            # this slic implementation has broken behaviour when using the existing lab transformation and convert2lab=false
            # just allowing this function to redo conversion from rgb
            # todo work out what this is doing differently when using the already converted lab image
            # could it be scaling?
            mask=circles_mask,
            n_segments=n_segments,
            compactness=compactness,
            # enforce_connectivity = True,
            convert2lab=True #
        )

        # The below doesn't seem needed any more.. what is the issue if a segment spans multiple circles?
        # The slic output may include segment labels that span circles which breaks graph building.
        # Clean it up by iterating through circles and relabel segments if found in another circle
        #logger.debug('Clean up segments spanning multiple circles')
        #circles_per_segment = defaultdict(int)
        #segment_counter = np.max(segments)
        #for circle in layout.circles:
        #    circle_mask = self.get_circle_mask(circle)
        #    circle_segments = segments.copy()
        #    circle_segments[~circle_mask] = -1 # to differentiate the background in circle from background outside
        #    circle_segment_ids = set(np.unique(circle_segments))
        #    circle_segment_ids.remove(-1)
        #    for i in list(circle_segment_ids):
        #        circles_per_segment[i] += 1
        #        if circles_per_segment[i] > 1 or i == 0 : #
        #            # add a new segment for this ID in this circle
        #            segment_counter += 1
        #            segments[circle_segments == i] = segment_counter
        #

        if self.args.debug:
            # add text label for each segment at the centroid for that superpixel
            regions = regionprops(segments, self.lab, extra_properties=(median_intensity,))
            segment_centroid = pd.DataFrame(
                data=[np.array(r.centroid) for r in regions],
                index=[r.label for r in regions],
                columns=['y', 'x']
            )
            invert_segment_colour_rgb = pd.DataFrame(
                data=1-lab2rgb([np.array(r.median_intensity) for r in regions]),
                index=segment_centroid.index,
                columns=['R', 'G', 'B']
            )
            fig = FigureBuilder(self.filepath, "Superpixel labels", nrows=2)
            fig.add_image(label2rgb(segments, self.rgb, kind='avg'), "Labels (average)")
            for i, r in segment_centroid.iterrows():
                fig.current_axis.text(
                    x=r['x'],
                    y=r['y'],
                    s=i,
                    size=2,
                    color=invert_segment_colour_rgb.loc[i],
                    horizontalalignment='center',
                    verticalalignment='center'
                )
            fig.add_image(label2rgb(segments, self.rgb), "Labels (false colour)")
            for i, r in segment_centroid.iterrows():
                fig.current_axis.text(
                    x=r['x'],
                    y=r['y'],
                    s=i,
                    size=2,
                    color='white',
                    horizontalalignment='center',
                    verticalalignment='center'
                )
            fig.print(large=True)
        return segments

    def get_segment_colours(self, segments):
        regions = regionprops(segments, self.lab, extra_properties=(median_intensity,))
        return pd.DataFrame(
            data = [np.array(r.median_intensity) for r in regions],
            index=[r.label for r in regions],
            columns = ['L', 'a', 'b']
        )

    def get_segment_distances(self, segments, segment_colours, reference_label, reference_colours, reference_dist):
        reference_colours = np.array(reference_colours)
        distances = pd.DataFrame(
            data = np.array([deltaE_cie76(segment_colours, r) for r in reference_colours]).transpose(),
            index = segment_colours.index,
            columns = [str(r) for r in reference_colours]
        )
        if self.args.debug:
            fig = FigureBuilder(
                self.filepath,
                f"Segment colours in CIELAB with {reference_label}"
            )
            # convert segment colours to rgb to colour scatter points
            segment_colours_rgb = pd.DataFrame(
                data = lab2rgb(segment_colours),
                index = segment_colours.index,
                columns = ['R','G','B']
            )
            fig.add_subplot(projection = '3d')
            fig.current_axis.scatter(
                xs=segment_colours['a'],
                ys=segment_colours['b'],
                zs=segment_colours['L'],
                s=10,
                c=segment_colours_rgb,
                lw=0
            )
            fig.current_axis.set_xlabel('a')
            fig.current_axis.set_ylabel('b')
            fig.current_axis.set_zlabel('L')
            for i, r in segment_colours.iterrows():
                fig.current_axis.text(x=r['a'], y=r['b'], z=r['L'], s=i, size=3)

            # todo add sphere for each target
            # draw sphere
            def plt_spheres(ax, list_center, list_radius):
                for c, r in zip(list_center, list_radius):
                    # draw sphere
                    # adapted from https://stackoverflow.com/questions/64656951/plotting-spheres-of-radius-r
                    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
                    x = r * np.cos(u) * np.sin(v)
                    y = r * np.sin(u) * np.sin(v)
                    z = r * np.cos(v)
                    ax.plot_surface(x + c[1], y + c[2], z + c[0], color=lab2rgb(c), alpha=0.2)

            plt_spheres(fig.current_axis, reference_colours, np.repeat(reference_dist, len(reference_colours)))
            fig.animate()
            fig.print()

            fig = FigureBuilder(self.filepath, f"Segment ΔE from any {reference_label} colour")
            distances_copy = distances.copy()
            distances_copy.loc[0] = np.repeat(0, distances_copy.shape[1]) # add back a distance for background segments
            distances_copy.sort_index(inplace=True)
            dist_image = distances_copy.min(axis=1).to_numpy()[segments]
            fig.add_image(dist_image, f"Minimum ΔE any {reference_label} target colour", color_bar = True, diverging=True, midpoint = reference_dist)
            fig.print()


            # add debug for EACH reference colour in a single figure, to help debug reference colour selection
            fig = FigureBuilder(self.filepath, f"Segment ΔE from each {reference_label} colour", nrows=len(reference_colours))
            for i, c in enumerate(reference_colours):
                dist_image = distances_copy[str(c)].to_numpy()[segments]
                fig.add_image(
                    dist_image,
                    str(c),
                    color_bar=True,
                    diverging=True,
                    midpoint=reference_dist
                )
            fig.print()
        return distances


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

    def get_target_mask_with_graph(self, segments, segment_colours, target_segments, non_target_segments):
        # First merge non-target segments with background
        segments[np.isin(segments, non_target_segments)] = 0
        # and we can drop the unused segment colours
        #segment_colours = segment_colours[~segment_colours.index.isin(non_target_segments)]

        # Create a regional adjacency graph with weights based on distance of a and b in Lab colourspace
        rag = self.build_graph(segments, segment_colours)
        starting_segments = set()
        collected_segments = set()

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

    def get_target_mask(self, segments):

        segment_colours = self.get_segment_colours(segments)

        non_target_distances = self.get_segment_distances(segments, segment_colours, 'non-target', self.args.non_target_colour, self.args.non_target_dist)
        target_distances = self.get_segment_distances(segments, segment_colours, 'target', self.args.target_colour, self.args.target_dist)

        target_segments = target_distances.index[target_distances.min(axis=1) <= self.args.target_dist].tolist()
        non_target_segments = non_target_distances.index[non_target_distances.min(axis=1) <= self.args.non_target_dist].tolist()

        if self.args.graph_dist == 0:
            target_mask = self.get_target_mask_without_graph(segments, target_segments, non_target_segments)
        else:
            target_mask = self.get_target_mask_with_graph(
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
        layout = LayoutDetector(self.rgb, self.lab, self.filepath, self.args).get_layout()

        result = {
            "filename": self.filepath,
            "units": []
        }

        segments = self.get_segments(layout)
        segment_colours = self.get_segment_colours(segments)

        # Target colour(s) for input images
        if not self.args.target_colour:
           logger.debug("Pick target colours")
           # get colour from image
           picker = Picker(self.rgb, self.lab, self.filepath, "target", args = self.args)
           target_colours = picker.get_reference_colours(segments, segment_colours, self.args.target_dist, 'Target')
           vars(self.args).update({"target_colour":[(c[0], c[1], c[2]) for c in target_colours]})


        # Non-target colour(s) for input images
        if not self.args.non_target_colour:
           logger.debug("Pick non-target colours")
           # get colour from a random image
           picker = Picker(self.rgb, self.lab, self.filepath, "non-target", args = self.args)
           non_target_colours = picker.get_reference_colours(segments, segment_colours, self.args.non_target_dist, 'Non-target')
           vars(self.args).update({"non_target_colour":[(c[0], c[1], c[2]) for c in non_target_colours]})


        if self.args.debug:
            fig = FigureBuilder(self.filepath, "Target colours")
            fig.plot_colours(vars(self.args)['target_colour'])
            fig.print()

            fig = FigureBuilder(self.filepath, "Non-target colours")
            fig.plot_colours(vars(self.args)['non_target_colour'])
            fig.print()

        target_mask = self.get_target_mask(segments)

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
        for p in layout.plates:
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
