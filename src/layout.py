import argparse
import logging

import numpy as np
from pandas import DataFrame
from scipy.cluster import hierarchy
from skimage import draw
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from .figurebuilder import FigureBuilder
from skimage.morphology import binary_dilation
from skimage.color import deltaE_cie76
from .image_loading import ImageLoaded
from .logging import CustomAdapter

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

logger = logging.getLogger(__name__)


class ImageContentException(Exception):
    pass


class InsufficientCircleDetection(Exception):
    pass


class InsufficientPlateDetection(Exception):
    pass


class OverlappingCircles(Exception):
    pass


class Plate:
    def __init__(self, cluster_id, circles):
        self.cluster_id: int = cluster_id
        self.circles = circles
        self.centroid = tuple(np.uint16(self.circles[:, 0:2].mean(axis=0)))
        self.plate_id = None


class Layout:
    def __init__(self, plates, image):
        self.logger = CustomAdapter(logger, {'image_filepath': str(image.filepath)})
        if plates is None:
            raise ImageContentException("The layout could not be detected")
        self.plates = plates
        self.image = image
        self.args = image.args
        self.dim = image.rgb.shape[0:2]
        self._mask = None
        self._overlay = None

    @property
    def circles(self):
        return np.array([c for p in self.plates for c in p.circles])

    @property
    def mask(self):
        if self._mask is None:
            self._draw_mask()
        return self._mask

    @property
    def overlay(self):
        if self._overlay is None:
            self._draw_overlay()
        return self._overlay

    def get_circle_mask(self, circle):
        x = circle[0]
        y = circle[1]
        radius = circle[2]
        circle_mask = np.full(self.dim, False)
        yy, xx = draw.disk((y, x), radius, shape=self.dim)
        circle_mask[yy, xx] = True
        return circle_mask.astype('bool')

    def _draw_mask(self):
        self.logger.debug("Draw the circles mask")
        circles_mask = np.full(self.dim, False).astype("bool")
        overlapping_circles = False
        for circle in self.circles:
            circle_mask = self.get_circle_mask(circle)
            if np.logical_and(circles_mask, circle_mask).any():
                overlapping_circles = True
                #raise OverlappingCircles("Circles overlapping - try again with a lower circle_expansion factor")
            circles_mask = circles_mask | circle_mask
        if overlapping_circles:
            self.logger.warning("Circles overlapping")
        if self.args.debug:
            fig = FigureBuilder(self.image.filepath, self.args, "Circles mask")
            fig.add_image(circles_mask)
            fig.print()
        self._mask = circles_mask

    def _draw_overlay(self):
        self.logger.debug("Prepare annotated overlay for testing layout")
        blended = self.image.rgb.copy()
        annotated_image = Image.fromarray(blended)
        draw_tool = ImageDraw.Draw(annotated_image)
        height = self.image.rgb.shape[0]
        font_file = font_manager.findfont(font_manager.FontProperties())
        large_font = ImageFont.truetype(font_file, size=int(height/50), encoding="unic")
        small_font = ImageFont.truetype(font_file, size=int(height/80), encoding="unic")
        for p in self.plates:
            for j, c in enumerate(p.circles):
                unit = j + 1 + 6 * (p.id - 1)
                # draw the outer circle
                x = c[0]
                y = c[1]
                r = c[2]
                draw_tool.text((x, y), str(unit), "blue", small_font)
                draw_tool.ellipse((x-r, y-r, x+r, y+r), outline=(255, 255, 0), fill=None, width=5)
            draw_tool.text(p.centroid, str(p.id), "red", large_font)
        self._overlay = annotated_image


class LayoutDetector:
    def __init__(
            self,
            image: ImageLoaded
    ):
        self.image = image
        self.args = image.args

        self.logger = CustomAdapter(logger, {'image_filepath': str(image.filepath)})
        self.logger.debug(f"Detect layout for: {self.image.filepath}")

        circles_like = np.full_like(self.image.lab, self.args.circle_colour)
        self.distance = deltaE_cie76(self.image.lab, circles_like)

        if self.args.debug:
            fig = FigureBuilder(self.image.filepath, self.args, "Circle distance")
            fig.add_image(self.distance, "ΔE from circle colour", color_bar=True)
            fig.print()

    def hough_circles(self, image, hough_radii):
        self.logger.debug(f"Find circles with radii: {hough_radii}")
        edges = canny(image, sigma=3, low_threshold=10, high_threshold=20)
        return hough_circle(edges, hough_radii)

    def find_n_circles(self, n, attempt=0, fig=None, allowed_overlap=0.3):
        num_circles = int(n + n * attempt)
        self.logger.debug(f"find {num_circles} circles")
        circle_radius_px = int(self.args.circle_diameter / 2)
        # each attempt we expand the number of radii to assess
        hough_radii = np.arange(circle_radius_px - (attempt + 1), circle_radius_px + (attempt + 1), 1)
        hough_result = self.hough_circles(self.distance, hough_radii)
        min_distance = int(self.args.circle_diameter * (1 - allowed_overlap))
        self.logger.debug(f"minimum distance between circle centers: {min_distance}")
        _accum, cx, cy, rad = hough_circle_peaks(
            hough_result,
            hough_radii,
            min_xdistance=min_distance,
            min_ydistance=min_distance,
            num_peaks=num_circles  # each time we increase the target peak number by the number sought
            # we are concurrently expanding the radii considered
        )
        # note the expansion factor appplied below to increase the search area for mask/superpixels
        self.logger.debug(f"mean detected circle radius: {np.around(np.mean(rad), decimals=0)}")
        circles = np.dstack(
            (cx, cy, np.repeat(int((self.args.circle_diameter/2)*self.args.circle_expansion), len(cx)))
        ).squeeze()
        if circles.shape[0] < n:
            self.logger.debug(f'{str(circles.shape[0])} circles found')
            raise InsufficientCircleDetection

        if fig:
            circle_debug = np.zeros_like(self.distance, dtype="bool")
            self.logger.debug("draw circles")
            for circle in circles:
                x = circle[0]
                y = circle[1]
                radius = circle[2]
                yy, xx = draw.circle_perimeter(y, x, radius, shape=circle_debug.shape)
                circle_debug[yy, xx] = 255
            circle_debug = binary_dilation(circle_debug, np.full((10, 10), 1, dtype="bool"))  # slow
            # todo find faster way to draw thick lines
            # consider skimage.segmentation.mark_boundaries
            overlay = np.copy(self.image.rgb)
            overlay[circle_debug] = 255
            fig.add_image(overlay, f"Attempt {attempt + 1}")
        self.logger.debug(
            f"{str(circles.shape[0])} circles found")
        return circles

    def find_n_clusters(self, circles, cluster_size, n, fig=None):
        centres = np.delete(circles, 2, axis=1)
        #cut_height_expansion = 1 + ((attempt+1)/2)
        #logger.debug(f"cut height expansion: {cut_height_expansion}")
        # explore other methods as the tree cut height is a bit fragile
        cut_height = int((self.args.circle_diameter + self.args.plate_circle_separation) * self.args.cut_height_expansion)
        self.logger.debug(f"cut height: {cut_height}")
        self.logger.debug("Create dendrogram of centre distances (linkage method)")
        dendrogram = hierarchy.linkage(centres)
        if fig:
            ax = fig.add_subplot()
            self.logger.debug("Output dendrogram and treecut height for circle clustering")
            hierarchy.dendrogram(dendrogram, ax=ax)
            self.logger.debug("Add cut-height line")
            ax.axhline(y=cut_height, c='k')
            ax.set_title("Dendrogram")
        self.logger.debug(f"Cut the dendrogram and select clusters containing {cluster_size} centre points only")
        clusters = hierarchy.cut_tree(dendrogram, height=cut_height)
        unique, counts = np.array(np.unique(clusters, return_counts=True))
        target_clusters = unique[[i for i, j in enumerate(counts.flat) if j == cluster_size]]
        self.logger.debug(f"Found {len(target_clusters)} plates")
        if len(target_clusters) < n:
            raise InsufficientPlateDetection(f"Only {len(target_clusters)} plates found")
        elif len(target_clusters) > n:
            raise ImageContentException(f"More than {n} plates found")
        return clusters, target_clusters

    def find_plates(self, fig=None):
        for i in range(5):  # try 5 times to find enough circles to make plates
            try:
                circles = self.find_n_circles(self.args.circles_per_plate * self.args.n_plates, i, fig)
            except InsufficientCircleDetection:
                continue
            try:
                clusters, target_clusters = self.find_n_clusters(
                    circles,
                    self.args.circles_per_plate,
                    self.args.n_plates,
                    fig=fig
                )
            except InsufficientPlateDetection:
                self.logger.debug(f"Try again with detection of more circles")
                if fig:
                    fig.add_subplot_row()
                continue
            if fig is not None:
                fig.print()
            self.logger.debug("Collect circles from target clusters into plates")
            plates = [
                Plate(
                    cluster_id,
                    circles[[i for i, j in enumerate(clusters.flat) if j == cluster_id]],
                ) for cluster_id in target_clusters
            ]
            return plates
        raise InsufficientPlateDetection(f"Insufficient plates detected - consider modifying the layout configuration")

    def get_axis_clusters(self, axis_values, rows_first: bool, cut_height, plate_id=None, fig=None):
        dendrogram = hierarchy.linkage(axis_values.reshape(-1, 1))
        if fig:
            ax = fig.add_subplot()
            self.logger.debug(f"Plot dendrogram for {'rows' if rows_first else 'cols'} axis plate {plate_id} clustering")
            self.logger.debug("Create dendrogram")
            hierarchy.dendrogram(dendrogram,  ax=ax)
            self.logger.debug("Add cut-height line")
            ax.axhline(y=cut_height, c='k')
            if plate_id:
                ax.set_title(f"Plate {plate_id}")
        return hierarchy.cut_tree(dendrogram, height=cut_height)

    def sort_plates(self, plates):
        rows_first = not self.args.plates_cols_first
        left_right = not self.args.plates_right_left
        top_bottom = not self.args.plates_bottom_top

        if not plates:
            return None
        if len(plates) == 1:
            p = plates[0]
            p.id = 1
            fig = FigureBuilder(
                self.image.filepath,
                self.args,
                f"Circle clustering by {'rows' if rows_first else 'cols'}"
            ) if self.args.debug else None
            self.sort_circles(
                p,
                rows_first=not self.args.circles_cols_first,
                left_right=not self.args.circles_right_left,
                top_bottom=not self.args.circles_bottom_top,
                fig=fig
            )
            if fig:
                fig.print()
            return plates
        self.logger.debug("Sort plates")
        axis_values = np.array([p.centroid[int(rows_first)] for p in plates])
        fig = FigureBuilder(
            self.image.filepath,
            self.args,
            f"Plate clustering by {'rows' if rows_first else 'cols'}"
        ) if self.args.debug else None
        cut_height = self.args.plate_width * 0.5
        clusters = self.get_axis_clusters(
            axis_values,
            rows_first,
            cut_height=cut_height,
            fig=fig
        )
        if fig:
            fig.print()
        clusters = DataFrame(
            {
                "cluster": clusters.flatten(),
                "plate": plates,
                "primary_axis": [p.centroid[int(rows_first)] for p in plates],
                "secondary_axis": [p.centroid[int(not rows_first)] for p in plates]
            }
        )
        clusters = clusters.sort_values(
            "primary_axis", ascending=top_bottom if rows_first else left_right
        ).groupby("cluster", sort=False, group_keys=True).apply(
            lambda x: x.sort_values("secondary_axis", ascending=left_right if rows_first else top_bottom)
        )
        plates = clusters.plate.values
        fig = FigureBuilder(
            self.image.filepath,
            self.args,
            f"Circle clustering by {'rows' if rows_first else 'cols'}"
        ) if self.args.debug else None
        first = True
        for i, p in enumerate(plates):
            if first:
                first = False
            elif fig:
                fig.add_subplot_row()
            p.id = i + 1
            self.sort_circles(
                p,
                rows_first=not self.args.circles_cols_first,
                left_right=not self.args.circles_right_left,
                top_bottom=not self.args.circles_bottom_top,
                fig=fig
            )
        if fig:
            fig.print()
        return plates.tolist()

    def sort_circles(self, plate, rows_first=True, left_right=True, top_bottom=True, fig=None):
        self.logger.debug(f"sort circles for plate {plate.id}")
        cut_height = int(self.args.circle_diameter * 0.5)
        axis_values = np.array([c[int(rows_first)] for c in plate.circles])
        clusters = self.get_axis_clusters(
            axis_values,
            rows_first,
            cut_height=cut_height,
            plate_id=plate.id,
            fig=fig
        )
        clusters = DataFrame(
            {
                "cluster": clusters.flatten(),
                "circle": plate.circles.tolist(),
                "primary_axis": [c[int(rows_first)] for c in plate.circles],
                "secondary_axis": [c[int(not rows_first)] for c in plate.circles]
            }
        )
        clusters = clusters.sort_values(
            "primary_axis", ascending=top_bottom if rows_first else left_right
        ).groupby("cluster", sort=False, group_keys=True).apply(
            lambda x: x.sort_values("secondary_axis", ascending=left_right if rows_first else top_bottom)
        )
        plate.circles = clusters.circle.tolist()

    def get_layout(self):
        fig = FigureBuilder(self.image.filepath, self.args, "Plate detection", ncols=2) if self.args.debug else None
        plates = self.find_plates(fig)
        plates = self.sort_plates(plates)
        return Layout(plates, self.image)
