import logging

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.cluster import hierarchy
from skimage import draw
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks


from matplotlib import pyplot as plt


from skimage.color import deltaE_cie76
from .image_loading import ImageLoaded, ImageFilepathAdapter
from .figurebuilder import FigureBase

logger = logging.getLogger(__name__)


class ExcessPlatesException(Exception):
    pass


class InsufficientCircleDetection(Exception):
    pass


class InsufficientPlateDetection(Exception):
    pass



class Plate:
    def __init__(self, cluster_id, circles, plate_id=None, centroid=None):
        self.cluster_id: int = cluster_id
        self.circles = circles
        if centroid is None:
            self.centroid = tuple(np.uint16(self.circles[:, 0:2].mean(axis=0)))
        else:
            self.centroid = centroid
        self.id = plate_id


class Layout:
    def __init__(self, plates, image: ImageLoaded):
        self.logger = ImageFilepathAdapter(logger, {"image_filepath": image.filepath})
        if plates is None:
            raise InsufficientPlateDetection("No plates detected")
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
            circles_mask = circles_mask | circle_mask
        if overlapping_circles:
            self.logger.info("Circles overlapping")
        fig = self.image.figures.new_figure("Circles mask")
        fig.plot_image(circles_mask)
        fig.print()
        self._mask = circles_mask

    def _draw_overlay(self):
        self.logger.debug("Draw overlay image")
        fig = self.image.figures.new_figure("Layout overlay", level="WARN")
        fig.plot_image(self.image.rgb)
        unit = 0
        for p in self.plates:
            logger.debug(f"Processing plate {p.id}")
            fig.add_label(str(p.id), p.centroid, "black", 10)
            for j, c in enumerate(p.circles):
                unit += 1
                fig.add_label(str(unit), (c[0], c[1]), "black", 5)
                fig.add_circle((c[0], c[1]), c[2])
        self._overlay = fig.as_array()
        #fig.print()  # we are forcing the level higher for use in app so don't print by default


class LayoutDetector:
    def __init__(
            self,
            image: ImageLoaded
    ):
        self.image = image
        self.args = image.args

        self.logger = ImageFilepathAdapter(logger, {"image_filepath": image.filepath})
        self.logger.debug(f"Detect layout for: {self.image.filepath}")

        circles_like = np.full_like(self.image.lab, self.args.circle_colour)
        self.distance = deltaE_cie76(self.image.lab, circles_like)

        fig = self.image.figures.new_figure("Circle distance")
        fig.plot_image(self.distance, "ΔE from circle colour", color_bar=True)
        fig.print()

    def hough_circles(self, image, hough_radii):
        self.logger.debug(f"Find circles with radii: {hough_radii}")
        edges = canny(image, sigma=3, low_threshold=10, high_threshold=20)
        return hough_circle(edges, hough_radii)

    def find_n_circles(self, n, fig=None, allowed_overlap=0.1):
        circle_radius_px = int(self.args.circle_diameter / 2)
        radius_range = int(self.args.circle_variability * circle_radius_px)
        hough_radii = np.arange(
            circle_radius_px - radius_range,  # start
            circle_radius_px + radius_range,  # stop
            max(1, int(radius_range/5))  # step size from radius range for more consistent processing time
            # todo : find a better way to determine an appropriate step size
        )
        self.logger.debug(f"Radius range to search for: {np.min(hough_radii), np.max(hough_radii)}")
        hough_result = self.hough_circles(self.distance, hough_radii)
        min_distance = int(self.args.circle_diameter * (1 - allowed_overlap))
        self.logger.debug(f"minimum distance allowed between circle centers: {min_distance}")
        _accum, cx, cy, rad = hough_circle_peaks(
            hough_result,
            hough_radii,
            min_xdistance=min_distance,
            min_ydistance=min_distance
        )
        # note the expansion factor appplied below to increase the search area for mask/superpixels
        self.logger.debug(f"mean detected circle radius: {np.around(np.mean(rad), decimals=0)}")
        circles = np.dstack(
            (cx, cy, np.repeat(int((self.args.circle_diameter/2)*(1+self.args.circle_expansion)), len(cx)))
        ).squeeze(axis=0)

        fig.plot_image(self.image.rgb, f"Circle detection")
        for c in circles:
            fig.add_circle((c[0], c[1]), c[2])

        if circles.shape[0] < n:
            self.logger.debug(f'{str(circles.shape[0])} circles found')
            fig.plot_text("Insufficient circles detected")
            raise InsufficientCircleDetection

        self.logger.debug(
            f"{str(circles.shape[0])} circles found")
        return circles

    def find_plate_clusters(self, circles, cluster_size, n, fig: FigureBase):
        if cluster_size == 1:
            logger.debug("Clusters of size one cannot be filtered")
            return range(len(circles)), range(len(circles))
        centres = np.delete(circles, 2, axis=1)
        cut_height = int(
            (self.args.circle_diameter + self.args.circle_separation) * (1 + self.args.circle_separation_tolerance)
        )
        self.logger.debug(f"cut height: {cut_height}")
        self.logger.debug("Create dendrogram of centre distances (linkage method)")
        dendrogram = hierarchy.linkage(centres)
        fig.plot_dendrogram(dendrogram, cut_height, label="Plate clusters")
        self.logger.debug(f"Cut the dendrogram and select clusters containing {cluster_size} centre points only")
        clusters = hierarchy.cut_tree(dendrogram, height=cut_height)
        unique, counts = np.array(np.unique(clusters, return_counts=True))
        target_clusters = unique[[i for i, j in enumerate(counts.flat) if j == cluster_size]]
        self.logger.debug(f"Found {len(target_clusters)} plates")
        if len(target_clusters) < n:
            raise InsufficientPlateDetection(f"Only {len(target_clusters)} plates found")
        elif len(target_clusters) > n:
            raise ExcessPlatesException(f"More than {n} plates found")
        return clusters.flatten(), target_clusters

    def find_plates(self):
        fig = self.image.figures.new_figure("Detect plates", cols=2)
        circles = self.find_n_circles(self.args.circles_per_plate * self.args.plates, fig)
        clusters, target_clusters = self.find_plate_clusters(
            circles,
            self.args.circles_per_plate,
            self.args.plates,
            fig=fig
        )
        fig.print()
        self.logger.debug("Collect circles from target clusters into plates")
        plates = [
            Plate(
                cluster_id,
                circles[[i for i, j in enumerate(clusters) if j == cluster_id]],
            ) for cluster_id in target_clusters
        ]
        return plates

    def get_axis_clusters(self, axis_values, cut_height, fig, plate_id=None):
        if len(axis_values) == 1:
            return np.array([0])
        dendrogram = hierarchy.linkage(axis_values.reshape(-1, 1))
        fig.plot_dendrogram(dendrogram, cut_height, label=f"Plate: {plate_id}" if plate_id else None)
        return hierarchy.cut_tree(dendrogram, height=cut_height)

    def sort_plates(self, plates):
        plates_rows_first = not self.args.plates_cols_first
        plates_left_right = not self.args.plates_right_left
        plates_top_bottom = not self.args.plates_bottom_top
        circles_rows_first = not self.args.circles_cols_first
        circles_left_right = not self.args.circles_right_left
        circles_top_bottom = not self.args.circles_bottom_top

        if len(plates) > 1:
            self.logger.debug("Sort plates")
            # First the plates themselves

            axis_values = np.array([p.centroid[int(plates_rows_first)] for p in plates])
            plate_clustering_fig = self.image.figures.new_figure(f"Plate {'row' if plates_rows_first else 'col'} clustering")
            cut_height = self.args.plate_width * 0.5
            clusters = self.get_axis_clusters(axis_values, cut_height, plate_clustering_fig)
            plate_clustering_fig.print()
            clusters = DataFrame(
                {
                    "cluster": clusters.flatten(),
                    "plate": plates,
                    "primary_axis": [p.centroid[int(plates_rows_first)] for p in plates],
                    "secondary_axis": [p.centroid[int(not plates_rows_first)] for p in plates]
                }
            )
            clusters = clusters.sort_values(
                "primary_axis", ascending=plates_top_bottom if plates_rows_first else plates_left_right
            ).groupby("cluster", sort=False, group_keys=True).apply(
                lambda x: x.sort_values("secondary_axis", ascending=plates_left_right if plates_rows_first else plates_top_bottom)
            )
            plates = clusters.plate.values

            # Now for circles within plates

            within_plate_fig = self.image.figures.new_figure(f"Within plate {'row' if circles_rows_first else 'col'} clustering")
            for i, p in enumerate(plates):
                p.id = i + 1
                self.sort_circles(plates[i], within_plate_fig, circles_rows_first, circles_left_right, circles_top_bottom)
            within_plate_fig.print()
            return plates.tolist()
        else:
            plates[0].id = 1
            within_plate_fig = self.image.figures.new_figure(f"Within plate {'row' if circles_rows_first else 'col'} clustering")
            self.sort_circles(plates[0], within_plate_fig, circles_rows_first, circles_left_right, circles_top_bottom)
            within_plate_fig.print()
            return plates

    def sort_circles(self, plate, fig: FigureBase, rows_first=True, left_right=True, top_bottom=True):
        if len(plate.circles) == 1:
            return

        # sometimes rotation is significant such that the clustering fails.
        # correct this by getting the rotation angle
        # get the two closest points to each corner origin (top left)
        self.logger.debug(f"sort circles for plate {plate.id}")

        def get_rotation(a, b):
            # get the angle of b relative to a
            # https://math.stackexchange.com/questions/1201337/finding-the-angle-between-two-points
            # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
            diff_xy = b - a
            deg = np.rad2deg(np.arctan2(*diff_xy))
            # we want to align vertically if closer to 0 degrees or horizontally if closer to 90 degrees
            if abs(deg) > 45:
                deg = deg - 90
            return deg

        sorted_from_origin = plate.circles[:, 0:2][np.argsort(np.linalg.norm(plate.circles[:, 0:2], axis=1))]
        if len(sorted_from_origin) == 2:
            rot_deg = get_rotation(*sorted_from_origin)
        else:
            rotations = list()
            # We are basically collecting from each corner, the presumed rotations assuming it is a square corner
            # by having a few guesses we can handle some errors in the layout and/or circle detection
            # we then take the median of these guesses.
            rotations.append(get_rotation(sorted_from_origin[0], sorted_from_origin[1]))
            rotations.append(get_rotation(sorted_from_origin[0], sorted_from_origin[2]))
            rotations.append(get_rotation(sorted_from_origin[-2], sorted_from_origin[-1]))
            rotations.append(get_rotation(sorted_from_origin[-3], sorted_from_origin[-1]))
            flipped_y = plate.circles.copy()[:, 0:2]
            flipped_y[:, 1] = self.image.rgb.shape[0] - flipped_y[:, 1]
            flipped_y_sorted_from_origin = flipped_y[np.argsort(np.linalg.norm(flipped_y, axis=1))]
            rotations.append(np.negative(get_rotation(flipped_y_sorted_from_origin[0], flipped_y_sorted_from_origin[1])))
            rotations.append(np.negative(get_rotation(flipped_y_sorted_from_origin[0], flipped_y_sorted_from_origin[2])))
            rotations.append(np.negative(get_rotation(flipped_y_sorted_from_origin[-2], flipped_y_sorted_from_origin[-1])))
            rotations.append(np.negative(get_rotation(flipped_y_sorted_from_origin[-3], flipped_y_sorted_from_origin[-1])))
            logger.debug(f"Calculated rotation suggestions: {rotations}")
            rot_deg = np.median(rotations)
            
        logger.debug(f"Rotate plate {plate.id} before row clustering by {rot_deg}")

        def rotate(points, origin, degrees=0):
            # https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
            angle = np.deg2rad(degrees)
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]])
            o = np.atleast_2d(origin)
            p = np.atleast_2d(points)
            return np.squeeze((rotation_matrix @ (p-o).T + o.T).T)
        
        rotated_coords = rotate(plate.circles[:, 0:2], origin=sorted_from_origin[0], degrees=rot_deg)

        cut_height = int(self.args.circle_diameter * 0.25)  # this seems like a suitable value
        axis_values = np.array([int(c[int(rows_first)]) for c in rotated_coords])
        clusters = self.get_axis_clusters(axis_values, cut_height, fig, plate_id=plate.id)

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
        plates = self.find_plates()
        plates = self.sort_plates(plates)
        return Layout(plates, self.image)


class LayoutLoader:
    def __init__(
            self,
            image: ImageLoaded
    ):
        self.image = image
        self.args = image.args

        self.logger = ImageFilepathAdapter(logger, {"image_filepath": image.filepath})
        self.logger.debug(f"Load layout for: {self.image.filepath}")

    def get_layout(self):
        layout_path = self.args.fixed_layout
        df = pd.read_csv(layout_path, index_col=["plate_id", "circle_id"])
        if self.args.downscale != 1:
            df = df.divide(self.args.downscale)
        df.sort_index(ascending=True)
        plates = list()
        for plate_id in df.index.get_level_values("plate_id").unique():
            plates.append(
                Plate(
                    cluster_id=plate_id,
                    circles=list(
                        df.loc[plate_id][["circle_x", "circle_y", "circle_radius"]].itertuples(index=False, name=None)
                    ),
                    plate_id=plate_id,
                    centroid=tuple(df.loc[plate_id,1][["plate_x", "plate_y"]].values)
                )
            )
        #fig = self.image.figures.new_figure("Loaded layout")
        #fig.plot_image(self.distance, "ΔE from circle colour", color_bar=True)
        #fig.print() #  todo add figure for loaded layout, maybe just the mask?
        return Layout(plates, self.image)


def get_layout(image, fig):
    plt.close()  # in case existing plots
    layout_detector = LayoutDetector(image)
    logger.debug(f"find circles: {image.args.circles_per_plate * image.args.plates}")
    circles = layout_detector.find_n_circles(image.args.circles_per_plate * image.args.plates, fig)
    logger.debug(f"cluster into plates: {image.args.plates}")
    clusters, target_clusters = layout_detector.find_plate_clusters(
        circles,
        image.args.circles_per_plate,
        image.args.plates,
        fig=fig
    )
    plates = [
        Plate(
            cluster_id,
            circles[[i for i, j in enumerate(clusters) if j == cluster_id]],
        ) for cluster_id in target_clusters
    ]
    logger.debug("Sort plates")
    plates = layout_detector.sort_plates(plates)
    return circles, plates, fig
