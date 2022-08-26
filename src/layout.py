import logging
import pdb

import numpy as np
from pandas import DataFrame
import cv2 as cv2
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt
from copy import copy
from enum import IntEnum
from typing import TYPE_CHECKING
from typing import List





class ImageContentException(Exception):
    pass


class InsufficientCircleDetection(Exception):
    pass


class InsufficientClusterDetection(Exception):
    pass


class Plate:
    def __init__(self, cluster_id, circles):
        self.cluster_id: int = cluster_id
        self.circles = circles
        self.centroid = tuple(np.uint16(self.circles[:, 0:2].mean(axis=0)))
        x_range = np.max(self.circles[:, 0]) - np.min(self.circles[:, 0])
        y_range = np.max(self.circles[:, 1]) - np.min(self.circles[:, 1])
        self.vertical = y_range > x_range
        self.plate_id = None


class Layout:
    def __init__(self, args, image, debugger):
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.image = cv2.medianBlur(image, self.args.kernel)
        self.debugger = debugger
        self.debugger.render_image(self.image, "Median blur")

    def find_circles(self, image, param2=30):
        self.logger.debug("Find circles")
        circle_radius_px = int((self.args.scale * self.args.circle_diameter)/2)
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(circle_radius_px*2),
            param1=20,
            param2=param2,
            minRadius=int(circle_radius_px * (1-self.args.circle_radius_tolerance)),
            maxRadius=int(circle_radius_px * (1+self.args.circle_radius_tolerance))
        )
        return circles

    def find_n_circles(self, param2, n=48):
        param2 = int(param2)
        circles = self.find_circles(self.image, param2)
        if circles is None:
            self.logger.debug(f"0 circles found with param2 = {param2}")
            raise InsufficientCircleDetection
        circles = np.squeeze(np.uint16(np.around(circles)))
        if circles.shape[0] < n:
            self.logger.debug(f'{str(circles.shape[0])} circles found with param2 = {param2}')
            raise InsufficientCircleDetection
        if self.args.image_debug:
            circle_debug = copy(self.image)
            for c in circles:
                cv2.circle(circle_debug, (c[0], c[1]), c[2], (255, 0, 0), 5)
            self.debugger.render_image(circle_debug, f"Circles found with param2 = {param2}")
        self.logger.debug(
            f"{str(circles.shape[0])} circles found with param2 = {param2}")
        return circles

    def find_n_clusters(self, circles, cluster_size=6, n=8):
        args = self.args
        centres = np.delete(circles, 2, axis=1)
        cut_height = int(args.scale * args.circle_diameter * (1 + args.cut_height_tolerance))
        self.logger.debug("Create dendrogram of centre distances (linkage method)")
        dendrogram = hierarchy.linkage(centres)
        if args.image_debug:
            self.logger.debug("Output dendrogram and treecut height for circle clustering")
            fig, ax = plt.subplots()
            self.logger.debug("Create dendrogram")
            hierarchy.dendrogram(dendrogram)
            self.logger.debug("Add cut-height line")
            plt.axhline(y=cut_height, c='k')
            self.debugger.render_plot("Dendrogram")
        self.logger.debug(f"Cut the dendrogram and select clusters containing {cluster_size} centre points only")
        clusters = hierarchy.cut_tree(dendrogram, height=cut_height)
        unique, counts = np.array(np.unique(clusters, return_counts=True))
        target_clusters = unique[[i for i, j in enumerate(counts.flat) if j == cluster_size]]
        self.logger.debug(f"Found {len(target_clusters)} clusters")
        if len(target_clusters) < n:
            raise InsufficientClusterDetection
        elif len(target_clusters) > n:
            raise ImageContentException(f"More than {n} clusters found")
        return clusters, target_clusters

    def find_plates(self):
        args = self.args
        self.logger.debug(
            "Find circles, raising param2 until we find enough"
        )
        ph_list = [args.param2] if args.param2 else range(50, 1, -5)
        for param2 in ph_list:
            try:
                circles = self.find_n_circles(param2)
            except InsufficientCircleDetection:
                continue
            try:
                clusters, target_clusters = self.find_n_clusters(circles)
            except InsufficientClusterDetection:
                continue

            self.logger.debug("Collect circles from target clusters into plates")
            plates = [
                Plate(
                    cluster_id,
                    circles[[i for i, j in enumerate(clusters.flat) if j == cluster_id]],
                ) for cluster_id in target_clusters
            ]
            return plates

    def get_axis_clusters(self, axis_values, rows_first: bool, cut_height):
        dendrogram = hierarchy.linkage(axis_values.reshape(-1, 1))
        if self.args.image_debug:
            self.logger.debug(f"Plot dendrogram for plate {'y' if rows_first else 'x'} axis plate clustering")
            fig, ax = plt.subplots()
            self.logger.debug("Create dendrogram")
            hierarchy.dendrogram(dendrogram)
            self.logger.debug("Add cut-height line")
            plt.axhline(y=cut_height, c='k')
            self.debugger.render_plot("Dendrogram for row index clustering")
        return hierarchy.cut_tree(dendrogram, height=cut_height)

    def sort_circles(self, plate, rows_first=False, left_right=True, top_bottom=False):
        self.logger.debug(f"sort circles for plate {plate.id}")
        cut_height = int(self.args.scale * self.args.circle_diameter * 0.5)
        axis_values = np.array([c[int(rows_first)] for c in plate.circles])
        clusters = self.get_axis_clusters(axis_values, rows_first, cut_height=cut_height)
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
        ).groupby("cluster", sort=False).apply(
            lambda x: x.sort_values("secondary_axis", ascending=left_right if rows_first else top_bottom)
        )
        plate.circles = clusters.circle.tolist()

    def sort_plates(self, plates, rows_first=True, left_right=True, top_bottom=False):
        # default is just what our lab was already doing, a better default would be top_bottom = True
        self.logger.debug("Sort plates")
        axis_values = np.array([p.centroid[int(rows_first)] for p in plates])
        # todo consider getting cut height from plate specification
        clusters = self.get_axis_clusters(axis_values, rows_first, cut_height=100)
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
        ).groupby("cluster", sort=False).apply(
            lambda x: x.sort_values("secondary_axis", ascending=left_right if rows_first else top_bottom)
        )
        plates = clusters.plate.values
        for i, p in enumerate(plates):
            p.id = i+1
            self.sort_circles(p)  # todo, pass arguments from options for customising layout
        return plates.tolist()

    def get_plates_sorted(self):
        plates = self.find_plates()
        if plates is None:
            raise ImageContentException("The plate layout could not be detected")
        return self.sort_plates(plates)  # todo, pass arguments from options for customising layout

