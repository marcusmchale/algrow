import logging
import numpy as np
import cv2 as cv2
from scipy.cluster import hierarchy
from matplotlib import pyplot
from copy import copy


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
        self.image = image
        self.debugger = debugger

    def blur(self):
        self.logger.debug("Blur image")
        self.image = cv2.medianBlur(self.image, self.args.kernel)
        self.debugger.debug_image(self.image, "Median blur")

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
            self.debugger.debug_image(circle_debug, f"Circles found with param2 = {param2}")
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
            self.logger.debug("Output dendrogram and treecut height for plate clustering")
            self.logger.debug("Create empty figure")
            fig = pyplot.figure()
            self.logger.debug("Create dendrogram")
            hierarchy.dendrogram(dendrogram)
            self.logger.debug("Add cut-height line")
            pyplot.axhline(y=cut_height, c='k')
            self.debugger.debug_plot(pyplot, "dendrogram")
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

    @staticmethod
    def sort_plates(plates):
        plates.sort(key=lambda x: (not x.vertical, x.centroid[0]))
        for i, p in enumerate(plates):
            p.plate_id = i + 1
            if p.vertical:
                # first split the plate into left and right
                p.circles = p.circles[p.circles[:, 0].argsort()][::-1]
                right = p.circles[0:3]
                right = right[right[:, 1].argsort()][::-1]
                left = p.circles[3:6]
                left = left[left[:, 1].argsort()][::-1]
                p.circles = np.concatenate((left, right))
            else:
                # first split the plate into left, middle and right
                p.circles = p.circles[p.circles[:, 0].argsort()][::-1]
                right = p.circles[0:2]
                right = right[right[:, 1].argsort()][::-1]
                middle = p.circles[2:4]
                middle = middle[middle[:, 1].argsort()][::-1]
                left = p.circles[4:6]
                left = left[left[:, 1].argsort()][::-1]
                p.circles = np.concatenate((left, middle, right))
        return plates

    def get_plates(self):
        plates = self.find_plates()
        if plates is None:
            raise ImageContentException("The plate layout could not be detected")
        return self.sort_plates(plates)

