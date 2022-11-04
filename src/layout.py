import logging
import numpy as np
from pandas import DataFrame
from scipy.cluster import hierarchy
from skimage import filters, draw
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from matplotlib import pyplot as plt
from skimage.morphology import binary_dilation
from .options import options

logger = logging.getLogger(__name__)
args = options()

class ImageContentException(Exception):
    pass


class InsufficientCircleDetection(Exception):
    pass


class InsufficientPlateDetection(Exception):
    pass


class Plate:
    def __init__(self, cluster_id, circles):
        self.cluster_id: int = cluster_id
        self.circles = circles
        self.centroid = tuple(np.uint16(self.circles[:, 0:2].mean(axis=0)))
        self.plate_id = None


class Layout:
    def __init__(
            self,
            image,
            debugger
    ):
        self.image = filters.median(image, np.full((3, 3), args.kernel, dtype=int))
        self.debugger = debugger
        self.debugger.render_image(self.image, "Median blur")

    @staticmethod
    def hough_circles(image, hough_radii):
        logger.debug("Find circles")
        edges = canny(image, sigma=3, low_threshold=10, high_threshold=20)
        return hough_circle(edges, hough_radii)

    def find_n_circles(self, n):
        logger.debug(f"find {n} circles")
        circle_radius_px = int(args.circle_diameter / 2)
        hough_radii = np.arange(circle_radius_px - 10, circle_radius_px + 10, 2)
        hough_result = self.hough_circles(self.image, hough_radii)
        _accum, cx, cy, rad = hough_circle_peaks(
            hough_result,
            hough_radii,
            min_xdistance=int(args.circle_diameter),
            min_ydistance=int(args.circle_diameter),
            num_peaks=n
        )
        circles = np.dstack((cx, cy, rad)).squeeze()
        if circles.shape[0] < n:
            logger.debug(f'{str(circles.shape[0])} circles found ')
            raise InsufficientCircleDetection
        if args.image_debug:
            circle_debug = np.zeros_like(self.image, dtype="bool")
            logger.debug("draw circles")
            for circle in circles:
                x = circle[0]
                y = circle[1]
                radius = circle[2]
                yy, xx = draw.circle_perimeter(y, x, radius, shape=circle_debug.shape)
                circle_debug[yy, xx] = 255
            circle_debug = binary_dilation(circle_debug, np.full((10, 10), 1, dtype="bool"))  # very slow
            # todo find faster way to draw thick lines
            overlay = np.copy(self.image)
            overlay[circle_debug] = 255
            self.debugger.render_image(overlay, f"Circles found")

        logger.debug(
            f"{str(circles.shape[0])} circles found")
        return circles

    def find_n_clusters(self, circles, cluster_size, n):
        centres = np.delete(circles, 2, axis=1)
        cut_height = int(args.circle_diameter * (1 + args.cut_height_tolerance))
        logger.debug("Create dendrogram of centre distances (linkage method)")
        dendrogram = hierarchy.linkage(centres)
        if args.image_debug:
            logger.debug("Output dendrogram and treecut height for circle clustering")
            fig, ax = plt.subplots()
            logger.debug("Create dendrogram")
            hierarchy.dendrogram(dendrogram)
            logger.debug("Add cut-height line")
            plt.axhline(y=cut_height, c='k')
            self.debugger.render_plot("Dendrogram")
        logger.debug(f"Cut the dendrogram and select clusters containing {cluster_size} centre points only")
        clusters = hierarchy.cut_tree(dendrogram, height=cut_height)
        unique, counts = np.array(np.unique(clusters, return_counts=True))
        target_clusters = unique[[i for i, j in enumerate(counts.flat) if j == cluster_size]]
        logger.debug(f"Found {len(target_clusters)} plates")
        if len(target_clusters) < n:
            raise InsufficientPlateDetection(f"Only {len(target_clusters)} plates found")
        elif len(target_clusters) > n:
            raise ImageContentException(f"More than {n} plates found")
        return clusters, target_clusters

    def find_plates(self, n_plates, n_per_plate):
        for i in range(5):  # try 5 times to find enough circles to make plates
            try:
                circles = self.find_n_circles(n_per_plate * n_plates)
            except InsufficientCircleDetection:
                continue
            try:
                clusters, target_clusters = self.find_n_clusters(circles, n_per_plate, n_plates)
            except InsufficientPlateDetection:
                continue

            logger.debug("Collect circles from target clusters into plates")
            plates = [
                Plate(
                    cluster_id,
                    circles[[i for i, j in enumerate(clusters.flat) if j == cluster_id]],
                ) for cluster_id in target_clusters
            ]
            return plates

    def get_axis_clusters(self, axis_values, rows_first: bool, cut_height, plate_id=None):
        dendrogram = hierarchy.linkage(axis_values.reshape(-1, 1))
        if args.image_debug:
            logger.debug(f"Plot dendrogram for {'rows' if rows_first else 'cols'} axis plate {plate_id} clustering")
            fig, ax = plt.subplots()
            logger.debug("Create dendrogram")
            hierarchy.dendrogram(dendrogram)
            logger.debug("Add cut-height line")
            plt.axhline(y=cut_height, c='k')
            plate_id_text = f"plate {plate_id}" if plate_id else "plates"
            self.debugger.render_plot(f"Dendrogram for {'rows' if rows_first else 'cols'} clustering for {plate_id_text}")
        return hierarchy.cut_tree(dendrogram, height=cut_height)

    def sort_circles(self, plate, rows_first=True, left_right=True, top_bottom=True):
        logger.debug(f"sort circles for plate {plate.id}")
        cut_height = int(args.circle_diameter * 0.5)
        axis_values = np.array([c[int(rows_first)] for c in plate.circles])
        clusters = self.get_axis_clusters(axis_values, rows_first, cut_height=cut_height, plate_id=plate.id)
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

    def sort_plates(self, plates, rows_first=True, left_right=True, top_bottom=True):
        # default is currently just what our lab is used to using, a better default would be top_bottom = True
        logger.debug("Sort plates")
        axis_values = np.array([p.centroid[int(rows_first)] for p in plates])
        # todo consider getting cut height from some plate specification, half the width of plate would do (in px)
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
        ).groupby("cluster", sort=False, group_keys=True).apply(
            lambda x: x.sort_values("secondary_axis", ascending=left_right if rows_first else top_bottom)
        )
        plates = clusters.plate.values
        for i, p in enumerate(plates):
            p.id = i+1
            self.sort_circles(
                p,
                rows_first=not args.circles_cols_first,
                left_right=not args.circles_right_left,
                top_bottom=not args.circles_bottom_top
            )

        return plates.tolist()

    def get_plates_sorted(self):
        plates = self.find_plates(args.n_plates,  args.circles_per_plate)
        if plates is None:
            raise ImageContentException("The plate layout could not be detected")
        return self.sort_plates(
            plates,
            rows_first=not args.plates_cols_first,
            left_right=not args.plates_right_left,
            top_bottom=not args.plates_bottom_top
        )

