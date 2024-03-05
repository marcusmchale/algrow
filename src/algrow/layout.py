import logging
import numpy as np
from skimage import draw

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
    def __init__(self, plates, shape):
        if plates is None:
            raise InsufficientPlateDetection("No plates detected")
        self.plates = plates
        self.shape = shape

    @property
    def circles(self):
        return np.array([c for p in self.plates for c in p.circles])

    def get_circle_mask(self, circle):
        x = circle[0]
        y = circle[1]
        radius = circle[2]
        circle_mask = np.full(self.shape, False)
        yy, xx = draw.disk((y, x), radius, shape=self.shape)
        circle_mask[yy, xx] = True
        return circle_mask.astype('bool')


