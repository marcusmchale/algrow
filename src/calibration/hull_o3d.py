import numpy as np
import logging
import argparse

from trimesh import PointCloud
from alphashape import optimizealpha, alphashape
import open3d as o3d

from typing import Optional
from ..image_loading import ImageLoaded
from .loading import CalibrationImage

logger = logging.getLogger(__name__)


class HullHolder:
    def __init__(
            self,
            points: np.ndarray,
            alpha: Optional[float] = None
    ):
        self.points = points
        self.alpha = None
        self.hull = None
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.scene: Optional[o3d.t.geometry.RaycastingScene] = None

        self.update_alpha(alpha)

    def update_alpha(self, alpha: float = None):
        if alpha is None:
            if len(self.points) >= 4:
                logger.debug(f"optimising alpha")
                self.alpha = 1/round(optimizealpha(self.points), ndigits=3)
                logger.info(f"optimised alpha: {self.alpha}")
            else:
                logger.debug(f"Insufficient points selected for automated alpha optimisation")
        else:
            self.alpha = alpha
        self.update_hull()

    def update_hull(self):
        if len(self.points) < 4:
            self.scene = None
            self.mesh = None
            return
        else:
            logger.debug("Constructing hull")
            if self.alpha is None or self.alpha == 0:
                logger.debug("creating convex hull")
                # the api for alphashape is a bit strange,
                # it returns a shapely polygon when alpha is 0
                # rather than a trimesh object which is returned for other values of alpha
                # so just calculate the convex hull with trimesh to ensure we get a consistent return value
                self.hull = PointCloud(self.points).convex_hull
            else:
                logger.debug("Constructing alpha shape")
                # note the alphashape package uses the inverse of the alpha radius as alpha
                self.hull = alphashape(self.points, 1/self.alpha)
                if len(self.hull.faces) == 0:
                    logger.debug("More points required for a closed hull with current alpha value")
                    self.hull = None
                    self.scene = None
                    self.mesh = None
                    return

        self.scene = o3d.t.geometry.RaycastingScene()
        self.mesh = o3d.geometry.TriangleMesh(self.hull.as_open3d)
        tmesh = o3d.t.geometry.TriangleMesh().from_legacy(mesh_legacy=self.mesh)
        self.scene.add_triangles(tmesh)

    def get_distances(self, points: np.ndarray):
        if self.scene is not None:
            logger.debug(f"Prepare tensor of all points")
            points_tensor = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
            logger.debug(f"Get distances from hull")
            distances = self.scene.compute_signed_distance(points_tensor)
            logger.debug(f"Convert distances to flat array")
            distances = distances.numpy().reshape(-1)
            logger.debug("Return distances")
            return distances
        else:
            return None


def hull_from_mask(image: CalibrationImage, mask: ImageLoaded, alpha, min_pixels):
    image = image
    mask_bool = np.all(mask.rgb == [1, 1, 1], axis=2)
    if not np.sum(mask_bool):
        logger.debug("No white pixels found in provided mask image")
        return None
    logger.debug(f"White pixels: {np.sum(mask_bool)}")
    target = image.lab[mask_bool]
    round_target = image.args.colour_rounding * np.round(target / image.args.colour_rounding)
    unique_target, counts = np.unique(round_target, axis=0, return_counts=True)
    colours = unique_target[counts >= min_pixels]
    return HullHolder(colours, alpha)
    #vertices = hh.hull.vertices
    #hull_vertices_string = f'{[",".join([str(j) for j in i]) for i in vertices]}'.replace("'", '"')
    #logger.debug(f"Hull vertices from mask: {hull_vertices_string}")
    #args.hull_vertices = vertices