import numpy as np
import logging
import itertools
import open3d as o3d

from trimesh import PointCloud, Trimesh, repair
from alphashape import optimizealpha, alphasimplices
from typing import Optional
from .image_loading import CalibrationImage

logger = logging.getLogger(__name__)


class HullHolder:
    def __init__(
            self,
            points: np.ndarray,
            alpha: Optional[float] = None
    ):
        self.points = points
        self.alpha = alpha
        self.hull = None
        self.mesh: Optional[o3d.t.geometry.TriangleMesh] = None
        self.scene: Optional[o3d.t.geometry.RaycastingScene] = None

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
        logger.debug(f"Update hull with  {len(self.points)} vertices")
        if len(self.points) < 4:
            self.scene = None
            self.mesh = None
            return
        else:
            logger.debug("Constructing hull")
            if self.alpha is None or self.alpha == 0:
                logger.debug(f"Constructing convex hull")
                # the api for alphashape is a bit strange,
                # it returns a shapely polygon when alpha is 0
                # rather than a trimesh object which is returned for other values of alpha
                # so just calculate the convex hull with trimesh to ensure we get a consistent return value
                try:
                    self.hull = PointCloud(self.points).convex_hull
                except Exception as e:
                    logger.debug("Error during convex hull construction")
                    raise e
            else:
                logger.debug("Constructing alpha hull")
                # note the alphashape package uses the inverse of the alpha radius as alpha
                # self.hull = alphashape(self.points, 1/self.alpha)
                # the below was adapted to modify behaviour from the above function in the alpha shape package
                edges = set()
                perimeter_edges = set()
                coords = np.array(self.points)
                for point_indices, circumradius in alphasimplices(coords):
                    if circumradius <= self.alpha:
                        for edge in itertools.combinations(point_indices, r=coords.shape[-1]):
                            if all([e not in edges for e in itertools.combinations(edge, r=len(edge))]):
                                edges.add(edge)
                                perimeter_edges.add(edge)
                            else:
                                perimeter_edges -= set(itertools.combinations(edge, r=len(edge)))
                self.hull = Trimesh(vertices=coords, faces=list(perimeter_edges))
                repair.fix_normals(self.hull)

                if not self.hull.is_watertight:
                    logger.warning("Hull is not watertight")
                if len(self.hull.faces) == 0:
                    logger.debug("More points required for a closed hull with current alpha value")
                    self.hull = None
                    self.scene = None
                    self.mesh = None
                    return
        self.update_scene()

    def update_scene(self):
        self.scene = o3d.t.geometry.RaycastingScene()
        #self.mesh = o3d.geometry.TriangleMesh(self.hull.as_open3d)
        # see https://github.com/mikedh/trimesh/issues/1116
        # todo keep an eye on this issue in case trimesh changes around this
        #tmesh = o3d.t.geometry.TriangleMesh().from_legacy(mesh_legacy=self.mesh)
        self.mesh = o3d.t.geometry.TriangleMesh().from_legacy(
            mesh_legacy=o3d.geometry.TriangleMesh(self.hull.as_open3d)
        )
        self.scene.add_triangles(self.mesh)

    def get_occupancy(self, points: np.ndarray) -> Optional[np.ndarray]:
        if self.scene is not None:
            logger.debug(f"Get occcupancy in hull")
            points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
            occupancy = self.scene.compute_occupancy(points_tensor)
            occupancy.to(o3d.core.Dtype.Bool).numpy()
            return occupancy.to(o3d.core.Dtype.Bool).numpy()
        else:
            return None

    def get_distances(self, points: np.ndarray) -> Optional[np.ndarray]:
        if self.scene is not None:
            logger.debug(f"Get distances from hull")
            points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
            distances = self.scene.compute_signed_distance(points_tensor)
            return distances.numpy()
        else:
            return None

    @staticmethod
    def get_from_mask(image: CalibrationImage, alpha, min_pixels, voxel_size):
        if image.true_mask is None:
            return None

        mask_bool = image.true_mask.mask
        if not np.sum(mask_bool):
            logger.debug("No true values found in provided mask")
            return None
        logger.debug(f"White pixels: {np.sum(mask_bool)}")
        idx = np.argwhere(mask_bool)
        target = image.lab[idx[:, 0], idx[:, 1], :]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(target)
        cloud, _, indices = cloud.voxel_down_sample_and_trace(voxel_size, min_bound=[0, -128, -128], max_bound=[100, 127, 127])
        common_indices = [i for i, j in enumerate(indices) if len(j) >= min_pixels]
        cloud = cloud.select_by_index(common_indices)
        colours = cloud.points
        return HullHolder(colours, alpha)
