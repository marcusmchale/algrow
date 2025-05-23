import argparse
import logging

import numpy as np
import pandas as pd
import open3d as o3d

from pathlib import Path
from copy import deepcopy

from skimage.io import imread, imsave
from skimage import draw
from skimage.util import img_as_bool, img_as_ubyte, img_as_float64 as img_as_float
# https://github.com/isl-org/Open3D/issues/4832
# Float 64 is better for the point cloud,
# but we are casting back to 32 bit for the distance calculations...
# this might be worthy of further consideration/optimisation,
# but probably just need to track the development of Open3D
from skimage.color import rgb2lab, rgb2gray, gray2rgb, deltaE_cie76
from skimage.transform import hough_circle, hough_circle_peaks, downscale_local_mean
from skimage.feature import canny
from skimage.restoration import denoise_bilateral

from scipy.cluster import hierarchy

from .logging import ImageFilepathAdapter
from .options import DebugEnum
from .figurebuilder import FigureBase, FigureMatplot, FigureNone
from .layout import Plate, Layout, ExcessPlatesException, InsufficientPlateDetection, InsufficientCircleDetection

from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ImageLoaded:

    def __init__(self, filepath: Path, args: argparse.Namespace):
        self.args = args
        self.filepath = filepath

        self.logger = ImageFilepathAdapter(logger, {'image_filepath': str(filepath)})

        self.logger.debug(f"Read image from file")
        self.rgb = img_as_float(imread(str(filepath)))
        self.logger.debug(f"Loaded RGB image data type: {self.rgb.dtype}")

        if len(self.rgb.shape) == 2:
            self.logger.info("Grayscale image - converting to RGB")
            # slice off the alpha channel
            self.rgb = gray2rgb(self.rgb)
        elif self.rgb.shape[2] == 4:
            self.logger.info("Removing alpha channel")
            # slice off the alpha channel
            self.rgb = self.rgb[:, :, :3]

        self.figures = ImageFigureBuilder(filepath, args)

        rgb_fig = self.figures.new_figure("RGB image")
        rgb_fig.plot_image(self.rgb, "RGB image")
        rgb_fig.print()

        if self.args.denoise:
            self.logger.debug(f"Denoise")
            self.rgb = denoise_bilateral(self.rgb, channel_axis=-1, sigma_color=1, sigma_spatial=1, mode='edge')
            rgb_denoise_fig = self.figures.new_figure(f"RGB denoised")
            rgb_denoise_fig.plot_image(self.rgb)
            rgb_denoise_fig.print()

        self.logger.debug(f"Convert to Lab")
        self.lab = rgb2lab(self.rgb)
        lab_fig = self.figures.new_figure("Lab channels")
        lab_fig.plot_image(self.lab[:, :, 0], "Lightness channel (L in Lab)", color_bar=True)
        lab_fig.plot_image(self.lab[:, :, 1], "Green-Red channel (a in Lab)", color_bar=True)
        lab_fig.plot_image(self.lab[:, :, 2], "Blue-Yellow channel (b in Lab)", color_bar=True)
        lab_fig.print()

        # downscale the image
        if self.args.downscale != 1:
            self.logger.debug(f"Downscale the RGB input image")
            self.rgb = downscale_local_mean(self.rgb, (self.args.downscale, self.args.downscale, 1))
            downscale_fig = self.figures.new_figure("Downscaled image")
            downscale_fig.plot_image(self.rgb, f"Downscale (factor={self.args.downscale})")
            downscale_fig.print()

            self.lab = downscale_local_mean(self.lab, (self.args.downscale, self.args.downscale, 1))
            lab_fig = self.figures.new_figure("Lab downscaled")
            lab_fig.plot_image(self.lab[:, :, 0], "Lightness channel (L in Lab)", color_bar=True)
            lab_fig.plot_image(self.lab[:, :, 1], "Green-Red channel (a in Lab)", color_bar=True)
            lab_fig.plot_image(self.lab[:, :, 2], "Blue-Yellow channel (b in Lab)", color_bar=True)
            lab_fig.print()

        self.logger.debug("Completed loading")

        self._layout_overlay = None
        self._layout_mask = None
        self.layout = None

    def __hash__(self):
        return hash(self.filepath)

    def __lt__(self, other):
        return self.filepath < other.filepath

    def __le__(self, other):
        return self.filepath <= other.filepath

    def __gt__(self, other):
        return self.filepath > other.filepath

    def __ge__(self, other):
        return self.filepath <= other.filepath

    def __eq__(self, other):
        return self.filepath == other.filepath

    def __ne__(self, other):
        return self.filepath != other.filepath

    def copy(self):
        return deepcopy(self)

    @property
    def layout_overlay(self):
        if self._layout_overlay is None:
            self._draw_layout_overlay()
        return self._layout_overlay

    @property
    def layout_mask(self):
        if self._layout_mask is None:
            self._draw_layout_mask()
        return self._layout_mask

    def _draw_layout_overlay(self):
        if self.layout is None:
            return

        fontsize = np.sqrt(np.multiply(*self.rgb.shape[0:2]))/100
        self.logger.debug("Draw overlay image")
        fig = self.figures.new_figure("Layout overlay", level="WARN")
        fig.plot_image(self.rgb, show_axes=False)
        unit = 0

        for p in self.layout.plates:
            logger.debug(f"Processing plate {p.id}")
            fig.add_label(str(p.id), p.centroid, "black", fontsize*2)
            for j, c in enumerate(p.circles):
                unit += 1
                fig.add_label(str(unit), (c[0], c[1]), "black", fontsize)
                fig.add_circle((c[0], c[1]), c[2], linewidth=fontsize/2)
        self._layout_overlay = fig.as_array(self.rgb.shape[1], self.rgb.shape[0])[:, :, :3]
        #fig.print()

    def _draw_layout_mask(self):
        if self.layout is None:
            self._layout_mask = np.full(self.rgb.shape[:2], 1).astype(dtype="bool")
            return

        self.logger.debug("Draw the circles mask")
        circles_mask = np.full(self.rgb.shape[:2], False).astype("bool")
        overlapping_circles = False
        for circle in self.layout.circles:
            circle_mask = self.layout.get_circle_mask(circle)
            if np.logical_and(circles_mask, circle_mask).any():
                overlapping_circles = True
            circles_mask = circles_mask | circle_mask
        if overlapping_circles:
            self.logger.info("Circles overlapping")
        fig = self.figures.new_figure("Circles mask")
        fig.plot_image(circles_mask)
        self._layout_mask = circles_mask
        fig.print()


class ImageFigureBuilder:
    def __init__(self, image_filepath, args):
        self.counter = 0
        self.image_filepath = image_filepath
        self.args = args
        self.logger = ImageFilepathAdapter(logger, {"image_filepath": image_filepath})
        self.logger.debug("Creating figure builder object")

    def new_figure(self, name, cols=1, level="DEBUG") -> FigureBase:
        if DebugEnum[level] >= self.args.image_debug:
            self.counter += 1
            return FigureMatplot(name, self.counter, self.args, cols=cols, image_filepath=self.image_filepath)
        else:
            return FigureNone(name, self.counter, self.args, cols=cols, image_filepath=self.image_filepath)

    def save_image(self, image, label, level="DEBUG", suffix='png'):
        if DebugEnum[level] >= self.args.image_debug:
            self.counter += 1
            if self.image_filepath is None:
                save_path = Path(self.args.out_dir, "Figures", Path(".".join(["_".join([str(self.counter), label]), suffix])))
            else:
                save_path = Path(self.args.out_dir, "Figures", "ImageAnalysis", Path(
                    ".".join(["_".join([Path(self.image_filepath).stem, str(self.counter), label]), suffix])
                ))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if image.dtype == 'bool':
                image = img_as_ubyte(image)
            imsave(save_path, image, check_contrast=False)
        else:
            logger.debug(f"Not saving image as image debug level is set above the level for this image: {label}")


class MaskLoaded:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        img = imread(str(filepath))
        if img.ndim > 2:
            img = rgb2gray(img)
            if img.ndim > 2:
                logger.debug(f"Attempt to load a mask with the wrong number of dimensions: {img.shape}")
                raise ValueError("Mask must be boolean or greyscale that can be coerced to boolean")
        self.mask = img != 0


class CalibrationImage:  # an adapter to allow zooming and hold other features that are only needed during calibration

    def __init__(self, image: ImageLoaded):
        self._image = image
        self.height, self.width = self.rgb.shape[0:2]

        # cloud is built once but after loading as we want to pickle the image after loading for multiprocessing
        # currently o3d cloud cannot be pickled
        self.cloud = None
        self.voxel_to_image = None  # keep in mind this is indexes in the masked image
        self.voxel_map = np.full(self._image.rgb.shape[:2], -1, dtype=int)

        # true_mask is used to calculate dice coefficient
        # also used to infer points for automated hull construction from a supplied image mask
        self.true_mask: Optional[MaskLoaded] = None  # a boolean mask of self.height * self.width

        self.target_mask: Optional[np.ndarray] = None  # a boolean mask of same shape as image
        self.selected_voxel_indices = set()  # a list of voxel indices selected
        self.selected_circle_indices = set()  # a set of indices in the image that were selected to define circle colour

        self.divisors = np.logspace(0, 10, num=11, base=2, dtype=int)
        self.zoom_index = 0
        self.displayed_start_x = 0
        self.displayed_start_y = 0

    def __hash__(self):
        return hash(self.filepath)

    def __lt__(self, other):
        return self.filepath < other.filepath

    def __le__(self, other):
        return self.filepath <= other.filepath

    def __gt__(self, other):
        return self.filepath > other.filepath

    def __ge__(self, other):
        return self.filepath <= other.filepath

    def __eq__(self, other):
        return self.filepath == other.filepath

    def __ne__(self, other):
        return self.filepath != other.filepath

    @property
    def zoom_factor(self):
        return 1/self.divisors[self.zoom_index]

    @property
    def args(self):
        return self._image.args

    @args.setter
    def args(self, args):
        self._image.args = args

    @property
    def layout(self):
        return self._image.layout

    @layout.setter
    def layout(self, layout):
        self._image.layout = layout

    @property
    def layout_overlay(self):
        return self._image.layout_overlay

    @property
    def layout_mask(self):
        return self._image.layout_mask

    @property
    def filepath(self):
        return self._image.filepath

    @property
    def figures(self):
        return self._image.figures

    def copy(self):
        self._image = self._image.copy()
        return self

    @property
    def lab(self):
        return self._image.lab

    @property
    def rgb(self):
        return self._image.rgb

    def change_layout(self, layout: Optional[Layout] = None):
        self.layout = layout
        self._image._layout_mask = None
        self._image._layout_overlay = None
        self.selected_voxel_indices = set()

    def apply_zoom(self, image):
        if self.zoom_factor != 1:
            logger.debug(f"Zoom on image: {self.zoom_factor}")
            zoomed_width = int(self.zoom_factor * self.width)
            zoomed_height = int(self.zoom_factor * self.height)
            cropped = image[
                      self.displayed_start_y:self.displayed_start_y + zoomed_height,
                      self.displayed_start_x:self.displayed_start_x + zoomed_width
                      ]
            xy_factor = self.divisors[self.zoom_index]
            # kronecker product is much faster than interpolation, and we only want/need perfect zooms
            image = np.kron(cropped, np.ones((xy_factor, xy_factor, 1)))
        return image

    def get_displayed_with_target(
            self,
            target_colour: Optional[Tuple[float, float, float]] = (1.0, 0.0, 0.0),
            selected_colour: Optional[Tuple[float, float, float]] = (0.0, 1.0, 0.0)
    ):
        logger.debug("Prepare displayed image")
        if target_colour is None and selected_colour is None and self.zoom_factor == 1:
            displayed = self.rgb
        else:
            displayed = self.rgb.copy()  # make a copy to mutate

        if target_colour is not None and self.target_mask is not None:
            logger.debug("Highlight pixels within delta of target hull")
            displayed[self.target_mask] = target_colour

        if selected_colour is not None and self.selected_voxel_indices:
            logger.debug("Highlight pixels from selected voxels")
            selected = [j for i in list(self.selected_voxel_indices) for j in np.asarray(self.voxel_to_image[i])]
            displayed.reshape(-1, 3)[selected] = selected_colour

        if self.layout is not None:
            displayed[~self.layout_mask] = (0, 0, 0)

        displayed = self.apply_zoom(displayed)
        logger.debug("Send displayed")
        return o3d.geometry.Image(displayed.astype(np.float32))

    def get_displayed_with_layout(self,):
        if self.layout is not None:
            logger.debug("Get layout overlay")
            displayed = self.apply_zoom(self.layout_overlay)
        else:
            displayed = self.apply_zoom(self.rgb)
        return o3d.geometry.Image(displayed.astype(np.float32))

    def get_displayed_as_circle_distance(self):
        if self.args.circle_colour is not None:
            logger.debug("get distance image")
            circles_like = np.full_like(self.lab, self.args.circle_colour)
            distance = gray2rgb(1-deltaE_cie76(self.lab, circles_like)/255)
            displayed = self.apply_zoom(distance)
        else:
            displayed = self.apply_zoom(self.rgb)
        return o3d.geometry.Image(displayed.astype(np.float32))

    def get_displayed_with_disk(self, x, y, line_colour: Optional[Tuple[float, float, float]] = (1.0, 0.0, 0.0)) -> np.array:
        # returns a copy of the current zoom level with a disk drawn
        displayed = self.rgb.copy()
        displayed = self.apply_zoom(displayed)
        x, y = self.coord_to_zoomed(x, y)
        disk = draw.disk((y, x), 5, shape=displayed.shape)
        displayed[disk] = line_colour
        return o3d.geometry.Image(displayed.astype(np.float32))

    def get_displayed_with_line(
            self, 
            x1,
            y1,
            x2,
            y2,
            line_colour: Optional[Tuple[float, float, float]] = (1.0, 0.0, 0.0)
    ) -> np.array:
        # returns a copy of the current zoom level with a line drawn
        displayed = self.rgb.copy()
        displayed = self.apply_zoom(displayed)
        x1, y1 = self.coord_to_zoomed(x1, y1)
        x2, y2 = self.coord_to_zoomed(x2, y2)
        yy, xx, val = draw.line_aa(y1, x1, y2, x2)
        # remove indices that are out of range in case the start is no longer in frame of displayed
        yy_in = np.argwhere(yy < displayed.shape[0])
        xx_in = np.argwhere(xx < displayed.shape[1])
        keep = np.intersect1d(yy_in, xx_in)
        line_vals = np.multiply.outer(val[keep], line_colour)
        displayed[yy[keep], xx[keep]] = line_vals
        return o3d.geometry.Image(displayed.astype(np.float32))

    # The down-sampling uses a fair bit of memory so moving this out from multi-loading:
    def prepare_cloud(self):
        logger.debug("Prepare voxel cloud")
        self.cloud, self.voxel_to_image = self.get_downscaled_cloud_and_indices()
        # build a reverse mapping from image index to index of voxel in cloud
        logger.debug("Build map from image to voxel")
        self.voxel_map.fill(-1)  # -1 means pixel has no corresponding voxel (masked)
        for i, jj in enumerate(self.voxel_to_image):
            xx, yy = self.pixel_to_coord(jj)
            self.voxel_map[yy, xx] = i

    def get_downscaled_cloud_and_indices(self):  # indices are the back reference to the image pixels
        cloud = o3d.geometry.PointCloud()
        lab = self.lab
        rgb = self.rgb
        if self.layout is not None:
            lab = lab.copy()
            lab[~self.layout_mask] = np.nan
        logger.debug("flatten image")
        lab = lab.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3)
        logger.debug("Set points")
        cloud.points = o3d.utility.Vector3dVector(lab)
        logger.debug("Set point colours")
        cloud.colors = o3d.utility.Vector3dVector(rgb)
        logger.debug("Downsample to voxels")
        cloud, _, indices = cloud.voxel_down_sample_and_trace(
            voxel_size=self._image.args.voxel_size,
            min_bound=[0, -128, -128],
            max_bound=[100, 127, 127]
        )
        logger.debug("Remove the masked values from cloud")
        if self.layout is not None and np.sum(np.isnan(np.sum(cloud.points, axis=1))):
            logger.debug("Get index of nan in cloud")
            # the second test is just a precaution, if we have a layout there *should* always be a voxel containing nan
            nan_cloud_index = np.argmax(np.isnan(np.sum(cloud.points, axis=1)))
            logger.debug("remove from points")
            cloud.points.pop(nan_cloud_index)
            logger.debug("remove from colours")
            cloud.colors.pop(nan_cloud_index)
            logger.debug("remove from indices")
            del indices[nan_cloud_index]
        logger.debug("Return cloud and indices")
        return cloud, indices

    def increment_zoom(self, zoom_increment):
        new_step = self.zoom_index + zoom_increment
        if new_step < 0:
            self.zoom_index = 0
        elif new_step > len(self.divisors) - 1:
            self.zoom_index = len(self.divisors) - 1
        else:
            self.zoom_index += zoom_increment

    def _get_zoom_start(self, x_center, y_center, new_width, new_height):
        if x_center < new_width/2:
            zoom_start_x = 0
        elif x_center > (self.rgb.shape[1] - (new_width/2)):
            zoom_start_x = self.rgb.shape[1] - new_width
        else:
            zoom_start_x = x_center - (new_width/2)
        if y_center < new_height/2:
            zoom_start_y = 0
        elif y_center > (self.rgb.shape[0] - (new_height/2)):
            zoom_start_y = self.rgb.shape[0] - new_height
        else:
            zoom_start_y = y_center - (new_height/2)
        return int(zoom_start_x), int(zoom_start_y)

    def drag(self, x_drag, y_drag):
        logger.debug(f"drag: x = {x_drag}, y = {y_drag}")
        self.displayed_start_x += x_drag
        self.displayed_start_y += y_drag
        if self.displayed_start_x < 0:
            self.displayed_start_x = 0
        if self.displayed_start_y < 0:
            self.displayed_start_y = 0

    def zoom(self, x_center, y_center, zoom_increment: int):
        self.increment_zoom(zoom_increment)
        display_width = int(self.zoom_factor * self.width)
        display_height = int(self.zoom_factor * self.height)
        self.displayed_start_x, self.displayed_start_y = self._get_zoom_start(
            x_center,
            y_center,
            display_width,
            display_height
        )

    def coord_to_zoomed(self, x, y):  # convert from full image x,y to zoomed x, y
        x = int(np.floor((x - self.displayed_start_x) / self.zoom_factor))
        y = int(np.floor((y - self.displayed_start_y) / self.zoom_factor))
        return x, y

    def coord_to_pixel(self, x, y) -> int:  # pixel is the index, coord is x,y
        if all([x >= 0, x < self.rgb.shape[1], y >= 0, y < self.rgb.shape[0]]):
            return (y * self.rgb.shape[1]) + x
        else:
            raise ValueError("Coordinates are outside of image")

    def pixel_to_coord(self, i: int) -> Tuple[int, int]:  # pixel is the index, coord is x,y
        return np.unravel_index(i, self.rgb.shape[0:2])[::-1]
        #return np.unravel_index(i, (self.height, self.width))


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
        # todo add these new args to the gui,
        #  do this at the same time as updating the circle detection GUI so it also shows circle detection
        edges = canny(image, sigma=self.args.canny_sigma, low_threshold=self.args.canny_low, high_threshold=self.args.canny_high)
        return hough_circle(edges, hough_radii)

    def find_n_circles(self, n, fig=None, allowed_overlap=0.1):
        circle_radius_px = int(self.args.circle_diameter / 2)
        radius_range = int(self.args.circle_variability * circle_radius_px)
        step_size = max(1, int(radius_range / 5))  # step size from radius range for more consistent processing time
        hough_radii = np.arange(
            circle_radius_px - radius_range,  # start
            circle_radius_px + radius_range + step_size,  # stop #add step_size to ensure we get the endpoint as well
            step_size
        )
        self.logger.debug(f"Radius range to search for: {np.min(hough_radii), np.max(hough_radii)}")
        hough_result = self.hough_circles(self.distance, hough_radii)
        min_distance = int(self.args.circle_diameter * (1 - allowed_overlap))
        self.logger.debug(f"minimum distance allowed between circle centers: {min_distance}")
        accum, cx, cy, rad = hough_circle_peaks(
            hough_result,
            hough_radii,
            min_xdistance=min_distance,
            min_ydistance=min_distance
        )
        # sort by the accumulator value here, so can preselect the most likely by being the strongest later if needed
        # just in case the result isn't always returned in order,
        # seems to be though, consider removing this additional processing,
        # but check the API for hough_circle_peaks first
        accum_order = np.flip(np.argsort(accum))
        # note the expansion factor appplied below to increase the search area for mask/superpixels
        self.logger.debug(f"mean detected circle radius: {np.around(np.mean(rad), decimals=0)}")
        circles = np.dstack(
            (cx, cy, np.repeat(int((self.args.circle_diameter / 2) * (1 + self.args.circle_expansion)), len(cx)))
        ).squeeze(axis=0)[accum_order]
        # remove detected circles that are not completely within the image frame
        # if x-radius < 0 or x+radius > self.image.rgb.shape[1]
        circles = circles[((circles[:, 0] - circle_radius_px) > 0) & ((circles[:, 0] + circle_radius_px) < self.image.rgb.shape[1])]
        # if y-radius < 0 or y+radius > self.image.rgb.shape[0]
        circles = circles[((circles[:, 1] - circle_radius_px) > 0) & ((circles[:, 1] + circle_radius_px) < self.image.rgb.shape[0])]


        fig.plot_image(self.image.rgb, f"Circle detection")
        for c in circles:
            fig.add_circle((c[0], c[1]), c[2])

        if circles.shape[0] < n:
            self.logger.debug(f'{str(circles.shape[0])} circles found')
            fig.plot_text("Insufficient circles detected")
            fig.print()
            raise InsufficientCircleDetection

        self.logger.debug(
            f"{str(circles.shape[0])} circles found")
        return circles

    def find_plate_clusters(self, circles, cluster_sizes, n_plates, fig: FigureBase):
        if all(np.unique(cluster_sizes) == 1):
            return range(len(circles)), range(0, self.args.circles)

        # find the largest clusters first:
        centres = np.delete(circles, 2, axis=1)
        cut_height = int(
            (self.args.circle_diameter + self.args.circle_separation) * (1 + self.args.circle_separation_tolerance)
        )
        self.logger.debug(f"Create dendrogram of centre distances (linkage method) and cut at: {cut_height}")
        dendrogram = hierarchy.linkage(centres)
        clusters = hierarchy.cut_tree(dendrogram, height=cut_height).flatten()
        unique, counts = np.unique(clusters, return_counts=True)
        fig.plot_dendrogram(dendrogram, cut_height, label="Plate clusters")
        target_cluster_indices = list()
        cluster_sizes.sort(reverse=True)  # make sure we do size 1's last
        n_found = 0
        for n in cluster_sizes:
            logger.debug(f"Find clusters of size: {n}")
            for i, count in enumerate(counts):
                if count == n:
                    if n == 1:
                        logger.debug("Clusters of size one cannot be filtered returning the next best matching circles")
                        # add next best to make total circles
                        # the circles are sorted by best matches, so we can return the top non matching ones.
                        for c in clusters:
                            if n_found == self.args.circles:
                                break
                            if c not in target_cluster_indices and counts[np.argwhere(unique == c)[0]][0] == 1:
                                target_cluster_indices.append(c)
                                n_found += 1
                    else:
                        target_cluster_indices.append(unique[i])
                        n_found += n

        if len(target_cluster_indices) < n_plates:
            fig.print()
            raise InsufficientPlateDetection(f"Only {len(target_cluster_indices)} plates found")
        elif len(target_cluster_indices) > n_plates:
            fig.print()
            raise ExcessPlatesException(f"More than {n_plates} plates found: {len(target_cluster_indices)}")
        return clusters, target_cluster_indices

    def find_plates(self, custom_fig=None):
        if custom_fig is None:
            fig = self.image.figures.new_figure("Detect plates", cols=2)
            print_fig = True
        else:
            logger.debug("")
            fig = custom_fig
            print_fig = False

        circles = self.find_n_circles(self.args.circles, fig)
        clusters, target_clusters = self.find_plate_clusters(
            circles,
            self.args.circles_per_plate,
            self.args.plates,
            fig=fig
        )
        if print_fig:
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
            plate_clustering_fig = self.image.figures.new_figure(
                f"Plate {'row' if plates_rows_first else 'col'} clustering")
            cut_height = self.args.plate_width * 0.5
            clusters = self.get_axis_clusters(axis_values, cut_height, plate_clustering_fig)
            plate_clustering_fig.print()
            clusters = pd.DataFrame(
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
                lambda x: x.sort_values("secondary_axis",
                                        ascending=plates_left_right if plates_rows_first else plates_top_bottom)
            )
            plates = clusters.plate.values

            # Now for circles within plates

            within_plate_fig = self.image.figures.new_figure(
                f"Within plate {'row' if circles_rows_first else 'col'} clustering")
            for i, p in enumerate(plates):
                p.id = i + 1
                self.sort_circles(plates[i], within_plate_fig, circles_rows_first, circles_left_right,
                                  circles_top_bottom)
            within_plate_fig.print()
            return plates.tolist()
        else:
            plates[0].id = 1
            within_plate_fig = self.image.figures.new_figure(
                f"Within plate {'row' if circles_rows_first else 'col'} clustering")
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
            self.logger.debug(f"Get rotation")
            # get the angle of b relative to a
            # https://math.stackexchange.com/questions/1201337/finding-the-angle-between-two-points
            # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
            diff_xy = b - a
            deg = np.rad2deg(np.arctan2(*diff_xy))
            # we want to align vertically if closer to 0 degrees or horizontally if closer to 90 degrees
            if abs(deg) > 45:
                deg = deg - 90
            return deg

        sorted_from_origin = np.array(plate.circles[:, :2][np.argsort(np.linalg.norm(plate.circles[:, :2], axis=1))])
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
            rotations.append(
                np.negative(get_rotation(flipped_y_sorted_from_origin[0], flipped_y_sorted_from_origin[1])))
            rotations.append(
                np.negative(get_rotation(flipped_y_sorted_from_origin[0], flipped_y_sorted_from_origin[2])))
            rotations.append(
                np.negative(get_rotation(flipped_y_sorted_from_origin[-2], flipped_y_sorted_from_origin[-1])))
            rotations.append(
                np.negative(get_rotation(flipped_y_sorted_from_origin[-3], flipped_y_sorted_from_origin[-1])))
            logger.debug(f"Calculated rotation suggestions: {rotations}")
            rot_deg = np.median(rotations)

        logger.debug(f"Rotate plate {plate.id} before row clustering by {rot_deg}")

        def rotate(points, origin, degrees=0):
            # https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
            angle = np.deg2rad(degrees)
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            o = np.atleast_2d(origin)
            p = np.atleast_2d(points)
            return np.squeeze((rotation_matrix @ (p - o).T + o.T).T)

        rotated_coords = rotate(plate.circles[:, 0:2], origin=sorted_from_origin[0], degrees=rot_deg)

        cut_height = int(self.args.circle_diameter * 0.25)  # this seems like a suitable value
        axis_values = np.array([int(c[int(rows_first)]) for c in rotated_coords])
        clusters = self.get_axis_clusters(axis_values, cut_height, fig, plate_id=plate.id)

        clusters = pd.DataFrame(
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
        plate.circles = np.array(clusters.circle.tolist())

    def get_layout(self):
        plates = self.find_plates()
        plates = self.sort_plates(plates)
        return Layout(plates, self.image.rgb.shape[:2])


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
        # todo consider the shape of the layout, if valid or not, e.g. if coords are out of range
        #  this is likely to cause exceptions downstream, we need to raise it here
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
                    centroid=tuple(df.loc[plate_id, 1][["plate_x", "plate_y"]].values)
                )
            )
        # fig = self.image.figures.new_figure("Loaded layout")
        # fig.plot_image(self.distance, "ΔE from circle colour", color_bar=True)
        # fig.print() #  todo add figure for loaded layout, maybe just the mask?
        return Layout(plates, self.image.rgb.shape[:2])


