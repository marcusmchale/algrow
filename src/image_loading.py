import argparse
import logging

import numpy as np
import open3d as o3d
from pathlib import Path
from copy import deepcopy

from skimage.io import imread
from skimage.util import img_as_bool, img_as_float64 as img_as_float  # 64 is actually faster with open3d so coerce to this
from skimage.color import rgb2lab, gray2rgb
from skimage.transform import downscale_local_mean

from scipy.ndimage import zoom

from .logging import ImageFilepathAdapter
from .options import DebugEnum
from .figurebuilder import FigureBase, FigureMatplot, FigureNone

from typing import Optional, List, Tuple


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


class MaskLoaded:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.mask = img_as_bool(imread(str(filepath)))


class CalibrationImage:  # an adapter to allow zooming and hold other image data only needed during calibration

    def __init__(self, image: ImageLoaded):
        self._image = image
        # this is constructed once but after loading as we want to pickle during loading for multiprocessing
        # currently o3d cloud cannot be pickled
        self.cloud = None
        self.indices = None
        self.image_to_voxel = dict()

        # restrict zooming to perfect zooms, i.e. where whole numbers of pixels
        # we don't handle interpolation/cropping for now
        self.displayed = self._image.rgb.copy()
        self.zoom_index = 0
        self.displayed_start_x = 0
        self.displayed_start_y = 0
        self.height, self.width = self._image.rgb.shape[0:2]
        self.divisors = list()
        gcd = np.gcd(self.height, self.width)
        for i in range(1, gcd+1):
            if gcd % i == 0:
                self.divisors.append(int(i))

        self.true_mask: Optional[MaskLoaded] = None

    def __hash__(self):
        return hash(self._image.filepath)

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
    def filepath(self):
        return self._image.filepath

    @property
    def figures(self):
        return self._image.figures

    def copy(self):
        self._image = self._image.copy()
        return self

    # The down-sampling uses a fair bit of memory so moving this out from multi-loading:
    def prepare_cloud(self):
        self.cloud, self.indices = self.get_downscaled_cloud_and_indices()

        # self._lab_points, self._rgb_points, self.indices = self.get_downscaled_cloud_and_indices()
        # need to build a reverse mapping from image index to index of voxel in cloud
        logger.debug("Build map from image to voxel")  # todo refactor this, it is slow
        for i, jj in enumerate(self.indices):
            for j in jj:
                self.image_to_voxel[j] = i

    def get_downscaled_cloud_and_indices(self):  # indices are the back reference to the image pixels
        cloud = o3d.geometry.PointCloud()
        logger.debug("flatten image")
        lab = self.lab.reshape(-1, 3)
        rgb = self.rgb.reshape(-1, 3)
        logger.debug("Set points")
        cloud.points = o3d.utility.Vector3dVector(lab)
        logger.debug("Set point colours")
        cloud.colors = o3d.utility.Vector3dVector(rgb)
        logger.debug("Downsample to voxels")
        # need to store cloud as ndarray so is pickleable
        cloud, _, indices = cloud.voxel_down_sample_and_trace(voxel_size=self._image.args.voxel_size, min_bound=[0, -128, -128], max_bound=[100, 127, 127])
        # the below were needed when pickling the result for multiprocessing
        #lab_points = np.asarray(cloud.points)
        #rgb_points = np.asarray(cloud.colors)
        #indices = [np.asarray(i) for i in indices]
        return cloud, indices

    @property
    def zoom_factor(self):
        return 1/self.divisors[self.zoom_index]

    @property
    def args(self):
        return self._image.args

    @args.setter
    def args(self, args):
        self._image.args = args

    def increment_zoom(self, zoom_increment):
        new_step = self.zoom_index + zoom_increment
        if new_step < 0:
            self.zoom_index = 0
        elif new_step > len(self.divisors) - 1:
            self.zoom_index = len(self.divisors) - 1
        else:
            self.zoom_index += zoom_increment

    def get_zoom_start(self, x_center, y_center, new_width, new_height):
        if x_center < new_width/2:
            zoom_start_x = 0
        elif x_center > (self._image.rgb.shape[1] - (new_width/2)):
            zoom_start_x = self._image.rgb.shape[1] - new_width
        else:
            zoom_start_x = x_center - (new_width/2)
        if y_center < new_height/2:
            zoom_start_y = 0
        elif y_center > (self._image.rgb.shape[0] - (new_height/2)):
            zoom_start_y = self._image.rgb.shape[0] - new_height
        else:
            zoom_start_y = y_center - (new_height/2)
        return int(zoom_start_x), int(zoom_start_y)

    @property
    def lab(self):
        return self._image.lab

    @property
    def rgb(self):
        return self._image.rgb

    @property
    def as_o3d(self):
        return o3d.geometry.Image(self.displayed.astype(np.float32))

    def apply_zoom(self, cropped_rescaled, x_start, y_start):
        self.displayed = cropped_rescaled
        self.displayed_start_x = x_start
        self.displayed_start_y = y_start

    def drag(self, x_drag, y_drag):
        logger.debug(f"drag: x = {x_drag}, y = {y_drag}")
        self.displayed_start_x += x_drag
        self.displayed_start_y += y_drag
        if self.displayed_start_x < 0:
            self.displayed_start_x = 0
        if self.displayed_start_y < 0:
            self.displayed_start_y = 0
        width = int(self.zoom_factor * self.width)
        height = int(self.zoom_factor * self.height)
        if self.displayed_start_x + width > self.width:
            self.displayed_start_x = self.width - width
            displayed_end_x = self.width
        else:
            displayed_end_x = self.displayed_start_x + width
        if self.displayed_start_y + height > self.height:
            self.displayed_start_y = self.height - height
            displayed_end_y = self.height
        else:
            displayed_end_y = self.displayed_start_y + height
        logger.debug(f"display: x = {self.displayed_start_x} : {displayed_end_x}, y = {self.displayed_start_y} : {displayed_end_y}")
        cropped = self._image.rgb[self.displayed_start_y:displayed_end_y, self.displayed_start_x:displayed_end_x]
        self.displayed = zoom(
            cropped,
            (self.divisors[self.zoom_index], self.divisors[self.zoom_index], 1),
            order=0,
            grid_mode=True,
            mode='nearest'
        )

    def calculate_zoom(self, x_center, y_center, zoom_increment: int):
        self.increment_zoom(zoom_increment)
        new_width = int(self.zoom_factor * self.width)
        new_height = int(self.zoom_factor * self.height)
        x_start, y_start = self.get_zoom_start(x_center, y_center, new_width, new_height)
        cropped = self._image.rgb[y_start:y_start + new_height, x_start:x_start + new_width]
        cropped_rescaled = zoom(
            cropped,
            (self.divisors[self.zoom_index], self.divisors[self.zoom_index], 1),
            order=0,
            grid_mode=True,
            mode='nearest'
        )
        logger.debug(f"new_shape: {cropped.shape}")
        return cropped_rescaled, x_start, y_start

    def indices_in_displayed(self, selected: List[int]) -> List[int]:
        # todo as an array rather than individually
        selected_in_displayed = [self.full_pixel_to_displayed_pixel(i) for i in selected]
        return [j for i in selected_in_displayed if i is not None for j in i]

    @staticmethod
    def coord_to_pixel(image, x, y) -> int:  # pixel is the index, coord is x,y
        if all([x >= 0, x < image.shape[1], y >= 0, y < image.shape[0]]):
            return (y * image.shape[1]) + x
        else:
            raise ValueError("Coordinates are outside of image")

    @staticmethod
    def pixel_to_coord(image, i: int) -> Tuple[int, int]:  # pixel is the index, coord is x,y
        return np.unravel_index(i, image.shape[0:2])[::-1]
        #x_length = image.shape[1]
        #return i % x_length, int(np.floor(i / x_length))  # in x,y order

    def full_pixel_to_displayed_pixel(self, pixel: int) -> Optional[List[int]]:  # can be many due to zooming but at least one
        if self.zoom_factor == 1:
            return [pixel]
        image_x, image_y = self.pixel_to_coord(self.rgb, pixel)
        displayed_x = (image_x - self.displayed_start_x) * self.divisors[self.zoom_index]
        displayed_y = (image_y - self.displayed_start_y) * self.divisors[self.zoom_index]
        try:
            starting = self.coord_to_pixel(self.displayed, displayed_x, displayed_y)
            # need to handle expansion due to zooming
            pixel_expansion = range(0, self.divisors[self.zoom_index])
            x_expanded = [starting + i for i in pixel_expansion]
            all_expanded = [j + self.displayed.shape[1] * i for i in pixel_expansion for j in x_expanded]
            return all_expanded
        except ValueError:
            return None

    @property
    def displayed_lab(self):
        height = int(self.height / self.divisors[self.zoom_index])
        width = int(self.width / self.divisors[self.zoom_index])
        x_start = self.displayed_start_x
        y_start = self.displayed_start_y
        return self.lab[y_start:y_start + height, x_start:x_start + width]

    @property
    def displayed_true_mask(self):
        height = int(self.height / self.divisors[self.zoom_index])
        width = int(self.width / self.divisors[self.zoom_index])
        x_start = self.displayed_start_x
        y_start = self.displayed_start_y
        return self.true_mask.mask[y_start:y_start + height, x_start:x_start + width]
