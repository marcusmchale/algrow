import logging

import numpy as np
import pandas as pd

from skimage.color import lab2rgb

from ..image_loading import ImageLoaded
from .loading import wait_for_result

logger = logging.getLogger(__name__)

# todo use voxel downsampling instead of rounding?

class ColourPoints:
    def __init__(self, image: ImageLoaded):
        self.image = image
        self.args = self.image.args
        self.nearest = self.args.colour_rounding

        # Start with colour bins at some grouping (nearest)
        # Each bin has:
        #    - lab (coordinates in Lab space)
        #    - rgb (colour for plotting)
        #    - count total (n pixels all images)
        #    - count per file (n pixels from each image)
        #    - indices per file (to map to x,y)
        self.pixel_to_lab, self.lab_to_pixel, self.counts = self._run()
        self.counts['distance'] = np.inf  # a placeholder until we calculate distance from hull later

    @wait_for_result
    def _run(self):
        cols = [("lab", "L"), ("lab", "a"), ("lab", "b")]
        cols_index = pd.MultiIndex.from_tuples(cols)
        logger.debug("Get image as dataframe")
        pixel_to_lab = pd.DataFrame(self.image.lab.reshape(-1, 3), columns=cols_index)
        # these dataframes are not references to the image arrays

        logger.debug(f"Round to nearest {self.nearest}")
        pixel_to_lab.index = pixel_to_lab.index.set_names("pixel")
        pixel_to_lab = pixel_to_lab.divide(self.nearest).round().multiply(self.nearest)
        pixel_to_lab = pixel_to_lab.sort_index()

        logger.debug(f"Count pixels per colour, set and sort index")
        counts = pixel_to_lab.value_counts().reset_index().sort_index()
        logger.debug(f"Calculate RGB for each colour")
        counts[[("rgb", "r"), ("rgb", "g"), ("rgb", "b")]] = lab2rgb(counts[cols])

        logger.debug(f"Generate dataframe with reverse index for colour to pixel")
        lab_to_pixel = pixel_to_lab.reset_index().set_index(cols).sort_index()
        logger.debug("done")
        return pixel_to_lab, lab_to_pixel, counts
