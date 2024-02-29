import sys
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.color import rgb2gray


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


mask_file1 = MaskLoaded(Path(sys.argv[1]))
mask_file2 = MaskLoaded(Path(sys.argv[2]))
mask1 = mask_file1.mask
mask2 = mask_file2.mask
tp = np.sum(mask1[mask2])
tn = np.sum(~mask1[~mask2])
fp = np.sum(~mask1[mask2])
fn = np.sum(mask1[~mask2])
sensitivity = (tp / (tp + fn))
specificity = (tn / (tn + fp))
precision = (tp/(tp + fp))
accuracy = (tp + tn) / (tp + tn + fp + fn)
dice = 2 * tp / ((2 * tp) + fp + fn)
print((
    f"Files: {mask_file1.filepath.name, mask_file2.filepath.name}, "
    f"Dice coefficient: {dice}, "
    f"Accuracy: {accuracy}, "
    f"Sensitivity: {sensitivity}, "
    f"Specificity: {specificity},"
    f"Precision: {precision}"
))
