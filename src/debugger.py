import logging
from pathlib import Path
from skimage.io import imshow, imsave
from matplotlib import pyplot as plt
from .options import options


logger = logging.getLogger(__name__)


class Debugger:
    def __init__(self, filepath, args):
        self.filepath = filepath
        self.plot = None
        self.args = args

    def render_image(self, img, label: str, prefix="debug", extension=".jpg", force=False):
        if self.args.image_debug or force:
            logger.debug("prepare to render image to screen or file")
            if self.args.image_debug == 'print':
                filepath = self.filepath.with_stem(f'{prefix}_{self.filepath.stem}').with_suffix(extension)
                imsave(str(Path(self.args.out_dir, filepath.stem).with_suffix(extension)), img)
            elif self.args.image_debug == 'plot' or force:
                imshow(img)
                plt.title(label)
                plt.show()

    def render_plot(self, label: str, prefix="debug", extension=".jpg", force=False):
        if self.args.image_debug or force:
            logger.debug("prepare to render plot to screen or file")
            if self.args.image_debug == 'print':
                filepath = self.filepath.with_stem(f'{prefix}_{self.filepath.stem}').with_suffix(extension)
                plt.savefig(str(Path(self.args.out_dir, filepath.stem).with_suffix(extension)))
            elif self.args.image_debug == 'plot' or force:
                plt.title(label)
                plt.show()
