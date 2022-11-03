import logging
from pathlib import Path
from skimage.io import imshow, imsave
from matplotlib import pyplot as plt
from .options import options


logger = logging.getLogger(__name__)
args = options()


class Debugger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.plot = None

    def render_image(self, img, label: str, prefix="debug", extension=".jpg"):
        logger.debug("prepare to render image to screen or file")
        if args.image_debug:
            if args.image_debug == 'print':
                filepath = self.filepath.with_stem(f'{prefix}_{self.filepath.stem}').with_suffix(extension)
                imsave(str(Path(args.out_dir, filepath.stem).with_suffix(extension)), img)
            elif args.image_debug == 'plot':
                imshow(img)
                plt.title(label)
                plt.show()

    def render_plot(self, label: str, prefix="debug", extension=".jpg"):
        logger.debug("prepare to render plot to screen or file")
        if args.image_debug:
            if args.image_debug == 'print':
                filepath = self.filepath.with_stem(f'{prefix}_{self.filepath.stem}').with_suffix(extension)
                plt.savefig(str(Path(args.out_dir, filepath.stem).with_suffix(extension)))
            elif args.image_debug == 'plot':
                plt.title(label)
                plt.show()
