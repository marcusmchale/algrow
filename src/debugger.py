import cv2
from pathlib import Path
from skimage.util import img_as_ubyte
from matplotlib import pyplot as plt


class Debugger:
    def __init__(self, args, filepath):
        self.args = args
        self.filepath = filepath
        self.plot = None

    def render_image(self, img, label: str, prefix="debug", extension=".jpg"):
        if self.args.image_debug:
            img = img_as_ubyte(img)
            if self.args.image_debug == 'print':
                filepath = self.filepath.with_stem(f'{prefix}_{self.filepath.stem}').with_suffix(extension)
                cv2.imwrite(str(Path(self.args.out_dir, filepath.stem).with_suffix(extension)), img)
            elif self.args.image_debug == 'plot':
                rescale = 0.2
                width = int(img.shape[1] * rescale)
                height = int(img.shape[0] * rescale)
                dim = (width, height)
                small_img = cv2.resize(img, dim)
                cv2.imshow(label, small_img)
                cv2.waitKey()
                cv2.destroyWindow(label)

    def render_plot(self, label: str, prefix="debug", extension=".jpg"):
        if self.args.image_debug:
            if self.args.image_debug == 'print':
                filepath = self.filepath.with_stem(f'{prefix}_{self.filepath.stem}').with_suffix(extension)
                plt.savefig(str(Path(self.args.out_dir, filepath.stem).with_suffix(extension)))
            elif self.args.image_debug == 'plot':
                plt.title(label)
                plt.show()
