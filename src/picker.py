import logging
from pathlib import Path
import cv2
import numpy as np
from .debugger import Debugger


class Picker:
    def __init__(self, image_path, args):
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.image_debugger = Debugger(args, Path(image_path))
        self.logger.debug("Load selected image for colour picking")
        self.rgb = cv2.imread(str(image_path))
        self.image_debugger.render_image(self.rgb, str(image_path))
        self.lab = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2Lab)
        l, a, b = cv2.split(self.lab)
        self.a = a
        self.image_debugger.render_image(a, f"Green-Red channel (a in Lab)")
        self.b = b
        self.image_debugger.render_image(b, f"Blue-Yellow channel (b in Lab)")

    def pick_regions(self):
        window_string = f"Select representative regions. Press space/enter for each selection and Esc to finish"
        rects = cv2.selectROIs(
            window_string,
            self.rgb,
            fromCenter=False,
            showCrosshair=False
        )
        cv2.destroyWindow(window_string)
        if len(rects) == 0:
            raise ValueError("No areas selected")
        mask = np.zeros(self.a.shape, np.uint8)
        for r in rects:
            # r is [Top_Left_X, Top_Left_Y, Width, Height]
            # mask[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = self.rgb[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
            mask[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = 255
        self.image_debugger.render_image(mask, f"Region mask")
        return mask

    def get_mean_colour(self):
        mask = self.pick_regions()
        mean_colour = cv2.mean(self.lab, mask)[0:3]
        mean_colour = dict(zip(["target_L", "target_a", "target_b"], [round(i) for i in mean_colour]))
        colour_image = np.zeros_like(self.lab)
        colour_image[:, :, :] = (mean_colour["target_L"], mean_colour["target_a"], mean_colour["target_b"])
        self.image_debugger.render_image(colour_image, f"Picked colour")
        self.logger.info(f"Mean colour: {mean_colour}")
        return mean_colour




