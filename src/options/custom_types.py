import argparse

from enum import IntEnum
from pathlib import Path


class DebugEnum(IntEnum):
    DEBUG = 0
    INFO = 1
    WARN = 2


def debug_level(s: str):
    try:
        return DebugEnum[s.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f'Log level must be one of: {[d.value for d in DebugEnum]}')


def lab(s: str | tuple[float|int]):
    if isinstance(s, tuple):
        if len(s) > 3:
            raise argparse.ArgumentTypeError(f'Colour must be a tuple with 3 float or int values: {s}')
        else:
            try:
                for i in s:
                    float(i)
            except ValueError:
                raise argparse.ArgumentTypeError(f'Colour must be a tuple with 3 float or int values: {s}')
        return s
    elif isinstance(s, str):
        try:
            l, a, b = map(float, s.split(','))
            return l, a, b
        except ValueError:
            raise argparse.ArgumentTypeError(f'Each colour must be a string with 3 comma separated float values: {s}')


def image_path(s: str):
    if Path(s).is_file():
        return [Path(s)]
    elif Path(s).is_dir():
        jpg = [p for p in Path(s).glob('**/*.jpg')]
        jpeg = [p for p in Path(s).glob('**/*.jpeg')]
        png = [p for p in Path(s).glob('**/*.png')]
        bmp = [p for p in Path(s).glob('**/*.bmp')]
        return jpg + jpeg + png + bmp
    else:
        raise FileNotFoundError


def layout_path(s: str):
    if Path(s).is_file():
        return Path(s)
    else:
        raise ValueError("Layout file not found")


arg_types = {
    "conf": str,
    "images": image_path,
    "sample_id": str,
    "out_dir": str,
    "time_regex": str,
    "time_format": str,
    "block_regex": str,
    "animations": bool,
    "processes": int,
    "image_debug": debug_level,
    "loglevel": debug_level,
    "num_superpixels": int,
    "superpixel_compactness": float,
    "sigma": float,
    "circle_colour": lab,
    "hull_vertices": lab,
    "alpha": float,
    "delta": float,
    "num_calibration": int,
    "blur": float,
    "remove": int,
    "fill": int,
    "area_file": str,
    "fit_start": float,
    "fit_end": float,
    "scale": float,
    "fixed_layout": layout_path,
    "circle_diameter": float,
    "circle_expansion": float,
    "plate_circle_separation": float,
    "plate_cut_expansion": float,
    "plate_width": float,
    "circles_per_plate": int,
    "n_plates": int,
    "plates_cols_first": bool,
    "plates_right_left": bool,
    "plates_bottom_top": bool,
    "circles_cols_first": bool,
    "circles_right_left": bool,
    "circles_bottom_top": bool
}
