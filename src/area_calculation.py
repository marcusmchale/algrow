"""Identify a layout of circles then calculate the area within each circle that is within the target hull """
import logging
import numpy as np

from pathlib import Path
from csv import reader, writer
from re import search
from datetime import datetime

from skimage.morphology import remove_small_holes, remove_small_objects

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from trimesh import proximity
from alphashape import alphashape
from trimesh import PointCloud

from .layout import LayoutDetector
from .image_loading import ImageFilepathAdapter, ImageLoaded


logger = logging.getLogger(__name__)


def calculate(args):
    logger.debug("Start calculations")
    # Construct alpha hull from target colours
    if args.hull_vertices is None or len(args.hull_vertices) < 4:
        raise ValueError("Insufficient hull vertices provided to construct a hull")
    logger.debug("Calculate hull")
    if args.alpha == 0:
        # the api for alphashape is a bit strange,
        # it returns a shapely polygon when alpha is 0
        # rather than a trimesh object which is returned for other values of alpha
        # so just calculate the convex hull with trimesh to ensure we get a consistent return value
        alpha_hull = PointCloud(args.hull_vertices).convex_hull
    else:
        alpha_hull = alphashape(np.array(args.hull_vertices), args.alpha)
    if len(alpha_hull.faces) == 0:
        raise ValueError("The provided vertices do not construct a complete hull with the chosen alpha parameter")

    # prepare output file for results
    area_out = Path(args.out_dir, args.area_file)
    area_header = ['ImageFile', 'Block', 'Plate', 'Unit', 'Time', 'Pixels', 'Area']
    logger.debug("Check file exists and if it already includes data from listed images")
    if area_out.is_file():  # if the file exists then check for any already processed images
        with open(area_out) as csv_file:
            csv_reader = reader(csv_file)
            # Skip the header
            try:
                next(csv_reader)
                files_done = {Path(row[0]) for row in csv_reader}
                files_done = set.intersection(set(args.images), files_done)
                image_filepaths = [i for i in args.images if i not in files_done]
                logger.info(
                    f'Area output file found, skipping the following images: {",".join([str(f) for f in files_done])}'
                )
            except StopIteration:
                logger.debug("Existing output file is only a single line (likely header only)")
                image_filepaths = args.images
    else:
        image_filepaths = args.images

    logger.debug(f"Processing {len(image_filepaths)} images")
    # Process images
    with open(area_out, 'a+') as csv_file:
        csv_writer = writer(csv_file)
        if area_out.stat().st_size == 0:  # True if output file is empty
            csv_writer.writerow(area_header)

        if args.processes > 1:
            with ProcessPoolExecutor(max_workers=args.processes, mp_context=get_context('spawn')) as executor:
                future_to_file = {
                    executor.submit(area_worker, filepath, alpha_hull, args): filepath for filepath in image_filepaths
                }
                for future in as_completed(future_to_file):
                    fp = future_to_file[future]
                    try:
                        result = future.result()
                        for record in result:
                            csv_writer.writerow(record)
                    except Exception as exc:
                        logger.info(f'{str(fp)} generated an exception: {exc}')
                    else:
                        logger.info(f'{str(fp)}: processed')
        else:
            for filepath in image_filepaths:
                try:
                    result = area_worker(filepath, alpha_hull, args)
                    for record in result:
                        csv_writer.writerow(record)
                except Exception as exc:
                    logger.info(f'{str(filepath)} generated an exception: {exc}', exc_info=True)
                else:
                    logger.info(f'{str(filepath)}: processed')


def area_worker(filepath, alpha_hull, args):
    adapter = ImageFilepathAdapter(logger, {'image_filepath': str(filepath)})
    adapter.debug(f"Processing file: {filepath}")

    result = ImageProcessor(filepath, alpha_hull, args).get_area()
    filename = result["filename"]

    block_match = search(args.block_regex, str(filename))
    if block_match:
        block = block_match.group(1)
    else:
        block = None
    time_match = search(args.time_regex, str(filename))
    if time_match:
        time = datetime(*[int(time_match[i]) for i in range(1, 6)]).isoformat(sep=" ")
    else:
        time = None

    def format_result(raw_result):
        for record in raw_result["units"]:
            plate = record[0]
            unit = record[1]
            pixels = record[2]
            area = None if pixels is None else round(pixels / (args.scale ** 2), 2)
            yield [filename, block, plate, unit, time, pixels, area]

    return list(format_result(result))


class ImageProcessor:
    def __init__(self, filepath, alpha_hull, args):
        self.image = ImageLoaded(filepath, args)
        self.alpha_hull = alpha_hull
        self.args = args
        self.logger = ImageFilepathAdapter(logger, {'image_filepath': str(filepath)})

    def get_area(self):
        if self.args.whole_image:
            layout = None
        else:
            layout = LayoutDetector(self.image).get_layout()

        # break up the masked image into chunks of points as the signed_distance calculation gives OOM otherwise
        if layout is None:
            masked_lab = self.image.lab.reshape(-1, self.image.lab.shape[-1])
        else:
            masked_lab = self.image.lab[layout.mask]
        ind = np.linspace(0, len(masked_lab), num=int(len(masked_lab)/1e5), dtype=int, endpoint=False)[1:]

        self.logger.debug("Calculate distance from hull")
        distances = list()
        for c in np.split(masked_lab, ind):
            dist_c = proximity.signed_distance(self.alpha_hull, c)
            distances.append(dist_c)

        distances_array = np.concatenate(distances)

        if layout is None:
            distance_image = np.negative(distances_array).reshape(self.image.lab.shape[0:2])
        else:
            distance_image = np.empty(self.image.lab.shape[0:2])
            distance_image[layout.mask] = np.negative(distances_array)
            distance_image[~layout.mask] = 0  # set masked region as 0

        hull_distance_figure = self.image.figures.new_figure('Hull distance')
        if hull_distance_figure is not None:
            hull_distance_figure.plot_image(distance_image, color_bar=True)
            hull_distance_figure.print()

        self.logger.debug("Create mask from distance threshold")
        if layout is None:
            target_mask = (distance_image < self.args.delta)
        else:
            target_mask = (distance_image < self.args.delta) & layout.mask

        fill_mask_figure = self.image.figures.new_figure("Fill mask")
        fill_mask_figure.plot_image(target_mask, "Raw mask")
        if self.args.remove:
            self.logger.debug("Remove small objects in the mask")
            target_mask = remove_small_objects(target_mask, self.args.remove)
            fill_mask_figure.plot_image(target_mask, "Small objects removed")
        if self.args.fill:
            self.logger.debug("Fill small holes in the mask")
            target_mask = remove_small_holes(target_mask, self.args.fill)
            fill_mask_figure.plot_image(target_mask, "Filled small holes")
        fill_mask_figure.print()

        result = {
            "filename": self.image.filepath,
            "units": []
        }

        overlay_figure = self.image.figures.new_figure("Overlay", level="INFO")
        overlay_figure.plot_image(self.image.rgb, "Layout and target overlay")
        overlay_figure.add_outline(target_mask)
        unit = 0

        if layout is None:
            pixels = np.count_nonzero(target_mask)
            result["units"].append(("N/A", "N/A", pixels))
        else:
            for p in layout.plates:
                self.logger.debug(f"Processing plate {p.id}")
                overlay_figure.add_label(str(p.id), p.centroid, "red", 10)
                for j, c in enumerate(p.circles):
                    unit += 1
                    circle_mask = layout.get_circle_mask(c)
                    circle_target = circle_mask & target_mask
                    pixels = np.count_nonzero(circle_target)
                    result["units"].append((p.id, unit, pixels))
                    overlay_figure.add_label(str(unit), (c[0], c[1]), "blue", 5)
                    overlay_figure.add_circle((c[0], c[1]), c[2], "white")

        overlay_figure.print()

        return result

