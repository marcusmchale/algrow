"""Identify a layout then calculate the area that is within the target hull for each target region"""
import logging
import numpy as np

import multiprocessing
import threading
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
from csv import reader, writer

from datetime import datetime

from skimage.morphology import remove_small_holes, remove_small_objects

from .image_loading import ImageLoaded, LayoutDetector, LayoutLoader
from .logging import ImageFilepathAdapter, logger_thread, worker_log_configurer
from .hull import HullHolder

logger = logging.getLogger(__name__)


def calculate_area(args):
    if not args.area_file:
        area_out = Path(args.out_dir, "area.csv")
    else:
        area_out = args.area_file

    logger.debug(f"Output to {Path(area_out).resolve()}")

    logger.debug("Start calculations")
    # Construct alpha hull from target colours
    if args.hull_vertices is None or len(args.hull_vertices) < 4:
        raise ValueError("Insufficient hull vertices provided to construct a hull")

    logger.debug("Calculate hull")

    # prepare output file for results
    area_header = ['File', 'Block', 'Plate', 'Unit', 'Time', 'Pixels', 'Area', 'RGB', "Lab"]

    logger.debug("Check file exists and if it already includes data from listed images")
    if area_out.is_file():  # if the file exists then check for any already processed images
        logger.debug("Area file found, checking its contents")
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
                if len(image_filepaths) == 0:
                    logger.info("No image files remain to be processed")
                    raise FileNotFoundError("No image files remain to be processed (remove existing area file)")
            except StopIteration:
                logger.debug("Existing output file is only a single line (likely header only)")
                image_filepaths = args.images
    else:
        logger.debug("Area file not found, a new file will be created")
        image_filepaths = args.images

    logger.debug(f"Processing {len(image_filepaths)} images")

    # Process images
    with open(area_out, 'a+') as csv_file:
        csv_writer = writer(csv_file)
        if area_out.stat().st_size == 0:  # True if output file is empty
            csv_writer.writerow(area_header)

        if args.processes > 1:
            queue = multiprocessing.Manager().Queue(-1)
            lp = threading.Thread(target=logger_thread, args=(queue,))
            lp.start()

            with ProcessPoolExecutor(max_workers=args.processes, mp_context=multiprocessing.get_context('spawn')) as executor:
                future_to_file = {
                    executor.submit(area_worker, filepath, args, queue=queue): filepath for filepath in image_filepaths
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

            queue.put(None)
            lp.join()

        else:
            for filepath in image_filepaths:
                try:
                    result = area_worker(filepath, args)
                    for record in result:
                        csv_writer.writerow(record)
                except Exception as exc:
                    logger.info(f'{str(filepath)} generated an exception: {exc}', exc_info=True)
                else:
                    logger.info(f'{str(filepath)}: processed')


def area_worker(filepath, args, queue=None):
    if queue is not None:
        worker_log_configurer(queue)

    hull_holder = HullHolder(args.hull_vertices, args.alpha)
    hull_holder.update_hull()
    logger.debug("Prepare logging adapter")
    adapter = ImageFilepathAdapter(logging.getLogger(__name__), {'image_filepath': str(filepath)})
    adapter.debug(f"Processing file: {filepath}")
    result = ImageProcessor(filepath, hull_holder, args).get_area()
    filepath = Path(result["filename"])
    filename = str(filepath)   # keep whole path in name so can detect already done more easily

    block = None
    time = None

    filename_groups = re.compile(args.filename_regex).groupindex
    filename_search = re.search(args.filename_regex, filename)

    if filename_search is not None:
        if 'block' in filename_groups:
            block = filename_search.group(filename_groups['block'])

        time_values = ['year', 'month', 'day', 'hour', 'minute', 'second']
        for i, tv in enumerate(time_values):
            if tv not in filename_groups:
                time_values = time_values[0:i]
                break
        time_tuple = tuple(int(filename_search.group(filename_groups[tv])) for tv in time_values)
        if time_tuple:
            time = datetime(*time_tuple)

    def format_result(raw_result):
        for record in raw_result["units"]:
            plate = record[0]
            unit = record[1]
            pixels = record[2]
            rgb = record[3]
            lab = record[4]
            area = None if any([pixels is None, args.scale is None]) else round(pixels / (args.scale ** 2), 2)
            yield [filename, block, plate, unit, time, pixels, area, rgb, lab]

    return list(format_result(result))


class ImageProcessor:
    def __init__(self, filepath, hull_holder, args):
        self.image = ImageLoaded(filepath, args)
        self.hull_holder = hull_holder
        self.args = args
        self.logger = ImageFilepathAdapter(logger, {'image_filepath': str(filepath)})

    def get_area(self):

        if self.args.fixed_layout is not None:
            self.image.layout = LayoutLoader(self.image).get_layout()
        elif not self.args.detect_layout:
            self.image.layout = None
        else:
            self.image.layout = LayoutDetector(self.image).get_layout()

        masked_lab = self.image.lab[self.image.layout_mask]
        logger.debug("Get occupancy to hull")
        occupancy = self.hull_holder.get_occupancy(masked_lab)
        logger.debug("Create distances array, where occupants are 0 and outside are 1")
        inverted_occupancy = ~occupancy  # have to do this step first, can't stream invert
        distances = inverted_occupancy.astype('float')
        logger.debug(f"Calculate distance from hull surface for points outside")
        distances[~occupancy] = self.hull_holder.get_distances(masked_lab[~occupancy])
        inside = distances <= self.args.delta

        if self.args.image_debug <= 0:
            if self.image.layout is None:
                distance_image = distances.reshape(self.image.lab.shape[0:2])
            else:
                distance_image = np.empty(self.image.lab.shape[0:2])
                distance_image[~self.image.layout_mask] = 0  # set masked region as 0
                distance_image[self.image.layout_mask] = distances
            hull_distance_figure = self.image.figures.new_figure('Hull distance')
            hull_distance_figure.plot_image(distance_image, "ΔE from hull surface", color_bar=True)
            hull_distance_figure.print()

        self.logger.debug("Create mask from distance threshold")
        if self.image.layout is None:
            target_mask = inside.reshape(self.image.rgb.shape[0:2])
        else:
            target_mask = self.image.layout_mask.copy()
            target_mask[target_mask] = inside

        mask_figure = self.image.figures.new_figure("Mask processing")
        mask_figure.plot_image(target_mask, "Raw mask")
        # todo these rely on connected components consider labelling then sharing this across
        #  rather than feeding both the boolean array
        if self.args.remove:
            self.logger.debug("Remove small objects in the mask")
            target_mask = remove_small_objects(target_mask, self.args.remove)
            mask_figure.plot_image(target_mask, "Small objects removed")
        if self.args.fill:
            self.logger.debug("Fill small holes in the mask")
            target_mask = remove_small_holes(target_mask, self.args.fill)
            mask_figure.plot_image(target_mask, "Filled small holes")
        mask_figure.print()

        # output the image mask in full scale
        self.image.figures.save_image(target_mask, "Mask", level="INFO")

        result = {
            "filename": self.image.filepath,
            "units": []
        }

        overlay_figure = self.image.figures.new_figure("Overlay", level="INFO")
        overlay_figure.plot_image(self.image.rgb, "Layout and target overlay")
        overlay_figure.add_outline(target_mask)

        unit = 0

        if self.image.layout is None:
            pixels = np.count_nonzero(target_mask)
            rgb = np.median(self.image.rgb[target_mask], axis=0)
            lab = np.median(self.image.lab[target_mask], axis=0)
            result["units"].append(("N/A", "N/A", pixels, rgb, lab))
        else:
            for p in self.image.layout.plates:
                self.logger.debug(f"Processing plate {p.id}")
                overlay_figure.add_label(str(p.id), p.centroid, "black", 10)
                for j, c in enumerate(p.circles):
                    unit += 1
                    circle_mask = self.image.layout.get_circle_mask(c)
                    circle_target = circle_mask & target_mask
                    pixels = np.count_nonzero(circle_target)
                    if pixels > 0:
                        rgb = np.median(self.image.rgb[circle_target], axis=0)
                        lab = np.median(self.image.lab[circle_target], axis=0)
                    else:
                        rgb = "NA"
                        lab = "NA"
                    result["units"].append((p.id, unit, pixels, rgb, lab))
                    overlay_figure.add_label(str(unit), (c[0], c[1]), "black", 5)
                    overlay_figure.add_circle((c[0], c[1]), c[2], linewidth=1)

        overlay_figure.print(large=True)

        return result

