"""Identify a layout then calculate the area that is within the target hull for each target region"""
import logging
import numpy as np

import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
from csv import reader, writer
from re import search
from datetime import datetime

from skimage.morphology import remove_small_holes, remove_small_objects

from alphashape import alphashape
from trimesh import PointCloud
import open3d as o3d

from .image_loading import ImageLoaded, LayoutDetector, LayoutLoader
from .logging import ImageFilepathAdapter, logger_thread, worker_log_configurer

logger = logging.getLogger(__name__)


def calculate_area(args):
    area_out = args.area_file

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

    area_header = ['File', 'Block', 'Plate', 'Unit', 'Time', 'Pixels', 'Area', 'RGB', "Lab"]

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
                if len(image_filepaths) == 0:
                    logger.info("No image files remain to be processed")
                    return
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
            queue = multiprocessing.Manager().Queue(-1)
            lp = threading.Thread(target=logger_thread, args=(queue,))
            lp.start()

            with ProcessPoolExecutor(max_workers=args.processes, mp_context=multiprocessing.get_context('spawn')) as executor:
                future_to_file = {
                    executor.submit(area_worker, filepath, alpha_hull, args, queue=queue): filepath for filepath in image_filepaths
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
                    result = area_worker(filepath, alpha_hull, args)
                    for record in result:
                        csv_writer.writerow(record)
                except Exception as exc:
                    logger.info(f'{str(filepath)} generated an exception: {exc}', exc_info=True)
                else:
                    logger.info(f'{str(filepath)}: processed')


def area_worker(filepath, alpha_hull, args, queue=None):
    if queue is not None:
        worker_log_configurer(queue)

    adapter = ImageFilepathAdapter(logging.getLogger(__name__), {'image_filepath': str(filepath)})
    adapter.debug(f"Processing file: {filepath}")

    result = ImageProcessor(filepath, alpha_hull, args).get_area()
    filepath = Path(result["filename"])
    filename = str(filepath)   # keep whole path in name so can detect already done more easily

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
            rgb = record[3]
            lab = record[4]
            area = None if pixels is None else round(pixels / (args.scale ** 2), 2)
            yield [filename, block, plate, unit, time, pixels, area, rgb, lab]

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
        elif self.args.fixed_layout is not None:
            layout = LayoutLoader(self.image).get_layout()
        else:
            layout = LayoutDetector(self.image).get_layout()

        if layout is None:
            masked_lab = self.image.lab.reshape(-1, 3)
        else:
            masked_lab = self.image.lab[self.image.layout_mask]

        # see https://github.com/mikedh/trimesh/issues/1116
        # todo keep an eye on this as the alphashape package is likely to change around this
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.alpha_hull.as_open3d))
        distances_array = scene.compute_signed_distance(o3d.core.Tensor.from_numpy(masked_lab.astype(np.float32))).numpy()
        inside = distances_array < self.args.delta

        if self.args.image_debug <= 0:
            # we create a debug for distance if using delta value and at debug level for image debug
            if layout is None:
                distance_image = distances_array.reshape(self.image.lab.shape[0:2])
            else:
                distance_image = np.empty(self.image.lab.shape[0:2])
                distance_image[self.image.layout_mask] = distances_array
                distance_image[~self.image.layout_mask] = 0  # set masked region as 0
            hull_distance_figure = self.image.figures.new_figure('Hull distance')
            hull_distance_figure.plot_image(distance_image, color_bar=True)
            hull_distance_figure.print()

        self.logger.debug("Create mask from distance threshold")
        if layout is None:
            target_mask = inside.reshape(self.image.rgb.shape[0:2])
        else:
            target_mask = self.image.layout_mask.copy()
            target_mask[target_mask] = inside

        fill_mask_figure = self.image.figures.new_figure("Fill mask")
        fill_mask_figure.plot_image(target_mask, "Raw mask")
        # todo these rely on connected components consider labelling then sharing this across
        #  rather than feeding both the boolean array
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
            rgb = np.mean(self.image.rgb[target_mask], axis=0)
            lab = np.mean(self.image.lab[target_mask], axis=0)
            result["units"].append(("N/A", "N/A", pixels, rgb, lab))
        else:
            for p in layout.plates:
                self.logger.debug(f"Processing plate {p.id}")
                overlay_figure.add_label(str(p.id), p.centroid, "black", 10)
                for j, c in enumerate(p.circles):
                    unit += 1
                    circle_mask = layout.get_circle_mask(c)
                    circle_target = circle_mask & target_mask
                    pixels = np.count_nonzero(circle_target)
                    if pixels > 0:
                        rgb = np.mean(self.image.rgb[circle_target], axis=0)
                        lab = np.mean(self.image.lab[circle_target], axis=0)
                    else:
                        rgb = "NA"
                        lab = "NA"
                    result["units"].append((p.id, unit, pixels, rgb, lab))
                    overlay_figure.add_label(str(unit), (c[0], c[1]), "black", 5)
                    overlay_figure.add_circle((c[0], c[1]), c[2])

        overlay_figure.print(large=True)

        return result

