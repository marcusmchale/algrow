import logging
from pathlib import Path

import numpy as np
from csv import reader, writer
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from alphashape import alphashape, optimizealpha
from trimesh import PointCloud

from .options import options, update_arg
from .figurebuilder import FigureBuilder
from .image_loading import ImageLoaded
from .picker import Picker
from .calibration import Configurator, Segmentor
from .area_calculation import area_worker
from .analysis import AreaAnalyser

# import sys
# import traceback

logger = logging.getLogger(__name__)

#   def excepthook(type, value, tb):
#       message = 'Uncaught exception:\n'
#       message += ''.join(traceback.format_exception(type, value, tb))
#       logger.debug(message)
#
#
#   sys.excepthook = excepthook  # this snippet is useful to catch exceptions from wx.Frame


def algrow():
    argparser = options()
    args = argparser.parse_args()
    logger.info(f"Start with: {argparser.format_values()}")

    # Organise input image file(s)
    image_filepaths = []
    if args.image:
        for i in args.image:
            if Path(i).is_file():
                image_filepaths.append(Path(i))
            elif Path(i).is_dir():
                image_filepaths = image_filepaths + [p for p in Path(i).glob('**/*.jpg')]
            else:
                raise FileNotFoundError
        logger.info(f"Processing {len(image_filepaths)} images")
    if not image_filepaths:
        logger.info(f'No images to process')

    # Organise output directory
    if not args.out_dir:
        if Path(args.image[0]).is_file():
            args.out_dir = Path(args.image[0]).parents[0]
        else:
            args.out_dir = Path(args.image[0])
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Need circle colour for layout detection, we can get this from a single image
    if not args.circle_colour:
        logger.debug("Pick circle colour")
        first_image = ImageLoaded(image_filepaths[0], args)
        # todo - make this window close after first selection is made or return as array and get median/mean
        circle_colours = Picker(first_image, 'circle colour').lasso_colours()
        circle_colour = np.round(np.median(circle_colours, axis=0), decimals=1)
        update_arg(args, 'circle_colour', tuple(circle_colour))

    if args.debug:  # add debug figure with circle colour
        logger.debug(f"Plotting debug image for circle colour: {vars(args)['circle_colour']}")
        fig = FigureBuilder(".", args, "Circle colour")
        fig.plot_colours([args.circle_colour])
        fig.print()

    # Need points to construct alpha hull for target colours selection
    # Get this from a sample of images across the sequence
    if args.target_colours is None or len(args.target_colours) < 4:  # 4 is the minimum points to define an alpha_hull
        if args.target_colours is not None and len(args.target_colours) < 0:
            logger.warning("Less than 4 target colours specified - discarding provided colours")
        idx = np.unique(np.round(np.linspace(0, len(image_filepaths) - 1, args.num_calibration)).astype(int))
        sample_images = list(np.array(image_filepaths)[idx])
        logger.info(f"Sampled {args.num_calibration} images for calibration")
        segmentor = Segmentor(sample_images, args)
        segmentor.run()
        logger.info(f"Calibration with {len(segmentor.image_filepaths)} images")

        app = Configurator(segmentor, args)
        app.MainLoop()

    if args.debug:
        fig = FigureBuilder(".", args, "Target colours")
        fig.plot_colours(args.target_colours)
        fig.print()

    if args.alpha is None:
        alpha = optimizealpha(np.array(args.target_colours))
        update_arg(args, 'alpha', alpha)

    # Output a file summarising the calibration values: selected colours, alpha and delta values
    with open(Path(args.out_dir, "colours.conf"), 'w') as text_file:
        circle_colour_string = f"\"{','.join([str(i) for i in args.circle_colour])}\""
        target_colours_string = f'{[",".join([str(j) for j in i]) for i in args.target_colours]}'.replace("'", '"')
        text_file.write(f"circle_colour = {circle_colour_string}\n")
        text_file.write(f"target_colours = {target_colours_string}\n")
        text_file.write(f"alpha = {args.alpha}\n")
        text_file.write(f"delta = {args.delta}\n")

    # Construct alpha hull from target colours
    if args.alpha == 0:
        # the api for alphashape is a bit strange,
        # it returns a shapely polygon when alpha is 0
        # rather than a trimesh object which is returned for other values of alpha
        # so just calculate the convex hull with trimesh to ensure we get a consistent return value
        alpha_hull = PointCloud(args.target_colours).convex_hull
    else:
        alpha_hull = alphashape(np.array(args.target_colours), args.alpha)

    if len(alpha_hull.faces) == 0:
        raise ValueError("The provided target colours and alpha value do not construct a complete hull")
    # prepare output file for results
    area_out = Path(args.out_dir, args.area_file)
    area_header = ['ImageFile', 'Block', 'Plate', 'Unit', 'Time', 'Pixels', 'Area']
    logger.debug("Check file exists and if it already includes data from listed images")
    if area_out.is_file():  # if the file exists then check for any already processed images
        while True:
            with open(area_out) as csv_file:
                csv_reader = reader(csv_file)
                # Skip the header
                try:
                    next(csv_reader)
                except StopIteration:
                    logger.debug("Existing file is not more than one line")
                    break
                files_done = {Path(row[0]) for row in csv_reader}
                files_done = set.intersection(set(image_filepaths), files_done)
                image_filepaths = [i for i in image_filepaths if i not in files_done]
                logger.info(
                    f'Area output file found, skipping the following images: {",".join([str(f) for f in files_done])}'
                )
    # Process images
    with open(area_out, 'a+') as csv_file:
        csv_writer = writer(csv_file)
        if area_out.stat().st_size == 0:  # True if output file is empty
            csv_writer.writerow(area_header)

        if args.processes > 1:
            with ProcessPoolExecutor(max_workers=args.processes,  mp_context=get_context('spawn')) as executor:
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

    if args.sample_id:
        area_analyser = AreaAnalyser(area_out, args.sample_id, args, area_header)
        area_analyser.fit_all(args.fit_start, args.fit_end)
        area_analyser.write_results(args.out_dir, group_plots=True)
    else:
        logger.info("No sample IDs provided")
