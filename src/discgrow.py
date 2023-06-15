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
from .calibration import Clicker
from .area_calculation import area_worker
from .analysis import AreaAnalyser


logger = logging.getLogger(__name__)


def discgrow():
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

    # Need points to construct alphashape for target colours selection
    # Get this from a sample of images across the sequence
    if args.target_colours is None or len(args.target_colours) < 4:  # 4 is the minimum points to define an alpha_shape
        if args.target_colours is not None and len(args.target_colours) < 0:
            logger.warning("Less than 4 target colours specified - discarding provided colours")
        n = 3  # select at most n images for calibration, spaced evenly across the input sequence
        idx = np.unique(np.round(np.linspace(0, len(image_filepaths) - 1, n)).astype(int))
        sample_images = list(np.array(image_filepaths)[idx])
        clicker = Clicker(sample_images, args)
        if clicker.alpha_shape is None:
            raise ValueError("Calibration not complete - please start again and select more than 4 points")
        if args.debug:
            clicker.plot_all(".")
        update_arg(args, 'alpha', clicker.alpha)
        update_arg(args, "target_colours", list(map(tuple, clicker.selected_lab.tolist())))
        update_arg(args, 'delta', clicker.delta_slider.val)

    if args.debug:
        fig = FigureBuilder(".", args, "Target colours")
        fig.plot_colours(args.target_colours)
        fig.print()

    if args.alpha is None:
        alpha = optimizealpha(np.array(args.target_colours))
        update_arg(args, 'alpha', alpha)

    # Output a file summarising the selected colours, alpha and delta values
    with open(Path(args.out_dir, "colours.conf"), 'w') as text_file:
        circle_colour_string = f"\"{','.join([str(i) for i in args.circle_colour])}\""
        target_colours_string = f'{[",".join([str(j) for j in i]) for i in args.target_colours]}'.replace("'",
                                                                                                          '"')
        text_file.write(f"circle_colour = {circle_colour_string}\n")
        text_file.write(f"target_colours = {target_colours_string}\n")
        text_file.write(f"alpha = {args.alpha}\n")
        text_file.write(f"delta = {args.delta}\n")

    # Construct alpha shape from target colours
    if args.alpha == 0:
        # the api for alphashape is a bit strange,
        # it returns a shapely polygon when alpha is 0
        # rather than a trimesh object which is returned for other values of alpha
        # so just calculate the convex hull with trimesh to ensure we get a consistent return value
        alpha_shape = PointCloud(args.target_colours).convex_hull
    else:
        alpha_shape = alphashape(np.array(args.target_colours), args.alpha)

    # prepare output file for results
    area_out = Path(args.out_dir, args.area_file)
    area_header = ['ImageFile', 'Block', 'Plate', 'Unit', 'Time', 'Pixels', 'Area']
    logger.debug("Check file exists and if it already includes data from listed images")
    if area_out.is_file():  # if the file exists then check for any already processed images
        with open(area_out) as csv_file:
            csv_reader = reader(csv_file)
            next(csv_reader)
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
                    executor.submit(area_worker, filepath, alpha_shape, args): filepath for filepath in image_filepaths
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
                    result = area_worker(filepath, alpha_shape, args)
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


