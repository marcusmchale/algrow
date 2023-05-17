import logging
from pathlib import Path
import numpy as np
from csv import reader, writer
from concurrent.futures import ProcessPoolExecutor, as_completed
from .image_processing import area_worker
from .analysis import AreaAnalyser
from .picker import Picker
from .options import options


logger = logging.getLogger(__name__)

def discgrow():
    argparser = options()
    args = argparser.parse_args()

    logger.info(f"Start with: {argparser.format_values()}")

    images = []
    #Organise input image(s)
    if args.image:
        for i in args.image:
            if Path(i).is_file():
                images.append(Path(i))
            elif Path(i).is_dir():
                images = images + [p for p in Path(i).glob('**/*.jpg')]
            else:
                raise FileNotFoundError

    if not images:
        logger.info(f'No images to process')

    # Organise target colour(s) for input images
    if not args.target_colour:
        if args.processes > 1:
            raise ValueError("Cannot interactively pick colours in multiprocess mode")
        logger.debug("Pick target colours")
        # get colour from a random image
        for first_image in images:
            picker = Picker(first_image, "target")
            target_colours = picker.get_colours()
            vars(args).update({"target_colour":[(c[0], c[1], c[2]) for c in target_colours]})
            break

    # Organise non-target colour(s) for input images
    if not args.non_target_colour:
        if args.processes > 1:
            raise ValueError("Cannot interactively pick colours in multiprocess mode")
        logger.debug("Pick non-target colours")
        # get colour from a random image
        for first_image in images:
            picker = Picker(first_image, "non-target")
            non_target_colours = picker.get_colours()
            vars(args).update({"non_target_colour":[(c[0], c[1], c[2]) for c in non_target_colours]})
            break


    # Organise circle colour(s) for layout detection
    if not args.circle_colour:
        if args.processes > 1:
            raise ValueError("Cannot interactively pick colours in multiprocess mode")
        logger.debug("Pick a circle colour")
        for first_image in images:
            picker = Picker(first_image, "circle")
            circle_colour = tuple(np.around(np.array(picker.get_colours()).mean(0), decimals=1))
            vars(args).update({"circle_colour": circle_colour})
            break

    # Organise output directory
    if not args.out_dir:
        if Path(args.image[0]).is_file():
            args.out_dir = Path(args.image[0]).parents[0]
        else:
            args.out_dir = Path(args.image[0])
    area_out = Path(args.out_dir, args.area_file)
    area_header = ['ImageFile', 'Block', 'Plate', 'Unit', 'Time', 'Pixels', 'Area']

    # Organise images to process (merging with existing analyses)
    logger.debug("Check file exists and if it already includes data from listed images")
    if area_out.is_file():  # if the file exists then check for any already processed images
        with open(area_out) as csv_file:
            csv_reader = reader(csv_file)
            next(csv_reader)
            files_done = {Path(row[0]) for row in csv_reader}
        files_done = set.intersection(set(images), files_done)
        images = [i for i in images if i not in files_done]
        logger.info(
            f'Area output file found, skipping the following images: {",".join([str(f) for f in files_done])}'
        )

    # Now process images
    with open(area_out, 'a+') as csv_file:
        csv_writer = writer(csv_file)
        if area_out.stat().st_size == 0:  # True if output file is empty
            csv_writer.writerow(area_header)
        if args.processes > 1:

            with ProcessPoolExecutor(max_workers=args.processes) as executor:
                future_to_file = {
                    executor.submit(area_worker, image, args): image for image in images
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
            for image in images:
                try:
                    result = area_worker(image, args)
                    for record in result:
                        csv_writer.writerow(record)
                except Exception as exc:
                    logger.info(f'{str(image)} generated an exception: {exc}', exc_info=True)
                else:
                    logger.info(f'{str(image)}: processed')

    #
    if args.sample_id:
        area_analyser = AreaAnalyser(area_out, args.sample_id, args, area_header)
        area_analyser.fit_all(args.fit_start, args.fit_end)
        area_analyser.write_results(args.out_dir, group_plots=True)
    else:
        logger.info("No sample IDs provided")


