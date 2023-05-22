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

    image_filepaths = []
    #Organise input image(s)
    if args.image:
        for i in args.image:
            if Path(i).is_file():
                image_filepaths.append(Path(i))
            elif Path(i).is_dir():
                image_filepaths = image_filepaths + [p for p in Path(i).glob('**/*.jpg')]
            else:
                raise FileNotFoundError

    if not image_filepaths:
        logger.info(f'No images to process')

    # Organise output directory
    if not args.out_dir:
        if Path(args.image[0]).is_file():
            args.out_dir = Path(args.image[0]).parents[0]
        else:
            args.out_dir = Path(args.image[0])

    area_out = Path(args.out_dir, args.area_file)
    area_header = ['ImageFile', 'Block', 'Plate', 'Unit', 'Time', 'Pixels', 'Area']

    if args.processes > 1:
        if not args.circle_colour & args.target_colour & args.non_target_colour:
            raise ValueError("Multiprocess mode requires supplied arguments for circle colour, target colour and non-target colour")

    # Organise images to process (merging with existing analyses)
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

    # Now process images
    with open(area_out, 'a+') as csv_file:
        csv_writer = writer(csv_file)
        if area_out.stat().st_size == 0:  # True if output file is empty
            csv_writer.writerow(area_header)
        if args.processes > 1:

            with ProcessPoolExecutor(max_workers=args.processes) as executor:
                future_to_file = {
                    executor.submit(area_worker, filepath, args): filepath for filepath in image_filepaths
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
                    result = area_worker(filepath, args)
                    for record in result:
                        csv_writer.writerow(record)
                except Exception as exc:
                    logger.info(f'{str(filepath)} generated an exception: {exc}', exc_info=True)
                else:
                    logger.info(f'{str(filepath)}: processed')

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir, "colours.txt"), 'w') as text_file:
        target_colours_string = f'{[",".join([str(j) for j in i]) for i in vars(args)["target_colour"]]}'.replace("'",'"')
        non_target_colours_string = f'{[",".join([str(j) for j in i]) for i in vars(args)["non_target_colour"]]}'.replace("'", '"')
        circle_colour_string = f"\"{','.join([str(i) for i in vars(args)['circle_colour']])}\""
        text_file.write("target_colours:")
        text_file.write(target_colours_string)
        text_file.write("\n")
        text_file.write("non_target_colours:")
        text_file.write(non_target_colours_string)
        text_file.write("\n")
        text_file.write("circle_colour:")
        text_file.write(circle_colour_string)
        text_file.write("\n")

    if args.sample_id:
        area_analyser = AreaAnalyser(area_out, args.sample_id, args, area_header)
        area_analyser.fit_all(args.fit_start, args.fit_end)
        area_analyser.write_results(args.out_dir, group_plots=True)
    else:
        logger.info("No sample IDs provided")


