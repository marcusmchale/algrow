import logging
from pathlib import Path
from csv import reader, writer
from concurrent.futures import ProcessPoolExecutor, as_completed
from .image_processing import area_worker
from .analysis import AreaAnalyser
from .picker import Picker
from .options import options

logger = logging.getLogger(__name__)
args = options()


def discgrow():
    logger.info(f"Start with: {args}")
    area_out = Path(args.out_dir, args.area_file)
    area_header = ['ImageFile', 'Block', 'Plate', 'Unit', 'Time', 'Pixels', 'Area']

    if args.image:
        if Path(args.image).is_file():
            images = {Path(args.image)}
        elif Path(args.image).is_dir():
            images = set(Path(args.image).glob('**/*.jpg'))
        else:
            raise FileNotFoundError

        logger.debug("Check file exists and if it already includes data from listed images")

        if area_out.is_file():  # if the file exists then check for any already processed images
            with open(area_out) as csv_file:
                csv_reader = reader(csv_file)
                next(csv_reader)
                files_done = {Path(row[0]) for row in csv_reader}
            files_done = set.intersection(images, files_done)
            images = images - files_done
            logger.info(
                f'Some images already included in result file, skipping these: {",".join([str(f) for f in files_done])}'
            )

        if not images:
            logger.info(f'No images to process')

        if not args.target_colour:
            logger.debug("Pick a colour")
            # get colour from a random image
            for first_image in images:
                picker = Picker(first_image)
                target_colours = picker.get_target_colours()
                vars(args).update({"target_colour":[",".join((str(c[0]), str(c[1]), str(c[2]))) for c in target_colours]})
                break

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
    if args.sample_id:
        area_analyser = AreaAnalyser(area_out, args.sample_id, args, area_header)
        area_analyser.fit_all(args.fit_start, args.fit_end)
        area_analyser.write_results(args.out_dir, group_plots=True)

