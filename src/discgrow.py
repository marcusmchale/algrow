import logging
import sys
from pathlib import Path
from csv import reader, writer
from concurrent.futures import ProcessPoolExecutor, as_completed
from .options import options
from .thresholding import area_worker
from .analysis import AreaAnalyser


def main():
    args = options()

    root_logger = logging.getLogger()
    if args.loglevel:
        if args.loglevel == 'debug':
            root_logger.setLevel(logging.DEBUG)
        elif args.loglevel == 'info':
            root_logger.setLevel(logging.INFO)
        else:
            root_logger.setLevel(logging.WARN)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    logger = logging.getLogger(__name__)
    logger.debug(f"Start with: {args}")

    area_out = Path(args.out_dir, args.area_file)

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
            images = images - files_done
            logger.info(
                f'Some images already included in result file, skipping these: {",".join([str(f) for f in files_done])}'
            )

        with open(area_out, 'a+') as csv_file:
            area_header = ['ImageFile', 'Block', 'Plate', 'Unit', 'Time', 'Pixels', 'Area']
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
                        logger.info(f'{str(image)} generated an exception: {exc}')
                    else:
                        logger.info(f'{str(image)}: processed')
    if args.sample_id:
        area_analyser = AreaAnalyser(area_out, args.sample_id, args)
        area_analyser.fit_all(args.fit_start, args.fit_end)
        area_analyser.write_results(args.out_dir, group_plots=True)

