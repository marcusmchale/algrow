import sys
import logging
from pathlib import Path
from csv import reader, writer


from concurrent.futures import ProcessPoolExecutor, as_completed
from options import options
from image_processing import ImageProcessor


def main():

    if Path(args.image).is_file():
        images = {Path(args.image)}
    elif Path(args.image).is_dir():
        images = set(Path(args.image).glob('**/*.jpg'))
    else:
        raise FileNotFoundError

    area_out = Path(args.out_dir, "area.csv")

    if area_out.is_file():  # if the file exists then check for any already processed images
        with open(area_out) as csv_file:
            csv_reader = reader(csv_file)
            next(csv_reader)
            files_done = {Path(row[0]) for row in csv_reader}
        images = images - files_done
        if args.debug:
            logging.info(
                f'Some images already included in result file, skipping these: {",".join([str(f) for f in files_done])}'
             )

    header = ['filename', 'plate', 'well', 'pixels', 'mm²']

    with open(area_out, 'a+') as csv_file:
        csv_writer = writer(csv_file)
        if area_out.stat().st_size == 0:  # True if empty
            csv_writer.writerow(header)

        with ProcessPoolExecutor(max_workers = args.processes) as executor:

            future_to_file = {executor.submit(ImageProcessor(image, args).get_area()): image for image in images}

            for future in as_completed(future_to_file):
                fp = future_to_file[future]
                try:
                    result = future.result()
                    for r in result[1]:
                        csv_writer.writerow([result[0], r[0], r[1], r[2], None if r[2] is None else r[2] / args.scale])
                except Exception as exc:
                    print('%r generated an exception: %s' % (str(fp), exc))
                else:
                    if args.debug:
                        print('%r processed' % (str(fp)))


if __name__ == '__main__':
    args = options()

    logger = logging.getLogger()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    main()



