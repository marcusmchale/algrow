from pathlib import Path
from .options import options
import logging
import logging.config
import logging.handlers

args, _ = options().parse_known_args()  #needed for pyisntaller multiprocessing which injects more args


def worker_log_configurer(queue):
    logging.config.dictConfig(LOGGING_CONFIG)
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)


def logger_thread(queue):
    while True:
        record = queue.get()
        if record is None:
            break
        named_logger = logging.getLogger(record.name)
        named_logger.handle(record)


class ImageFilepathAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['image_filepath'], msg), kwargs


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s : %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': args.loglevel.name,
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
        'logfile': {
            'level': args.loglevel.name,
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': Path(args.out_dir, 'algrow.log'),
            'mode': 'a'
        }
    },
    'loggers': {
        '': {  # root logger  # I prefer to set this as ERROR, otherwise we get messages from loaded packages as well
            'handlers': ['default'],
            'level': 'ERROR',
            # e.g. there is a warning in alphashapes.alphasimplices that isn't a concern
            # in particular the presence of simplices in a single axis. These points are still found in other
            'propagate': False
        },
        'algrow': {
            'handlers': ['default', 'logfile'],
            'level': args.loglevel.name,
            'propagate': False
        }
    }
}
