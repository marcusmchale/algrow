from pathlib import Path
from .options.parse_args import options
import logging
import logging.config
import logging.handlers


args = options().parse_args()


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
        '': {  # root logger  # I prefer to set this as WARNING, otherwise we get debug from loaded packages as well
            'handlers': ['default'],
            'level': 'WARNING',
            'propagate': False
        },
        'src': {
            'handlers': ['default', 'logfile'],
            'level': args.loglevel.name,
            'propagate': False
        },
        '__main__': {  # if __name__ == '__main__'
            'handlers': ['default'],
            'level': args.loglevel.name,
            'propagate': False
        },
    }
}
