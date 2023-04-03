#!/usr/bin/env python
import logging.config
from src.logging import LOGGING_CONFIG
from src.discgrow import discgrow


logging.config.dictConfig(LOGGING_CONFIG)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.debug("Start application")
    discgrow()
