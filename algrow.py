#!/usr/bin/env python
from src.algrow import algrow
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    algrow()
