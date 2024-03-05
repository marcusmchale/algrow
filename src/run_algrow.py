#!/usr/bin/env python
from algrow.launch import run
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # required for compiled binaries (pyinstaller) multiprocessing support
    run()
