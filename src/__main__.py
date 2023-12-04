from .algrow import algrow
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # required for pyinstaller/multiprocessing combination
    algrow()

