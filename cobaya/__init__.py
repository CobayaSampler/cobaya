import sys
import platform

if sys.version_info < (3, 7):
    if sys.version_info < (3, 6):
        print('Cobaya requires Python 3.6+, please upgrade.')
        sys.exit(1)

    if platform.python_implementation() != 'CPython':
        raise ValueError('Cobaya only supports CPython implementations on Python 3.6')

__author__ = "Jesus Torrado and Antony Lewis"
__version__ = "3.0.4"
__obsolete__ = False
__year__ = "2021"
__url__ = "https://cobaya.readthedocs.io"
