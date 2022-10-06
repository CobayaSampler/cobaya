import sys
import platform
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory
from cobaya.run import run
from cobaya.model import get_model
from cobaya.typing import InputDict, PostDict
from cobaya.log import LoggedError
from cobaya.post import post

if sys.version_info < (3, 7):
    if sys.version_info < (3, 6):
        print('Cobaya requires Python 3.6+, please upgrade.')
        sys.exit(1)

    # PyPyl likely won't work with likelihoods, but might as well allow here
    if platform.python_implementation() not in ['CPython', 'PyPy']:
        raise ValueError('Cobaya only supports CPython/PyPy on Python 3.6')

__author__ = "Jesus Torrado and Antony Lewis"
__version__ = "3.2.1"
__obsolete__ = False
__year__ = "2022"
__url__ = "https://cobaya.readthedocs.io"
