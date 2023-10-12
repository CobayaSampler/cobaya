import sys
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory
from cobaya.run import run
from cobaya.model import get_model
from cobaya.typing import InputDict, PostDict
from cobaya.log import LoggedError
from cobaya.post import post
from cobaya.output import load_samples


if sys.version_info < (3, 8):
    print('Cobaya requires Python 3.8+, please upgrade.')
    sys.exit(1)


__author__ = "Jesus Torrado and Antony Lewis"
__version__ = "3.4.1"
__obsolete__ = False
__year__ = "2023"
__url__ = "https://cobaya.readthedocs.io"
