__author__ = "Jesus Torrado and Antony Lewis"
__version__ = "3.6.2"
__obsolete__ = False
__year__ = "2026"
__url__ = "https://cobaya.readthedocs.io"

import os

# Prepare omnibus packages
_known_omnibus_packages = ["cobaya_cosmo"]
_omnibus_env = "COBAYA_OMNIBUS"
for p in os.getenv(_omnibus_env, "").replace(",", " ").split(" "):
    if p:
        _known_omnibus_packages.append(p)

from cobaya.likelihood import Likelihood as Likelihood
from cobaya.log import LoggedError as LoggedError
from cobaya.model import get_model as get_model
from cobaya.output import load_samples as load_samples
from cobaya.post import post as post
from cobaya.run import run as run
from cobaya.theory import Theory as Theory
from cobaya.typing import InputDict as InputDict
from cobaya.typing import PostDict as PostDict
