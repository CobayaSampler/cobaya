from __future__ import division, absolute_import
from contextlib import contextmanager
import sys
import numpy as np
from scipy import stats
import os
from six import StringIO



skip_theories = []
if os.environ.get('NO_CLASS_TESTS', False): skip_theories += ["classy"]
if os.environ.get('NO_CAMB_TESTS', False): skip_theories += ["camb"]


@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
