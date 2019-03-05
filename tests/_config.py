from __future__ import division, absolute_import
from contextlib import contextmanager
import sys
import numpy as np
from scipy import stats
import os
import subprocess
from pkg_resources import parse_version
from six import StringIO


def check_reproducible():
    np.random.seed(0)
    _test_mat = stats.special_ortho_group.rvs(3)
    # signs etc. can flip on different platforms, don't test if results not expected to be the same
    return np.abs(_test_mat[0, 0] + 0.8577182409977431) < 1e-10 and \
           np.abs(_test_mat[1, 1] + 0.7206340034201331) < 1e-10


random_reproducible = check_reproducible()

test_figs = False

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


def call_command(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    except:
        return None


def get_gcc_version():
    return call_command("gcc -dumpversion")


def gcc_version_atleast(min_version='6.3'):
    version = get_gcc_version()
    return version is not None and parse_version(str(min_version)) <= parse_version(version)
