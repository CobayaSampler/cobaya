from __future__ import division, print_function

import pytest
from flaky import flaky

from common_sampler import body_of_test, body_of_test_speeds

### @pytest.mark.mpi


@flaky(max_runs=3, min_passes=1)
def test_polychord(modules, tmpdir):
    dimension = 3
    n_modes = 1
    info_sampler = {"polychord": {"nlive": 25 * dimension * n_modes}}
    body_of_test(dimension=dimension, n_modes=n_modes,
                 info_sampler=info_sampler, tmpdir=str(tmpdir), modules=modules)


@flaky(max_runs=5, min_passes=1)
def test_polychord_multimodal(modules, tmpdir):
    dimension = 2
    n_modes = 2
    info_sampler = {"polychord": {"nlive": 40 * dimension * n_modes}}
    body_of_test(dimension=dimension, n_modes=n_modes,
                 info_sampler=info_sampler, tmpdir=str(tmpdir), modules=modules)


@pytest.skip
@flaky(max_runs=2, min_passes=1)
def test_polychord_speeds(modules):
    info_polychord = {"polychord": {}}
    body_of_test_speeds(info_polychord, modules=modules)
