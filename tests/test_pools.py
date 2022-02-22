"""
Tests for the PoolXD classes, which are used to assist caching of computed quantities
which are a function of a finite set of fixed values.
"""

import pytest
import numpy as np
from flaky import flaky

from cobaya.tools import Pool1D, Pool2D

# Max number of tries per test
max_runs = 3

# Number of pool and test points
n_pool = 500
n_test = 100
# size of the perturbation rel factor and test abs factor
r_perturb = 1e-16
a_tol_test = 1e-8


@flaky(max_runs=max_runs, min_passes=1)
def test_pool1d():
    values = np.random.random(n_pool)
    pool = Pool1D(values)
    test_values = np.random.choice(values, n_test) + r_perturb * np.random.random(n_test)
    # At least a duplicate, for robustness
    test_values[-1] = test_values[0]
    indices = pool.find_indices(test_values)
    assert(np.all(np.abs(test_values - pool[indices] < a_tol_test)))


def test_pool1d_fail():
    values = np.random.random(1)
    pool = Pool1D(values)
    test_values = [2]  # out of range
    with pytest.raises(ValueError):
        pool.find_indices(test_values)


@flaky(max_runs=max_runs, min_passes=1)
def test_pool2d(from_list=False):
    if from_list:
        # num of combinations N needs to be ~= n_test
        # if list of length m (large): N = (m**2 - m) / 2 ~= m**2 / 2
        n_list = int(np.ceil(np.sqrt(2 * n_pool)))
        values = np.random.random(n_list)
    else:
        values = np.random.random(2 * n_pool).reshape((n_pool, 2))
    pool = Pool2D(values)
    test_values = pool.values[np.random.choice(range(len(pool.values)), n_test)] + \
        r_perturb * np.random.random(2 * n_test).reshape((n_test, 2))
    # At least a duplicate, for robustness
    test_values[-1] = test_values[0]
    indices = pool.find_indices(test_values)
    assert(np.all(np.abs(test_values - pool[indices]) < a_tol_test))


@flaky(max_runs=max_runs, min_passes=1)
def test_pool2d_from_list():
    test_pool2d(from_list=True)


def test_pool2d_fail():
    values = np.random.random(2)
    pool = Pool1D(values)
    test_values = [2, 2]  # out of range
    with pytest.raises(ValueError):
        pool.find_indices(test_values)
