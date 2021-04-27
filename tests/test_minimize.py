# Minimization of a random Gaussian likelihood using the minimize sampler.

import numpy as np
from scipy.stats import multivariate_normal
from flaky import flaky
import pytest
import os

from cobaya.conventions import kinds
from cobaya.likelihoods.gaussian_mixture import info_random_gaussian_mixture
from cobaya.run import run
from cobaya import mpi

pytestmark = pytest.mark.mpi


@flaky(max_runs=3, min_passes=1)
@mpi.sync_errors
def test_minimize_gaussian(tmpdir):
    # parameters
    dimension = 3
    n_modes = 1
    # Info of likelihood and prior

    ranges = np.array([[0, 1] for _ in range(dimension)])
    prefix = "a_"
    info = info_random_gaussian_mixture(ranges=ranges, n_modes=n_modes,
                                        input_params_prefix=prefix, derived=True)
    mean = info[kinds.likelihood]["gaussian_mixture"]["means"][0]
    cov = info[kinds.likelihood]["gaussian_mixture"]["covs"][0]
    maxloglik = multivariate_normal.logpdf(mean, mean=mean, cov=cov)
    if mpi.is_main_process():
        print("Maximum of the gaussian mode to be found: %s" % mean)
    info[kinds.sampler] = {"minimize": {"ignore_prior": True}}
    info["debug"] = False
    info["debug_file"] = None

    products = run(info).sampler.products()
    # Done! --> Tests
    if mpi.is_main_process():
        rel_error = abs(maxloglik - -products["minimum"]["minuslogpost"]) / abs(maxloglik)
        assert rel_error < 0.001

    info['output'] = os.path.join(tmpdir, 'testmin')
    products = run(info).sampler.products()
    from getdist.types import BestFit
    res = BestFit(info['output'] + '.bestfit').getParamDict()
    assert np.isclose(res["loglike"], products["minimum"]["minuslogpost"])
    for p, v in list(res.items())[:-2]:
        assert np.isclose(products["minimum"][p], v)
