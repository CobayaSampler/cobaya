# Minimization of a random Gaussian likelihood using the minimize sampler.

import numpy as np
from scipy.stats import multivariate_normal
from flaky import flaky
import pytest
import os

from cobaya.typing import InputDict
from cobaya.run import run
from cobaya import mpi

pytestmark = pytest.mark.mpi

info_min: InputDict = {'likelihood': {
    'gaussian_mixture': {'means': [np.array([0.30245268, 0.61884443, 0.5])],
                         'covs': [np.array([[0.00796336, -0.0014805, -0.00479433],
                                            [-0.0014805, 0.00561415, 0.00434189],
                                            [-0.00479433, 0.00434189, 0.03208593]])],
                         'input_params_prefix': 'a_', 'output_params_prefix': '',
                         'derived': True}},
    'params': {'a__0': {'prior': {'min': 0, 'max': 1}, 'latex': '\\alpha_{0}'},
               'a__1': {'prior': {'min': 0, 'max': 1}, 'latex': '\\alpha_{1}'},
               'a__2': {'prior': {'min': 0, 'max': 1}, 'latex': '\\alpha_{2}'},
               '_0': None,
               '_1': None,
               '_2': None}}


@flaky(max_runs=3, min_passes=1)
@mpi.sync_errors
def test_minimize_gaussian(tmpdir):
    # parameters
    # dimension = 3
    # n_modes = 1
    # Info of likelihood and prior
    # ranges = np.array([[0, 1] for _ in range(dimension)])
    # info = info_random_gaussian_mixture(ranges=ranges, n_modes=n_modes,
    #      input_params_prefix = "a_", derived = True)
    info: InputDict = info_min.copy()
    mean = info["likelihood"]["gaussian_mixture"]["means"][0]
    cov = info["likelihood"]["gaussian_mixture"]["covs"][0]
    maxloglik = multivariate_normal.logpdf(mean, mean=mean, cov=cov)
    if mpi.is_main_process():
        print("Maximum of the gaussian mode to be found: %s" % mean)
    info["sampler"] = {"minimize": {"ignore_prior": True}}
    info["debug"] = False
    info["debug_file"] = None

    products = run(info).sampler.products()
    # Done! --> Tests
    if mpi.is_main_process():
        rel_error = abs(maxloglik - -products["minimum"]["minuslogpost"]) / abs(maxloglik)
        assert rel_error < 0.001

    info['output'] = os.path.join(tmpdir, 'testmin')
    products = run(info).sampler.products()
    if mpi.is_main_process():
        from getdist.types import BestFit
        res = BestFit(info['output'] + '.bestfit').getParamDict()
        assert np.isclose(res["loglike"], products["minimum"]["minuslogpost"])
        for p, v in list(res.items())[:-2]:
            assert np.isclose(products["minimum"][p], v)
