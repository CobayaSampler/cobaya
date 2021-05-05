# Minimization of a random Gaussian likelihood using the minimize sampler.

import numpy as np
import pytest
import os

from cobaya import mpi, run, InputDict, Likelihood

pytestmark = pytest.mark.mpi

mean = np.array([0.30245268, 0.61884443, 0.5])
cov = np.array(
    [[0.00796336, -0.0014805, -0.00479433], [-0.0014805, 0.00561415, 0.00434189],
     [-0.00479433, 0.00434189, 0.03208593]])
_inv_cov = np.linalg.inv(cov)


class NoisyCovLike(Likelihood):
    params = {'a': [0, 1], 'b': [0, 1], 'c': [0, 1]}

    def logp(self, **params_values):
        x = np.array([params_values['a'], params_values['b'], params_values['c']]) - mean
        return -_inv_cov.dot(x).dot(x) / 2 + np.random.random_sample() * 0.005


@mpi.sync_errors
def test_minimize_gaussian(tmpdir):
    # parameters
    # dimension = 3
    # n_modes = 1
    # Info of likelihood and prior
    # ranges = np.array([[0, 1] for _ in range(dimension)])
    # info = info_random_gaussian_mixture(ranges=ranges, n_modes=n_modes,
    #      input_params_prefix = "a_", derived = True)

    maxloglik = 0
    info: InputDict = {'likelihood': {'like': NoisyCovLike},
                       "sampler": {"minimize": {"ignore_prior": True}}}
    products = run(info).sampler.products()
    # Done! --> Tests
    if mpi.is_main_process():
        error = abs(maxloglik - -products["minimum"]["minuslogpost"])
        assert error < 0.01

    info['output'] = os.path.join(tmpdir, 'testmin')
    products = run(info).sampler.products()
    if mpi.is_main_process():
        from getdist.types import BestFit
        res = BestFit(info['output'] + '.bestfit').getParamDict()
        assert np.isclose(res["loglike"], products["minimum"]["minuslogpost"])
        for p, v in list(res.items())[:-2]:
            assert np.isclose(products["minimum"][p], v)
