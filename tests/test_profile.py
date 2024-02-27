# Profile of a random Gaussian likelihood using the profile sampler.

import numpy as np
import pytest
import os

from cobaya import mpi, run, InputDict, Likelihood
from cobaya.samplers.minimize import valid_methods
valid_methods = tuple(method for method in valid_methods if method != 'iminuit')

pytestmark = pytest.mark.mpi

mean = np.array([0.30245268, 0.61884443, 0.5])
cov = np.array(
    [[0.00796336, -0.0014805, -0.00479433], [-0.0014805, 0.00561415, 0.00434189],
     [-0.00479433, 0.00434189, 0.03208593]])
_inv_cov = np.linalg.inv(cov)

mean_c = mean[2]
sigma_c = round(np.sqrt(cov[2, 2]), 3)
profiled_values = [
    mean_c - 2 * sigma_c,
     mean_c - sigma_c,
     mean_c,
     mean_c + sigma_c,
     mean_c + 2 * sigma_c
]


class NoisyCovLike(Likelihood):
    params = {'a': [0, 1, 0.5, 0.3, 0.08], 'b': [0, 1, 0.5, 0.3, 0.08],
              'c': [0, 1, 0.5, 0.3, 0.08]}
    noise = 0

    def logp(self, **params_values):
        x = np.array([params_values['a'], params_values['b'], params_values['c']]) - mean
        return -_inv_cov.dot(x).dot(x) / 2 + self.noise * np.random.random_sample()


@mpi.sync_errors
def test_profile_gaussian(tmpdir):
    loglikes_vals = [-4, -1, 0, -1, -4]
    for method in reversed(valid_methods):
        NoisyCovLike.noise = 0.001 if method == 'bobyqa' else 0
        info: InputDict = {'likelihood': {'like': NoisyCovLike},
                           "sampler": {"profile": {"ignore_prior": True,
                                                   "profiled_param": "c",
                                                   "profiled_values": profiled_values,
                                                   "method": method}}}
        products = run(info)[1].products()
        if mpi.is_main_process():
            errors = abs(loglikes_vals - -products["minima"]["chi2"])
            assert all(error < 0.01 for error in errors)

        info['output'] = os.path.join(tmpdir, 'testmin')
        info['force'] = True
        products = run(info, force=True)[1].products()
        if mpi.is_main_process():
            filename = info['output'] + ".like_profile.txt"
            res = np.loadtxt(filename, skiprows=1)
            with open(filename, "rb") as f:
                lines = f.readlines()
                header = lines[0].decode("utf-8").split()[1:]
            res = dict(zip(header, res.T))
            assert all(np.isclose(res["chi2"], products["minima"]["chi2"]))
            for p, v in list(res.items())[:-2]:
                assert all(np.isclose(products["minima"][p], list(v)))


@mpi.sync_errors
def test_run_profile(tmpdir):
    NoisyCovLike.noise = 0
    info: InputDict = {'likelihood': {'like': NoisyCovLike},
                       "sampler": {"mcmc": {"Rminus1_stop": 0.5,
                                            'Rminus1_cl_stop': 0.4,
                                            'seed': 2}},
                       "output": os.path.join(tmpdir, 'testchain')}
    run(info, force=True)
    min_info: InputDict = dict(info, sampler={'profile': {
                                                "profiled_param": "c",
                                                "profiled_values": profiled_values
                                                }})
    sampler = run(min_info, force=True)[1]
    if mpi.is_main_process():
        # Select third value where c is equal to mean_c
        assert (abs(sampler.products()["minima"]["a"][2] - mean[0]) < 0.01)
        assert (abs(sampler.products()["minima"]["b"][2] - mean[1]) < 0.01)
