# Minimization of a random Gaussian likelihood using the minimize sampler.

import os
import warnings

import numpy as np
import pytest

from cobaya import InputDict, Likelihood, mpi, run
from cobaya.component import ComponentNotInstalledError
from cobaya.samplers.minimize import valid_methods

pytestmark = pytest.mark.mpi

mean = np.array([0.30245268, 0.61884443, 0.5])
cov = np.array(
    [
        [0.00796336, -0.0014805, -0.00479433],
        [-0.0014805, 0.00561415, 0.00434189],
        [-0.00479433, 0.00434189, 0.03208593],
    ]
)
_inv_cov = np.linalg.inv(cov)


class NoisyCovLike(Likelihood):
    params = {
        "a": [0, 1, 0.5, 0.3, 0.08],
        "b": [0, 1, 0.5, 0.3, 0.08],
        "c": [0, 1, 0.5, 0.3, 0.08],
    }
    noise = 0

    def logp(self, **params_values):
        x = np.array([params_values["a"], params_values["b"], params_values["c"]]) - mean
        return -_inv_cov.dot(x).dot(x) / 2 + self.noise * np.random.random_sample()


@mpi.sync_errors
def test_minimize_gaussian(tmpdir, skip_not_installed):
    maxloglik = 0
    for method in reversed(valid_methods):
        NoisyCovLike.noise = 0.005 if method == "bobyqa" else 0
        info: InputDict = {
            "likelihood": {"like": NoisyCovLike},
            "sampler": {"minimize": {"ignore_prior": True, "method": method}},
        }
        try:
            products = run(info)[1].products()
        except ComponentNotInstalledError:
            if skip_not_installed:
                warnings.warn(
                    f"Could not test minimization method '{method}': not installed."
                )
                continue
            raise
        error = abs(maxloglik - -products["minimum"]["minuslogpost"])
        assert error < 0.01
        info["output"] = os.path.join(tmpdir, "testmin")
        products = run(info, force=True)[1].products()
        if mpi.is_main_process():
            from getdist.types import BestFit

            res = BestFit(info["output"] + ".bestfit").getParamDict()
            assert np.isclose(res["loglike"], products["minimum"]["minuslogpost"])
            for p, v in list(res.items())[:-2]:
                assert np.isclose(products["minimum"][p], v)
            assert isinstance(products["full_set_of_mins"], dict)


@mpi.sync_errors
def test_minimize_single_point(tmpdir, skip_not_installed):
    if mpi.get_mpi_size() > 1:
        return
    for method in reversed(valid_methods):
        NoisyCovLike.noise = 0.005 if method == "bobyqa" else 0
        info: InputDict = {
            "likelihood": {"like": NoisyCovLike},
            "sampler": {
                "minimize": {"ignore_prior": True, "method": method, "best_of": 1}
            },
            "output": os.path.join(tmpdir, "testmin"),
        }
        try:
            products = run(info, force=True)[1].products()
        except ComponentNotInstalledError:
            if skip_not_installed:
                warnings.warn(
                    f"Could not test minimization method '{method}': not installed."
                )
                continue
            raise
        if mpi.is_main_process():
            assert products["full_set_of_mins"] is None


@mpi.sync_errors
def test_run_minimize(tmpdir):
    NoisyCovLike.noise = 0
    info: InputDict = {
        "likelihood": {"like": NoisyCovLike},
        "sampler": {"mcmc": {"Rminus1_stop": 0.5, "Rminus1_cl_stop": 0.4, "seed": 2}},
        "output": os.path.join(tmpdir, "testchain"),
    }
    # No need to use skip_not_installed: uses default method, which should be requisite
    run(info, force=True)
    min_info: InputDict = dict(info, sampler={"minimize": None})
    output_info, sampler = run(min_info, force=True)
    assert abs(sampler.products()["minimum"]["b"] - mean[1]) < 0.01
