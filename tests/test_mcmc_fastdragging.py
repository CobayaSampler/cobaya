# Samples from a random Gaussian likelihood using the MCMC sampler.

from __future__ import division

import pytest
import numpy as np
from collections import OrderedDict as odict
from scipy.stats import multivariate_normal

from cobaya.conventions import _likelihood, _params, _sampler, _prior, _p_proposal, _p_ref
from cobaya.run import run


derived_func_b0 = lambda x: x**2
derived_func_b1 = lambda x: x*10
derived_func_b2 = lambda x: x+1
def loglik_slow(a0, derived={"b0": None}):
    derived["b0"] = derived_func_b1(a0)
    return 0
def loglik_correlation(a0, a1, derived={"b1": None}):
    correlation = 0.9
    derived["b1"] = derived_func_b1(a1)
    return multivariate_normal.logpdf([a0,a1], cov=[[1,correlation],[correlation,1]])
def loglik_derived(a1, derived={"b2": None}):
    derived["b2"] = derived_func_b2(a1)
    return 0

@pytest.mark.slow # just to ignore it for now!
def test_mcmc_fastdragging():
    """
    In 2d, tests a slow parameter "a0" vs a fast one "a1".
    Both share a gaussian lik with a high correlation.
    "a0" is made slower by a mock unit likelihood.
    Both likelihoods have mock derived paramters,
    just for testing that they are tracked correctly along the dragging.
    """
    info = {
        _params: odict([
            ["a0", {"prior": {"min": -4, "max": 4}, "ref": 0}],
            ["a1", {"prior": {"min": -4, "max": 4}, "ref": 0}],
            ["b0", None], ["b1", None]]),
        _likelihood: {
            "slow":        {"external": "import_module('test_mcmc_fastdragging').loglik_slow", "speed": 0.25},
            "correlation": {"external": "import_module('test_mcmc_fastdragging').loglik_correlation", "speed": 1},
            "derived":     {"external": "import_module('test_mcmc_fastdragging').loglik_derived", "speed": 1}},
        _sampler: {
            "mcmc": {
                "max_samples": np.inf, "burn_in": 0, "Rminus1_stop": 0.01, "Rminus1_cl_stop": 0.01,
                "learn_proposal": True, "covmat": np.eye(2), "covmat_params": ["a0","a1"],
                "drag_nfast_times": 5, "max_speed_slow": 0.5}}}
    info["debug"] = False
    info["debug_file"] = None
    info["output_prefix"] = "chains_test/"
    updated_info, products = run(info)
