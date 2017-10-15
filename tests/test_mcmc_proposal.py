"""
Tests the correct speed-blocking of the proposer.
"""

from __future__ import division

from collections import OrderedDict as odict
from random import shuffle
import numpy as np

from cobaya.conventions import _likelihood, _params, _sampler
from cobaya.likelihoods.gaussian import random_cov
from cobaya.run import run

dim = 10

def test_mcmc_proposal_blocking_simple():
    body_of_test(dim, covariances=True, oversample=False, fast_slow=False)

def test_mcmc_proposal_oversampling():
    body_of_test(dim, covariances=True, oversample=True, fast_slow=False)

def body_of_test(dim, covariances=True, oversample=False, fast_slow=False):
    assert dim<99
    assert not (oversample and fast_slow), (
        "Not possible to test oversampling and fast_slow hierarchy simultaneously.")
    # Shuffle the parameters
    i_p = range(dim)
    shuffle(i_p)
    info = {_likelihood:
                odict([["l%.2d"%i, {"external": "lambda a%.2d: 1"%i, "speed": (i+1)/dim}]
                        for i in i_p]),
            _params: odict([["a%.2d"%i, {"prior": {"min": 0, "max": 1}}] for i in i_p]),
            _sampler: {"mcmc": {"callback_every": 1, "max_samples": 10, "burn_in": 0,
                                "learn_proposal": False, "oversample": oversample}}}
    if covariances:
        ranges = [[i["prior"]["min"],i["prior"]["max"]] for i in info[_params].values()]
        cov = random_cov(ranges, O_std_min=1e-2, O_std_max=1e-1)
        info[_sampler]["mcmc"].update(
            {"covmat": cov, "covmat_params": info[_params].keys()})
    def callback(sampler):
        params = sampler.parametrisation.sampled_params().keys()
        dim = len(params)
        oversampling_factors = np.array([1+(i if oversample else 0) for i in range(dim)])
        # Resetting the proposer
        sampler.proposer.cycler_all.loop_index = -1
        sampler.proposer.samples_left = 0
        punchcard = np.zeros(dim, dtype=int)
        times = 3
        for i in range((dim+sum(oversampling_factors-1))*times):
            x = np.zeros(dim)
            sampler.proposer.get_proposal(x)
            punchcard[np.where((x != 0))] += 1
        expected = [sum(oversampling_factors[j] for j in range((0 if covariances else i),i+1)) for i in range(dim)]
        assert np.all(punchcard == times*np.array(expected)[i_p])
    info["sampler"]["mcmc"].update({"callback_function": callback})
    info["debug"] = False
    updated_info, products = run(info)
