"""
Tests the correct speed-blocking of the proposer.
"""

from __future__ import division

from collections import OrderedDict as odict
from random import shuffle, choice
import numpy as np

from cobaya.conventions import _likelihood, _params, _sampler
from cobaya.likelihoods.gaussian import random_cov
from cobaya.run import run

# Dimensionality for the tests (it shouldn't matter)
dim = 4  # < 100


def test_mcmc_proposal_blocking_simple():
    body_of_test(dim, covariances=True)


def test_mcmc_proposal_oversampling():
    body_of_test(dim, covariances=True, oversample=True)


# named this way because we use "slow" for slow test
def test_mcmc_proposal_fastSlow():
    body_of_test(dim, covariances=True, fast_slow=True)


def body_of_test(dim, covariances=True, oversample=False, fast_slow=False):
    assert dim < 99, "Dimensionality too high."
    assert not (oversample and fast_slow), (
        "Not possible to test oversampling and fast_slow hierarchy simultaneously.")
    # Shuffle the parameters
    i_p = list(range(dim))
    shuffle(i_p)
    info = {_likelihood: odict([["l%.2d"%i,
                                 {"external": "lambda a%.2d: 1"%i, "speed": (i+1)/dim}]
                                for i in i_p]),
            _params: odict([["a%.2d"%i, {"prior": {"min": 0, "max": 1}}] for i in i_p]),
            _sampler: {"mcmc": {"callback_every": 1, "max_samples": 10, "burn_in": 0,
                                "learn_proposal": False, "oversample": oversample,
                                "drag_nfast_times": (1 if fast_slow else None)}}}
    speeds = [v["speed"] for v in info[_likelihood].values()]
    oversampling_factors = np.array([1+(i if oversample else 0) for i in range(dim)])
    if fast_slow:
        speed_cut_i = choice(range(dim-1))
        max_speed_slow = speeds[i_p.index(speed_cut_i)]
        info[_sampler]["mcmc"]["max_speed_slow"] = max_speed_slow
        dim_slow, dim_fast = 1 + speed_cut_i, dim - (1 + speed_cut_i)
    if covariances:
        ranges = [[i["prior"]["min"],i["prior"]["max"]] for i in info[_params].values()]
        cov = random_cov(ranges, O_std_min=1e-2, O_std_max=1e-1)
        info[_sampler]["mcmc"].update(
            {"covmat": cov, "covmat_params": info[_params].keys()})
    def callback_oversampling(sampler):
        # Resetting the proposer
        sampler.proposer.cycler_all.loop_index = -1
        sampler.proposer.samples_left = 0
        punchcard = np.zeros(dim, dtype=int)
        times = 3
        for i in range((dim+sum(oversampling_factors-1))*times):
            x = np.zeros(dim)
            sampler.proposer.get_proposal(x)
            punchcard[np.where((x != 0))] += 1
        expected = [sum(oversampling_factors[j]
                        for j in range((0 if covariances else i),i+1))
                    for i in range(dim)]
        assert np.all(punchcard == times*np.array(expected)[i_p])
    def callback_fast_slow(sampler):
        # Resetting the proposer
        sampler.proposer.cycler_slow.loop_index = -1
        sampler.proposer.cycler_fast.loop_index = -1
        times = 3
        # fast
        punchcard = np.zeros(dim, dtype=int)
        for i in range(dim_fast*times):
            x = np.zeros(dim)
            sampler.proposer.get_proposal_fast(x)
            punchcard[np.where((x != 0))] += 1
        expected = ([0] * dim_slow +
                    [i+1 for i in range(dim_fast)])
        assert np.all(punchcard == times*np.array(expected)[i_p])
        # slow
        punchcard = np.zeros(dim, dtype=int)
        for i in range(dim_slow*times):
            x = np.zeros(dim)
            sampler.proposer.get_proposal_slow(x)
            print(x, speeds, max_speed_slow)
            punchcard[np.where((x != 0))] += 1
        expected = ([i+1 for i in range(dim_slow)] +
                    [dim_slow] * dim_fast)
        assert np.all(punchcard == times*np.array(expected)[i_p])
    info["sampler"]["mcmc"]["callback_function"] = (
        callback_fast_slow if fast_slow else callback_oversampling)
    info["debug"] = False
    updated_info, products = run(info)
