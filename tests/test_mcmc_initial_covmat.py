"""
Tests if the initial covariance matrix of the MCMC proposal is build with the right
inheritance order.
"""

import os
import numpy as np
from itertools import chain

from cobaya.likelihoods.gaussian_mixture import random_cov
from cobaya.typing import InputDict
from cobaya.run import run
from cobaya.sampler import CovmatSampler
from cobaya import mpi


def test_mcmc_initial_covmat_interactive():
    dim = 40
    body_of_test(dim)


def test_mcmc_initial_covmat_yaml(tmpdir):
    dim = 40
    body_of_test(dim, tmpdir=tmpdir)


def body_of_test(dim, tmpdir=None, random_state=None):
    mindim = 4
    assert dim > mindim, "Needs dimension>%d for the test." % mindim
    if mpi.is_main_process():
        random_state = np.random.default_rng(random_state)
        i_s = list(range(dim))
        random_state.shuffle(i_s)
        initial_random_covmat = random_cov(dim * [[0, 1]], random_state=random_state)
        mpi.share((i_s, initial_random_covmat))
    else:
        i_s, initial_random_covmat = mpi.share()

    n_altered = int(dim / 4)
    i_proposal = i_s[:n_altered]
    i_ref = i_s[n_altered:2 * n_altered]
    i_prior = i_s[2 * n_altered:3 * n_altered]
    removed = list(chain(*(i_proposal, i_ref, i_prior)))
    i_covmat = [i for i in range(dim) if i not in removed]
    for i in removed:
        diag = initial_random_covmat[i, i]
        initial_random_covmat[:, i] = 0
        initial_random_covmat[i, :] = 0
        initial_random_covmat[i, i] = diag
    # Prepare info, including refs, priors and reduced covmat
    prefix = "a_"
    if mpi.is_main_process():
        input_order = list(range(dim))
        random_state.shuffle(input_order)
    else:
        input_order = None
    input_order = mpi.share(input_order)
    info: InputDict = {"likelihood": {"one": None}, "params": {}}
    fallback_scale = np.sqrt(CovmatSampler.fallback_covmat_scale)
    for i in input_order:
        p = prefix + str(i)
        info["params"][p] = {"prior": {"dist": "norm", "loc": 0, "scale": 1000}}
        sigma = np.sqrt(initial_random_covmat[i, i])
        if i in i_proposal:
            info["params"][p]["proposal"] = sigma
        elif i in i_ref:
            info["params"][prefix + str(i)]["ref"] = {"dist": "norm",
                                                      "scale": sigma * fallback_scale}
        elif i in i_prior:
            info["params"][prefix + str(i)]["prior"]["scale"] = sigma * fallback_scale
    reduced_covmat = initial_random_covmat[np.ix_(i_covmat, i_covmat)]
    reduced_covmat_params = [prefix + str(i) for i in i_covmat]
    info["sampler"] = {"mcmc": {}}
    if tmpdir:
        filename = os.path.join(str(tmpdir), "mycovmat.dat")
        header = " ".join(reduced_covmat_params)
        np.savetxt(filename, reduced_covmat, header=header)
        info["sampler"]["mcmc"]["covmat"] = str(filename)
    else:
        info["sampler"]["mcmc"]["covmat_params"] = reduced_covmat_params
        info["sampler"]["mcmc"]["covmat"] = reduced_covmat
    to_compare = initial_random_covmat[np.ix_(input_order, input_order)]

    def callback(sampler):
        assert np.allclose(to_compare, sampler.proposer.get_covariance())

    info["sampler"]["mcmc"].update({
        "callback_function": callback, "callback_every": 1, "max_samples": 1,
        "burn_in": 0})
    run(info)
