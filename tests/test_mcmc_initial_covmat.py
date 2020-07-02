"""
Tests if the initial covariance matrix of the MCMC proposal is build with the right
inheritance order.
"""

import os
import numpy as np
from itertools import chain
from random import shuffle

from cobaya.likelihoods.gaussian_mixture import random_cov
from cobaya.conventions import kinds, partag,_prior, _params
from cobaya.run import run


def test_mcmc_initial_covmat_interactive():
    dim = 40
    body_of_test(dim)


def test_mcmc_initial_covmat_yaml(tmpdir):
    dim = 40
    body_of_test(dim, tmpdir=tmpdir)


def body_of_test(dim, tmpdir=None):
    mindim = 4
    assert dim > mindim, "Needs dimension>%d for the test." % mindim
    initial_random_covmat = random_cov(dim * [[0, 1]])
    i_s = list(range(dim))
    shuffle(i_s)
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
    input_order = list(range(dim))
    shuffle(input_order)
    info = {kinds.likelihood: {"one": None}, _params: {}}
    for i in input_order:
        p = prefix + str(i)
        info[_params][p] = {_prior: {partag.dist: "norm", "loc": 0, "scale": 1000}}
        sigma = np.sqrt(initial_random_covmat[i, i])
        if i in i_proposal:
            info[_params][p][partag.proposal] = sigma
        elif i in i_ref:
            info[_params][prefix + str(i)][partag.ref] = {partag.dist: "norm", "scale": sigma}
        elif i in i_prior:
            info[_params][prefix + str(i)][_prior]["scale"] = sigma
    reduced_covmat = initial_random_covmat[np.ix_(i_covmat, i_covmat)]
    reduced_covmat_params = [prefix + str(i) for i in i_covmat]
    info[kinds.sampler] = {"mcmc": {}}
    if tmpdir:
        filename = os.path.join(str(tmpdir), "mycovmat.dat")
        header = " ".join(reduced_covmat_params)
        np.savetxt(filename, reduced_covmat, header=header)
        info[kinds.sampler]["mcmc"]["covmat"] = str(filename)
    else:
        info[kinds.sampler]["mcmc"]["covmat_params"] = reduced_covmat_params
        info[kinds.sampler]["mcmc"]["covmat"] = reduced_covmat
    to_compare = initial_random_covmat[np.ix_(input_order, input_order)]

    def callback(sampler):
        assert np.allclose(to_compare, sampler.proposer.get_covariance())

    info[kinds.sampler]["mcmc"].update({
        "callback_function": callback, "callback_every": 1, "max_samples": 1, "burn_in": 0})
    updated_info, products = run(info)
