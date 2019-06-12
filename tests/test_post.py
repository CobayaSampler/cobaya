from __future__ import division, print_function, absolute_import

import os
import numpy as np
from collections import OrderedDict as odict
from scipy.stats import multivariate_normal
from flaky import flaky

from cobaya.run import run
from cobaya.post import post
from cobaya.tools import KL_norm
from getdist.mcsamples import loadMCSamples


mean = np.array([0,0])
sigma = 0.5
cov = np.array([[sigma**2, 0], [0, sigma**2]])
sampled = {"mean": mean, "cov": cov}
target = {"mean": mean + np.array([sigma/2, 0]), "cov": cov}
sampled_pdf = lambda a, b: multivariate_normal.logpdf(
    [a, b], mean=sampled["mean"], cov=sampled["cov"])
def target_pdf(a, b, c=0, _derived=["cprime"]):
    if _derived == {}:
        _derived["cprime"] = c
    return multivariate_normal.logpdf([a, b], mean=target["mean"], cov=target["cov"])

range = {"min": -2, "max": 2}
ref_pdf = {"dist": "norm", "loc": 0, "scale": 0.1}
info_params = odict([
    ["a", {"prior": range, "ref": ref_pdf, "proposal": sigma}],
    ["b", {"prior": range, "ref": ref_pdf, "proposal": sigma}],
    ["a_plus_b", {"derived": lambda a, b: a + b}]])

info_sampler = {"mcmc": {"Rminus1_stop": 0.01}}
info_sampler_dummy = {"evaluate": {"N": 10}}

@flaky(max_runs=3, min_passes=1)
def test_post_prior(tmpdir):
    # Generate original chain
    info = {
        "output": os.path.join(str(tmpdir), "gaussian"), "force": True,
        "params": info_params, "sampler": info_sampler,
        "likelihood": {"one": None}, "prior": {"gaussian": sampled_pdf}}
    run(info)
    info_post = {
        "output": info["output"], "force": True,
        "post": {"suffix": "foo",
                 "remove": {"prior": {"gaussian": None}},
                 "add": {"prior": {"target": target_pdf}}}}
    post(info_post)
    # Load with GetDist and compare
    mcsamples = loadMCSamples(info_post["output"] + "_post_" + info_post["post"]["suffix"])
    new_mean = mcsamples.mean(["a", "b"])
    new_cov = mcsamples.getCovMat().matrix
    assert abs(KL_norm(target["mean"], target["cov"], new_mean, new_cov)) < 0.02


@flaky(max_runs=3, min_passes=1)
def test_post_likelihood(tmpdir):
    # Generate original chain
    info = {
        "output": os.path.join(str(tmpdir), "gaussian"),  "force": True,
        "params": info_params, "sampler": info_sampler,
        "likelihood": {"gaussian": sampled_pdf}}
    run(info)
    info_post = {
        "output": info["output"], "force": True,
        "post": {"suffix": "foo",
                 "remove": {"likelihood": {"gaussian": None}},
                 "add": {"likelihood": {"target": target_pdf}}}}
    post(info_post)
    # Load with GetDist and compare
    mcsamples = loadMCSamples(info_post["output"] + "_post_" + info_post["post"]["suffix"])
    new_mean = mcsamples.mean(["a", "b"])
    new_cov = mcsamples.getCovMat().matrix
    assert abs(KL_norm(target["mean"], target["cov"], new_mean, new_cov)) < 0.02


def test_post_params():
    # Tests:
    # - added simple dynamical derived parameter "a+b"
    # - added dynamical derived parameter that depends on *new* chi2__target
    # - added new fixed input "c" + new derived-from-external-function "cprime"
    # Generate original chain
    info = {
        "params": info_params, "sampler": info_sampler_dummy,
        "likelihood": {"gaussian": sampled_pdf}}
    updated_info_gaussian, products_gaussian = run(info)
    info_post = {
        "post": {"suffix": "foo",
                 "remove": {"params": {"a_plus_b": None}},
                 "add": {
                     "likelihood": {"target": target_pdf},
                     "params": {
                         "c": 1.234,
                         "a_minus_b": {"derived": "lambda a,b: a-b"},
                         "my_chi2__target": {"derived": "lambda chi2__target: chi2__target"},
                         "cprime": None}}}}
    info_post.update(updated_info_gaussian)
    updated_info, products = post(info_post, products_gaussian["sample"])
    # Compare parameters
    assert np.allclose(
        products["sample"]["a"] - products["sample"]["b"], products["sample"]["a_minus_b"])
    assert np.allclose(
        products["sample"]["cprime"], info_post["post"]["add"]["params"]["c"])
    assert np.allclose(
        products["sample"]["my_chi2__target"], products["sample"]["chi2__target"])

