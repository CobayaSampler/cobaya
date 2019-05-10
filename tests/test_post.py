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
target_pdf = lambda a, b: multivariate_normal.logpdf(
    [a, b], mean=target["mean"], cov=target["cov"])

range = {"min": -2, "max": 2}
info_params = odict([
    ["a", {"prior": range, "ref": 0, "proposal": sigma}],
    ["b", {"prior": range, "ref": 0, "proposal": sigma}],
    ["a_plus_b", {"derived": lambda a, b: a + b}]])

info_sampler = {"mcmc": {"Rminus1_stop": 0.01}}


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
    assert abs(KL_norm(target["mean"], target["cov"], new_mean, new_cov)) < 0.01
