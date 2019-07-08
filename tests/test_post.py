from __future__ import division, print_function, absolute_import

import os
import numpy as np
from collections import OrderedDict as odict
from scipy.stats import multivariate_normal
from flaky import flaky

from cobaya.run import run
from cobaya.post import post
from cobaya.tools import KL_norm
from cobaya.conventions import _output_prefix, _params, _force, _likelihood, _sampler
from cobaya.conventions import _prior, _p_dist, _p_proposal, _p_derived, _separator_files
from cobaya.conventions import _post, _post_add, _post_remove, _post_suffix
from getdist.mcsamples import loadMCSamples

_post_ = _separator_files + _post + _separator_files

mean = np.array([0, 0])
sigma = 0.5
cov = np.array([[sigma ** 2, 0], [0, sigma ** 2]])
sampled = {"mean": mean, "cov": cov}
target = {"mean": mean + np.array([sigma / 2, 0]), "cov": cov}
sampled_pdf = lambda a, b: multivariate_normal.logpdf(
    [a, b], mean=sampled["mean"], cov=sampled["cov"])


def target_pdf(a, b, c=0, _derived=["cprime"]):
    if _derived == {}:
        _derived["cprime"] = c
    return multivariate_normal.logpdf([a, b], mean=target["mean"], cov=target["cov"])


range = {"min": -2, "max": 2}
ref_pdf = {_p_dist: "norm", "loc": 0, "scale": 0.1}
info_params = odict([
    ["a", {"prior": range, "ref": ref_pdf, _p_proposal: sigma}],
    ["b", {"prior": range, "ref": ref_pdf, _p_proposal: sigma}],
    ["a_plus_b", {_p_derived: lambda a, b: a + b}]])

info_sampler = {"mcmc": {"Rminus1_stop": 0.01}}
info_sampler_dummy = {"evaluate": {"N": 10}}


@flaky(max_runs=3, min_passes=1)
def test_post_prior(tmpdir):
    # Generate original chain
    info = {
        _output_prefix: os.path.join(str(tmpdir), "gaussian"), _force: True,
        _params: info_params, _sampler: info_sampler,
        _likelihood: {"one": None}, _prior: {"gaussian": sampled_pdf}}
    run(info)
    info_post = {
        _output_prefix: info[_output_prefix], _force: True,
        _post: {_post_suffix: "foo",
                _post_remove: {_prior: {"gaussian": None}},
                _post_add: {_prior: {"target": target_pdf}}}}
    post(info_post)
    # Load with GetDist and compare
    mcsamples = loadMCSamples(
        info_post[_output_prefix] + _post_ + info_post[_post][_post_suffix])
    new_mean = mcsamples.mean(["a", "b"])
    new_cov = mcsamples.getCovMat().matrix
    assert abs(KL_norm(target["mean"], target["cov"], new_mean, new_cov)) < 0.02


@flaky(max_runs=3, min_passes=1)
def test_post_likelihood(tmpdir):
    # Generate original chain
    info = {
        _output_prefix: os.path.join(str(tmpdir), "gaussian"), _force: True,
        _params: info_params, _sampler: info_sampler,
        _likelihood: {"gaussian": sampled_pdf}}
    run(info)
    info_post = {
        _output_prefix: info[_output_prefix], _force: True,
        _post: {_post_suffix: "foo",
                _post_remove: {_likelihood: {"gaussian": None}},
                _post_add: {_likelihood: {"target": target_pdf}}}}
    post(info_post)
    # Load with GetDist and compare
    mcsamples = loadMCSamples(
        info_post[_output_prefix] + _post_ + info_post[_post][_post_suffix])
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
        _params: info_params, _sampler: info_sampler_dummy,
        _likelihood: {"gaussian": sampled_pdf}}
    updated_info_gaussian, products_gaussian = run(info)
    info_post = {
        _post: {_post_suffix: "foo",
                _post_remove: {_params: {"a_plus_b": None}},
                _post_add: {
                    _likelihood: {"target": target_pdf},
                    _params: {
                        "c": 1.234,
                        "a_minus_b": {_p_derived: "lambda a,b: a-b"},
                        "my_chi2__target": {_p_derived: "lambda chi2__target: chi2__target"},
                        "cprime": None}}}}
    info_post.update(updated_info_gaussian)
    updated_info, products = post(info_post, products_gaussian["sample"])
    # Compare parameters
    assert np.allclose(
        products["sample"]["a"] - products["sample"]["b"], products["sample"]["a_minus_b"])
    assert np.allclose(
        products["sample"]["cprime"], info_post[_post][_post_add][_params]["c"])
    assert np.allclose(
        products["sample"]["my_chi2__target"], products["sample"]["chi2__target"])
