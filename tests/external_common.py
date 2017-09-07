"""
Common tools for testing external priors and likelihoods.
"""

# Global
from __future__ import division
import os
import shutil
from random import random
import numpy as np
import scipy.stats as stats
import inspect

# Local
from cobaya.conventions import input_output_prefix, input_params, input_prior
from cobaya.conventions import input_sampler, input_likelihood, output_full_suffix
from cobaya.conventions import _chi2, separator, input_likelihood_external
from cobaya.run import run
from cobaya.yaml_custom import yaml_custom_load

# Definition of external (log)pdf's

half_ring_str  = "lambda x, y: stats.norm.logpdf(np.sqrt(x**2 + y**2), loc=0.5, scale=0.1)"
half_ring_func = lambda x, y: eval(half_ring_str)(x, y)

gaussian_str  = "lambda y: stats.norm.logpdf(y, loc=0, scale=0.2)"
gaussian_func = lambda y: eval(gaussian_str)(y)

# Info for the different tests

info_string   = {"half_ring": half_ring_str}
info_callable = {"half_ring": half_ring_func}
info_mixed    = {"half_ring": half_ring_func, "gaussian_y": gaussian_str}
info_import   = {"half_ring": "import_module('external_common').half_ring_func"}

# Common part of all tests

def body_of_test(info_logpdf, kind, tmpdir, manual=False):
    # For pytest's handling of tmp dirs
    if hasattr(tmpdir, "dirpath"):
        tmpdir = tmpdir.dirname
    prefix = os.path.join(tmpdir, "%d"%round(1e8*random()))+"/"
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    # build full info
    info = {
        input_output_prefix: prefix,
        input_params: {
            "x": {input_prior: {"min":  0, "max": 1}, "proposal": 0.05},
            "y": {input_prior: {"min": -1, "max": 1}, "proposal": 0.05}},
        input_sampler: {
            "mcmc": {"max_samples": (10 if not manual else 5000),
                     "learn_proposal": False}}
    }
    # Complete according to kind
    if kind == input_prior:
        info.update({input_prior: info_logpdf,
                     input_likelihood: {"one": {"mock_prefix": ""}}})
    elif kind == input_likelihood:
        info.update({input_likelihood: info_logpdf})
    else:
        raise ValueError("Kind of test not known.")
    # Run
    updated_info, products = run(info)
    # Test values
    logprior_base = - np.log(
        (info[input_params]["x"][input_prior]["max"]-
         info[input_params]["x"][input_prior]["min"])*
        (info[input_params]["y"][input_prior]["max"]-
         info[input_params]["y"][input_prior]["min"]))
    logps = dict([
        (name, logpdf(**dict([(arg, products["sample"][arg].values) for arg in inspect.getargspec(logpdf)[0]])))
        for name, logpdf in {"half_ring":half_ring_func, "gaussian_y":gaussian_func}.iteritems()])
    # Test #1: values of logpdf's
    if kind == input_prior:
        assert np.allclose(logprior_base+sum(logps[p] for p in info_logpdf),
                           -products["sample"]["minuslogprior"].values), (
            "The value of the prior is not reproduced correctly.")
    elif kind == input_likelihood:
        for lik in info[input_likelihood]:
            assert np.allclose(-2*logps[lik],
                               products["sample"][_chi2+separator+lik].values), (
                "The value of the likelihood '%s' is not reproduced correctly."%lik)
    assert np.allclose(logprior_base+sum(logps[p] for p in info_logpdf),
                       -products["sample"]["minuslogpost"].values), (
        "The value of the posterior is not reproduced correctly.")
    # Test updated info -- scripted
    assert info.get(input_prior, None) == updated_info.get(input_prior, None), (
        "The prior information has not been updated correctly.")
    # Transform the likelihood info to the "external" convention
    info_likelihood = info[input_likelihood]
    for lik, value in info_likelihood.iteritems():
        if not hasattr(value, "get"):
            info_likelihood[lik] = {input_likelihood_external: value}
    assert info_likelihood == updated_info[input_likelihood], (
        "The likelihood information has not been updated correctly.")



    # Test updated info -- yaml (for now, only if ALL external pdfs are given as strings!)
    stringy = dict([(k,v) for k,v in info_logpdf.iteritems() if isinstance(v, basestring)])
    if stringy == info_logpdf:
        full_output_file = os.path.join(prefix, output_full_suffix+".yaml")
        with open(full_output_file) as full:
            updated_yaml = yaml_custom_load("".join(full.readlines()))
        for k,v in stringy.iteritems():
            assert updated_yaml[kind][k] == info_logpdf[k], (
                "The updated external pdf info has not been written correctly.")


# Plots! for the documentation -- pass `manual=True` to `body_of_test`

def plot_sample(sample, params):
    import matplotlib.pyplot as plt
    import getdist as gd
    import getdist.plots as gdplt
    gdsamples = sample.as_getdist_mcsamples()
    gdplot = gdplt.getSubplotPlotter()
    gdplot.triangle_plot(gdsamples, params, filled=True)
    gdplot.export("test.png")
