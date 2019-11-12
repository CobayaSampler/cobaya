"""
Common tools for testing external priors and likelihoods.
"""

# Global
from __future__ import division
import os
import shutil
from random import random
import numpy as np
import six
import scipy.stats as stats
from copy import deepcopy

# Local
from cobaya.conventions import _output_prefix, _params, _prior
from cobaya.conventions import _sampler, _likelihood, _updated_suffix
from cobaya.conventions import _chi2, _separator, _external, _input_params, _output_params
from cobaya.run import run
from cobaya.yaml import yaml_load
from cobaya.likelihood import class_options
from cobaya.tools import getfullargspec

# Definition of external (log)pdf's

half_ring_str = "lambda x, y: stats.norm.logpdf(np.sqrt(x**2 + y**2), loc=0.5, scale=0.1)"
half_ring_func = lambda x, y: eval(half_ring_str)(x, y)

derived_funcs = {"r": lambda x, y: np.sqrt(x ** 2 + y ** 2),
                 "theta": lambda x, y: np.arctan2(x, y) / np.pi}


def half_ring_func_derived(x, y=0.5, _derived=["r", "theta"]):
    _derived.update(dict([[p, derived_funcs[p](x, y)] for p in ["r", "theta"]]))
    return eval(half_ring_str)(x, y)


gaussian_str = "lambda y: stats.norm.logpdf(y, loc=0, scale=0.2)"
gaussian_func = lambda y: eval(gaussian_str)(y)

# Info for the different tests

info_string = {"half_ring": half_ring_str}
info_callable = {"half_ring": half_ring_func}
info_mixed = {"half_ring": half_ring_func, "gaussian_y": gaussian_str}
info_import = {"half_ring": "import_module('.common_external','tests').half_ring_func"}
info_derived = {"half_ring": half_ring_func_derived}


# Common part of all tests

def body_of_test(info_logpdf, kind, tmpdir, derived=False, manual=False):
    # For pytest's handling of tmp dirs
    if hasattr(tmpdir, "dirpath"):
        tmpdir = tmpdir.dirname
    prefix = os.path.join(tmpdir, "%d" % round(1e8 * random())) + os.sep
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    # build updated info
    info = {
        _output_prefix: prefix,
        _params: {
            "x": {_prior: {"min": 0, "max": 1}, "proposal": 0.05},
            "y": {_prior: {"min": -1, "max": 1}, "proposal": 0.05}},
        _sampler: {
            "mcmc": {"max_samples": (10 if not manual else 5000),
                     "learn_proposal": False}}
    }
    if derived:
        info[_params].update({"r": {"min": 0, "max": 1},
                              "theta": {"min": -0.5, "max": 0.5}})
    # Complete according to kind
    if kind == _prior:
        info.update({_prior: info_logpdf,
                     _likelihood: {"one": None}})
    elif kind == _likelihood:
        info.update({_likelihood: info_logpdf})
    else:
        raise ValueError("Kind of test not known.")
    # If there is an ext function that is not a string, don't write output!
    stringy = dict([(k, v) for k, v in info_logpdf.items() if isinstance(v, six.string_types)])
    if stringy != info_logpdf:
        info.pop(_output_prefix)
    # Run
    updated_info, products = run(info)
    # Test values
    logprior_base = - np.log(
        (info[_params]["x"][_prior]["max"] -
         info[_params]["x"][_prior]["min"]) *
        (info[_params]["y"][_prior]["max"] -
         info[_params]["y"][_prior]["min"]))
    logps = dict([
        (name, logpdf(**dict([(arg, products["sample"][arg].values) for arg in getfullargspec(logpdf)[0]])))
        for name, logpdf in {"half_ring": half_ring_func, "gaussian_y": gaussian_func}.items()])
    # Test #1: values of logpdf's
    if kind == _prior:
        columns_priors = [c for c in products["sample"].data.columns
                          if c.startswith("minuslogprior")]
        assert np.allclose(
            products["sample"][columns_priors[0]].values,
            np.sum(products["sample"][columns_priors[1:]].values, axis=-1)), (
            "The single prior values do not add up to the total one.")
        assert np.allclose(logprior_base + sum(logps[p] for p in info_logpdf),
                           -products["sample"]["minuslogprior"].values), (
            "The value of the total prior is not reproduced correctly.")
    elif kind == _likelihood:
        for lik in info[_likelihood]:
            assert np.allclose(-2 * logps[lik],
                               products["sample"][_chi2 + _separator + lik].values), (
                    "The value of the likelihood '%s' is not reproduced correctly." % lik)
    assert np.allclose(logprior_base + sum(logps[p] for p in info_logpdf),
                       -products["sample"]["minuslogpost"].values), (
        "The value of the posterior is not reproduced correctly.")
    # Test derived parameters, if present -- for now just for "r"
    if derived:
        derived_values = dict([
            (param, func(**dict([(arg, products["sample"][arg].values) for arg in ["x", "y"]])))
            for param, func in derived_funcs.items()])
        assert np.all(
            [np.allclose(v, products["sample"][p].values)
             for p, v in derived_values.items()]), (
            "The value of the derived parameters is not reproduced correctly.")
    # Test updated info -- scripted
    if kind == _prior:
        assert info[_prior] == updated_info[_prior], (
            "The prior information has not been updated correctly.")
    elif kind == _likelihood:
        # Transform the likelihood info to the "external" convention and add defaults
        info_likelihood = deepcopy(info[_likelihood])
        for lik, value in list(info_likelihood.items()):
            if not hasattr(value, "get"):
                info_likelihood[lik] = {_external: value}
            info_likelihood[lik].update({k: v for k, v in class_options.items()
                                         if k not in info_likelihood[lik]})
            for k in [_input_params, _output_params]:
                updated_info[_likelihood][lik].pop(k)
        assert info_likelihood == dict(updated_info[_likelihood]), (
                "The likelihood information has not been updated correctly\n %r vs %r" % (
            info_likelihood, dict(updated_info[_likelihood])))
    # Test updated info -- yaml
    # For now, only if ALL external pdfs are given as strings, since the YAML load fails otherwise
    if stringy == info_logpdf:
        updated_output_file = os.path.join(prefix, _updated_suffix + ".yaml")
        with open(updated_output_file) as updated:
            updated_yaml = yaml_load("".join(updated.readlines()))
        for k, v in stringy.items():
            to_test = updated_yaml[kind][k]
            if kind == _likelihood:
                to_test = to_test[_external]
            assert to_test == info_logpdf[k], (
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
