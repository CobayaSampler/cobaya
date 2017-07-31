"""
Tests the facility for importing external priors.

It tests all possible input methods: callable and string
(direct evaluation and ``import_module``).

In each case, it tests the correctness of the values generated, and of the updated info.

The test prior is a gaussian half-ring, combined with a gaussian in one of the tests.

For manual testing, and observing the density, set `manual=True`.
"""

# Global
from __future__ import division
import os
import shutil
import numpy as np
import scipy.stats as stats
    
# Local
from cobaya.conventions import input_params, input_prior, input_p_dist, input_sampler
from cobaya.conventions import input_likelihood, input_debug, input_output_prefix
from cobaya.conventions import output_full_suffix
from cobaya.run import run

# Set to True for manual testing
manual = False

# Prior -- evaluable string
half_ring_str = "lambda x, y: stats.norm.logpdf(np.sqrt(x**2 + y**2), loc=0.5, scale=0.1)"
gaussian_str  = "lambda y: stats.norm.logpdf(y, loc=0, scale=0.2)"

def half_ring_func(x, y):
    return eval(half_ring_str)(x, y)
def gaussian_func(y):
    return eval(gaussian_str)(y)

def test_prior_external_callable(tmpdir):
    info_prior = {"half_ring": half_ring_func}
    updated_info_prior = body_of_test(info_prior, tmpdir, test_yaml=False)
    
def test_prior_external_string(tmpdir):
    info_prior = {"half_ring": half_ring_str}
    updated_info_prior = body_of_test(info_prior, tmpdir)
    
def test_prior_external_import(tmpdir):
    info_prior = {"half_ring": "import_module('aux_prior_external').half_ring_func"}
    updated_info_prior = body_of_test(info_prior, tmpdir)
    
def test_prior_external_multiple(tmpdir):
    info_prior = {"half_ring": half_ring_func, "gaussian": gaussian_func}
    updated_info_prior = body_of_test(info_prior, tmpdir, gaussian=True)
    
def body_of_test(info_prior, tmpdir, gaussian=False, test_yaml=True):
    # For pytest's handling of tmp dirs
    if hasattr(tmpdir, "dirpath"):
        tmpdir = tmpdir.dirname
    from random import random
    prefix = os.path.join(tmpdir, "%d"%round(1e8*random()))+"/"
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    info = {
        input_output_prefix: prefix,
        input_params: {
            "x": {input_prior: {"min":  0, "max": 1}, "proposal": 0.05},
            "y": {input_prior: {"min": -1, "max": 1}, "proposal": 0.05}},
        input_likelihood: {
            "one": {"mock_prefix": ""}},
        input_sampler: {
            "mcmc": {"max_samples": (10 if not manual else 5000),
                     "learn_proposal": False}}
    }
    info[input_prior] = info_prior
    updated_info, products = run(info)
    # Test values
    x, y = products["sample"][["x", "y"]].values.T
    manual_prior = half_ring_func(x, y) - np.log(
        (info[input_params]["x"][input_prior]["max"]-
         info[input_params]["x"][input_prior]["min"])*
        (info[input_params]["y"][input_prior]["max"]-
         info[input_params]["y"][input_prior]["min"]))
    if gaussian:
        manual_prior += gaussian_func(y)
    assert np.allclose(manual_prior, -products["sample"]["minuslogprior"].values), (
        "The value of the prior is not reproduced correctly.")
    # Test updated info -- scripted
    assert updated_info[input_prior] == info[input_prior], (
        "The prior information has not been updated correctly.")
    # Test updated info -- yaml
    if test_yaml:
        full_output_file = os.path.join(prefix, output_full_suffix+".yaml")
        from cobaya.yaml_custom import yaml_custom_load
        with open(full_output_file) as full:
            assert yaml_custom_load(full)[input_prior] == info[input_prior], (
                "The prior information has not been written correctly.")

# Plots!
def plot_sample(sample, params):
    import matplotlib.pyplot as plt
    import getdist as gd
    import getdist.plots as gdplt
    gdsamples = sample.as_getdist_mcsamples()
    gdplot = gdplt.getSubplotPlotter()
    gdplot.triangle_plot(gdsamples, params, filled=True)
    gdplot.export("test.png")


if __name__ == "__main__":
#    if manual:
        from tempfile import gettempdir
        test_prior_external_callable(gettempdir())
