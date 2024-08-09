"""
Common tools for testing external priors and likelihoods.
"""

# Global
import os
import shutil
from random import random
from typing import Mapping
import numpy as np
from copy import deepcopy
import scipy.stats as stats

# Local
from cobaya.conventions import FileSuffix, get_chi2_name
from cobaya.run import run
from cobaya.yaml import yaml_load
from cobaya.tools import getfullargspec
from cobaya.likelihood import Likelihood
from cobaya.typing import InputDict
from cobaya import mpi

# Definition of external (log)pdfs

half_ring_str = "lambda x, y: stats.norm.logpdf(np.sqrt(x**2 + y**2), loc=0.5, scale=0.1)"
half_ring_func = lambda x, y: eval(half_ring_str)(x, y)

derived_funcs = {"r": lambda x, y: np.sqrt(x ** 2 + y ** 2),
                 "theta": lambda x, y: np.arctan2(x, y) / np.pi}


def half_ring_func_derived(x, y=0.5):
    derived = {p: derived_funcs[p](x, y) for p in ["r", "theta"]}
    return eval(half_ring_str)(x, y), derived


gaussian_str = "lambda y: stats.norm.logpdf(y, loc=0, scale=0.2)"
gaussian_func = lambda y: eval(gaussian_str)(y)
assert gaussian_func(0.1) == stats.norm.logpdf(0.1, loc=0, scale=0.2)


class HalfRing:

    def __init__(self):
        self.logp_func = half_ring_func

    def logp_args(self, x, y):
        return self.logp_func(x, y)

    def logp_kwargs(self, x=None, y=None):
        return self.logp_func(x, y)

    def logp_unnamed_kwargs(self, **kwargs):
        return self.logp_func(**kwargs)


half_ring_instance = HalfRing()

# Info for the different tests

info_string = {"half_ring": half_ring_str}
info_callable = {"half_ring": half_ring_func}
info_mixed = {"half_ring": half_ring_func, "gaussian_y": gaussian_str}
info_import = {"half_ring": "import_module('.common_external','tests').half_ring_func"}
info_derived = {"half_ring": {
    "external": half_ring_func_derived, "output_params": ["r", "theta"]}}
info_method_args = {"half_ring": {"external": half_ring_instance.logp_args}}
info_method_kwargs = {"half_ring": {"external": half_ring_instance.logp_kwargs}}
info_method_unnamed_kwargs = {
    "half_ring": {
        "external": half_ring_instance.logp_unnamed_kwargs, "input_params": ["x", "y"]}}


# Common part of all tests

@mpi.sync_errors
def body_of_test(info_logpdf, kind, tmpdir, derived=False, manual=False):
    rand = mpi.share(random())
    prefix = os.path.join(tmpdir, "%d" % round(1e8 * rand)) + os.sep
    if mpi.is_main_process():
        if os.path.exists(prefix):
            shutil.rmtree(prefix)
    # build updated info
    info: InputDict = {
        "output": prefix,
        "params": {
            "x": {"prior": {"min": 0, "max": 1}, "proposal": 0.05},
            "y": {"prior": {"min": -1, "max": 1}, "proposal": 0.05}},
        "sampler": {
            "mcmc": {"max_samples": (10 if not manual else 5000),
                     "learn_proposal": False}}
    }
    if derived:
        info["params"].update({"r": {"min": 0, "max": 1},
                               "theta": {"min": -0.5, "max": 0.5}})
    # Complete according to kind
    if kind == "prior":
        info.update({"prior": info_logpdf,
                     "likelihood": {"one": None}})
    elif kind == "likelihood":
        info.update({"likelihood": info_logpdf})
    else:
        raise ValueError("Kind of test not known.")
    # If there is an ext function that is not a string, don't write output!
    stringy = {k: v for k, v in info_logpdf.items() if isinstance(v, str)}
    if stringy != info_logpdf:
        info.pop("output")
    # Run
    updated_info, sampler = run(info)
    products = sampler.products()
    # Test values
    logprior_base = - np.log(
        (info["params"]["x"]["prior"]["max"] -
         info["params"]["x"]["prior"]["min"]) *
        (info["params"]["y"]["prior"]["max"] -
         info["params"]["y"]["prior"]["min"]))
    logps = {
        name: logpdf(**{arg: products["sample"][arg].to_numpy(dtype=np.float64) for arg in
                        getfullargspec(logpdf)[0]}) for name, logpdf in
        {"half_ring": half_ring_func, "gaussian_y": gaussian_func}.items()}
    # Test #1: values of logpdfs
    if kind == "prior":
        columns_priors = [c for c in products["sample"].data.columns
                          if c.startswith("minuslogprior")]
        assert np.allclose(
            products["sample"][columns_priors[0]].to_numpy(dtype=np.float64),
            np.sum(products["sample"][columns_priors[1:]].to_numpy(dtype=np.float64),
                   axis=-1)), (
            "The single prior values do not add up to the total one.")
        assert np.allclose(logprior_base + sum(logps[p] for p in info_logpdf),
                           -products["sample"]["minuslogprior"].to_numpy(
                               dtype=np.float64)), (
            "The value of the total prior is not reproduced correctly.")
        assert np.isclose(sampler.model.logprior({'x': products["sample"]["x"][0],
                                                  'y': products["sample"]["y"][0]}),
                          -products["sample"]["minuslogprior"][0]), (
            "The value of the total prior is not reproduced from mode.logprior.")
    elif kind == "likelihood":
        for lik in info["likelihood"]:
            assert np.allclose(-2 * logps[lik],
                               products["sample"][get_chi2_name(lik)].to_numpy(
                                   dtype=np.float64)), (
                    "The value of the likelihood '%s' is not reproduced correctly." % lik)
    assert np.allclose(logprior_base + sum(logps[p] for p in info_logpdf),
                       -products["sample"]["minuslogpost"].to_numpy(dtype=np.float64)), (
        "The value of the posterior is not reproduced correctly.")
    # Test derived parameters, if present -- for now just for "r"
    if derived:
        derived_values = {
            param: func(**{arg: products["sample"][arg].to_numpy(dtype=np.float64)
                           for arg in ["x", "y"]})
            for param, func in derived_funcs.items()}
        assert all(np.allclose(v, products["sample"][p].to_numpy(dtype=np.float64))
                   for p, v in derived_values.items()), (
            "The value of the derived parameters is not reproduced correctly.")
    # Test updated info -- scripted
    if kind == "prior":
        assert info["prior"] == updated_info["prior"], (
            "The prior information has not been updated correctly.")
    elif kind == "likelihood":
        # Transform the likelihood info to the "external" convention and add defaults
        info_likelihood = deepcopy(info["likelihood"])
        for lik, value in list(info_likelihood.items()):
            if not isinstance(value, Mapping):
                info_likelihood[lik] = {"external": info["likelihood"][lik]}
            # We need to restore the original non-copied callable/method for comparison
            original_like = info["likelihood"][lik]["external"] \
                if isinstance(info["likelihood"][lik], Mapping) \
                   else info["likelihood"][lik]
            info_likelihood[lik]["external"] = original_like
            info_likelihood[lik].update({k: v for k, v in
                                         Likelihood.get_defaults().items()
                                         if k not in info_likelihood[lik]})
            for k in ["input_params", "output_params"]:
                info_likelihood[lik].pop(k, None)
                updated_info["likelihood"][lik].pop(k)
        assert info_likelihood == updated_info["likelihood"], (
                "The likelihood information has not been updated correctly\n %r vs %r"
                % (info_likelihood, updated_info["likelihood"]))
    # Test updated info -- yaml
    # For now, only if ALL external pdfs are given as strings,
    # since the YAML load fails otherwise
    if stringy == info_logpdf:
        updated_output_file = os.path.join(prefix, FileSuffix.updated + ".yaml")
        with open(updated_output_file, encoding='utf-8') as updated:
            updated_yaml = yaml_load(updated.read())
        for k, v in stringy.items():
            to_test = updated_yaml[kind][k]
            if kind == "likelihood":
                to_test = to_test["external"]
            assert to_test == info_logpdf[k], (
                "The updated external pdf info has not been written correctly.")


# Plots! for the documentation -- pass `manual=True` to `body_of_test`

def plot_sample(sample, params):
    import getdist.plots as gdplt
    # noinspection PyProtectedMember
    gdsamples = sample._sampled_to_getdist()
    gdplot = gdplt.getSubplotPlotter()
    gdplot.triangle_plot(gdsamples, params, filled=True)
    gdplot.export("test.png")
