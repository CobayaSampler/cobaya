"""
Tests different aspects of the reparameterization layer, using an external (to force hard
parameter names) gaussian likelihood.
"""

# Global
from scipy.stats import multivariate_normal
import numpy as np
# Local
from cobaya.conventions import kinds, _params
from cobaya.yaml import yaml_load
from cobaya.run import run
from cobaya.tools import get_external_function

x_func = lambda _: _ / 3
e_func = lambda _: _ + 1
b_func = "lambda a, bprime: a+2*bprime"
c_func = "lambda a, cprime: a+3*cprime"
f_func = "lambda b: b**2"
g_func = "lambda x: 3*x"
h_func = "lambda i: i**2"
j_func = "lambda b: b**2"
k_func = "lambda f: f**3"


def loglike(a, b, c, d, h, i, j):
    logp = multivariate_normal.logpdf((a, b, c, d, h, i, j), cov=0.1 * np.eye(7))
    derived = {"x": x_func(c), "e": e_func(b)}
    return logp, derived


# Info
info = {
    kinds.likelihood:
        {"test_lik": {"external": loglike, "output_params": ["x", "e"]}},
    kinds.sampler: {"mcmc": {"burn_in": 0, "max_samples": 10}},
    _params: yaml_load("""
       # Fixed to number
       a: 0.01
       # Fixed to function, non-explicitly requested as derived
       b: "%s"
       # Fixed to function, explicitly requested as derived
       c:
         value: "%s"
         derived: True
       # Sampled, dynamically defined from functions
       bprime:
         prior:
           min: -1
           max:  1
         drop: True
         proposal: 0.0001
       cprime:
         prior:
           min: -1
           max:  1
         drop: True
         proposal: 0.0001
       # Simple sampled parameter
       d:
         prior:
           min: -1
           max:  1
         proposal: 0.0001
       # Simple derived parameter
       e:
       # Dynamical derived parameter
       f:
         derived: "%s"
       # Dynamical derived parameter, needing non-mentioned output parameter (x)
       g:
         derived: "%s"
       # Fixing parameter whose only role is being an argument for a different one
       h: "%s"
       i: 2
       # Multi-layer: input parameter of "2nd order", i.e. dep on another dyn input
       j: "%s"
       # Multi-layer derived parameter of "2nd order", i.e. depe on another dyn derived
       k:
         derived: "%s"
    """ % (b_func, c_func, f_func, g_func, h_func, j_func, k_func))}


def test_parameterization():
    updated_info, sampler = run(info)
    products = sampler.products()
    sample = products["sample"]
    from getdist.mcsamples import MCSamplesFromCobaya
    gdsample = MCSamplesFromCobaya(updated_info, products["sample"])
    for i, point in sample:
        a = info[_params]["a"]
        b = get_external_function(info[_params]["b"])(a, point["bprime"])
        c = get_external_function(info[_params]["c"])(a, point["cprime"])
        e = get_external_function(e_func)(b)
        f = get_external_function(f_func)(b)
        g = get_external_function(info[_params]["g"]["derived"])(x_func(point["c"]))
        h = get_external_function(info[_params]["h"])(info[_params]["i"])
        j = get_external_function(info[_params]["j"])(b)
        k = get_external_function(info[_params]["k"]["derived"])(f)
        assert np.allclose(
            point[["b", "c", "e", "f", "g", "h", "j", "k"]], [b, c, e, f, g, h, j, k])
        # Test for GetDist too (except fixed ones, ignored by GetDist)
        bcefffg_getdist = [gdsample.samples[i][gdsample.paramNames.list().index(p)]
                           for p in ["b", "c", "e", "f", "g", "j", "k"]]
        assert np.allclose(bcefffg_getdist, [b, c, e, f, g, j, k])


# MARKED FOR DEPRECATION IN v3.0 -- Everything below this line

from typing import Sequence, Union

DerivedArg = Union[dict, Sequence, None]

def loglik_OLD(a, b, c, d, h, i, j, _derived: DerivedArg = ("x", "e")):
    if isinstance(_derived, dict):
        _derived.update({"x": x_func(c), "e": e_func(b)})
    return multivariate_normal.logpdf((a, b, c, d, h, i, j), cov=0.1 * np.eye(7))

# MARKED FOR DEPRECATION IN v3.0
info_OLD = info.copy()
info_OLD[kinds.likelihood] = {"test_lik": loglik_OLD}


# MARKED FOR DEPRECATION IN v3.0
def test_parameterization_old_derived_specification():
    updated_info, sampler = run(info_OLD)
    products = sampler.products()
    sample = products["sample"]
    from getdist.mcsamples import MCSamplesFromCobaya
    gdsample = MCSamplesFromCobaya(updated_info, products["sample"])
    for i, point in sample:
        a = info[_params]["a"]
        b = get_external_function(info[_params]["b"])(a, point["bprime"])
        c = get_external_function(info[_params]["c"])(a, point["cprime"])
        e = get_external_function(e_func)(b)
        f = get_external_function(f_func)(b)
        g = get_external_function(info[_params]["g"]["derived"])(x_func(point["c"]))
        h = get_external_function(info[_params]["h"])(info[_params]["i"])
        j = get_external_function(info[_params]["j"])(b)
        k = get_external_function(info[_params]["k"]["derived"])(f)
        assert np.allclose(
            point[["b", "c", "e", "f", "g", "h", "j", "k"]], [b, c, e, f, g, h, j, k])
        # Test for GetDist too (except fixed ones, ignored by GetDist)
        bcefffg_getdist = [gdsample.samples[i][gdsample.paramNames.list().index(p)]
                           for p in ["b", "c", "e", "f", "g", "j", "k"]]
        assert np.allclose(bcefffg_getdist, [b, c, e, f, g, j, k])
