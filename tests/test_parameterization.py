"""
Tests different aspects of the reparameterization layer, using an external (to force hard
parameter names) gaussian likelihood.
"""

# Global
from __future__ import division
from scipy.stats import multivariate_normal
import numpy as np

# Local
from cobaya.conventions import _likelihood, _params, _sampler
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


def loglik(a, b, c, d, h, i, _derived=["x", "e"]):
    _derived.update({"x": x_func(c), "e": e_func(b)})
    return multivariate_normal.logpdf((a, b, c, d, h, i), cov=0.1 * np.eye(6))


# Info
info = {
    _likelihood:
        {"test_lik": loglik},
    _sampler: {"mcmc": {"burn_in": 0, "max_samples": 10}},
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
    """ % (b_func, c_func, f_func, g_func, h_func))}


def test_parameterization():
    updated_info, products = run(info)
    sample = products["sample"]
    from getdist.mcsamples import loadCobayaSamples
    gdsample = loadCobayaSamples(updated_info, products["sample"])
    for i, point in sample:
        a = info[_params]["a"]
        b = get_external_function(info[_params]["b"])(a, point["bprime"])
        c = get_external_function(info[_params]["c"])(a, point["cprime"])
        e = get_external_function(e_func)(b)
        f = get_external_function(f_func)(b)
        g = get_external_function(info[_params]["g"]["derived"])(x_func(point["c"]))
        h = get_external_function(info[_params]["h"])(info[_params]["i"])
        assert np.allclose(point[["b", "c", "e", "f", "g", "h"]], [b, c, e, f, g, h])
        # Test for GetDist too (except fixed ones, ignored by GetDist)
        bcefg_getdist = [gdsample.samples[i][gdsample.paramNames.list().index(p)]
                          for p in ["b", "c", "e", "f", "g"]]
        assert np.allclose(bcefg_getdist, [b, c, e, f, g])
