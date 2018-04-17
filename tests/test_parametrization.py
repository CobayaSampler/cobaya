"""
Tests different aspects of the reparametrization layer, using an external (to force hard
parameter names) gaussian likelihood.
"""

# Global
from __future__ import division
#import os
#import shutil
from scipy.stats import multivariate_normal
import numpy as np

# Local
from cobaya.conventions import _likelihood, _params, _sampler, _derived_pre
from cobaya.yaml import yaml_load
from cobaya.run import run
from cobaya.tools import get_external_function

x_func = lambda c: c/3
d_func = lambda a: a+1
b_func = "lambda a, bprime: a+2*bprime"
e_func = "lambda b: b**2"
f_func = "lambda x: 3*x"

def loglik(a,b,c, derived=["x", "d"]):
    derived.update({"x": x_func(c), "d": d_func(a)})
    return multivariate_normal.logpdf((a,b), cov=0.1*np.eye(2))

# Info
info = {
    _likelihood:
        {"test_lik": loglik},
    _sampler: {"evaluate": None},
    _params: yaml_load("""
       a: 0.01
       b: "%s"
       bprime:
         prior:
           min: -1
           max:  1
         drop: True
       c:
         prior:
           min: -1
           max:  1
       d:
       e:
         derived: "%s"
       f:
         derived: "%s"
    """%(b_func, e_func, f_func))}

def test_parametrization():
    updated_info, products = run(info)
    sample = products["sample"]
    for i, point in sample:
        a = info[_params]["a"]
        b = get_external_function(info[_params]["b"])(a, point["bprime"])
        d = get_external_function(d_func)(a)
        e = get_external_function(e_func)(b)
        f = get_external_function(info[_params]["f"]["derived"])(x_func(point["c"]))
        assert np.allclose(point[[_derived_pre+p for p in ["d", "e", "f"]]], [d, e, f])
