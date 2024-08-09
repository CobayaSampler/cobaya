"""
Tests different aspects of the reparameterization layer, using an external (to force hard
parameter names) gaussian likelihood.
"""

# Global
import pytest
from scipy.stats import multivariate_normal
import numpy as np
# Local
from cobaya.yaml import yaml_load
from cobaya.run import run
from cobaya.tools import get_external_function
from cobaya.likelihood import Likelihood
from cobaya.model import get_model
from cobaya.log import LoggedError

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
    "likelihood":
        {"test_lik": {"external": loglike, "output_params": ["x", "e"]}},
    "sampler": {"mcmc": {"burn_in": 0, "max_samples": 10}},
    "params": yaml_load("""
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
    gdsample = products["sample"].to_getdist()
    for i, point in sample:
        a = info["params"]["a"]
        b = get_external_function(info["params"]["b"])(a, point["bprime"])
        c = get_external_function(info["params"]["c"])(a, point["cprime"])
        e = get_external_function(e_func)(b)
        f = get_external_function(f_func)(b)
        g = get_external_function(info["params"]["g"]["derived"])(x_func(point["c"]))
        h = get_external_function(info["params"]["h"])(info["params"]["i"])
        j = get_external_function(info["params"]["j"])(b)
        k = get_external_function(info["params"]["k"]["derived"])(f)
        assert np.allclose(
            point[["b", "c", "e", "f", "g", "h", "j", "k"]].to_numpy(np.float64),
            [b, c, e, f, g, h, j, k])
        # Test for GetDist too (except fixed ones, ignored by GetDist)
        bcefffg_getdist = [gdsample.samples[i][gdsample.paramNames.list().index(p)]
                           for p in ["b", "c", "e", "f", "g", "j", "k"]]
        assert np.allclose(bcefffg_getdist, [b, c, e, f, g, j, k])


def test_parameterization_dependencies():
    class TestLike(Likelihood):
        params = {'a': None, 'b': None}

        def get_can_provide_params(self):
            return ['D']

        def logp(self, **params_values):
            a = params_values['a']
            b = params_values['b']
            params_values['_derived']['D'] = -7
            return a + 100 * b

    info_yaml = r"""
    params:
      aa:  
        prior: [2,4]
      bb:
        prior: [0,1]
        ref: [0.5, 0.1]
      c:
        value: "lambda aa, bb: aa+bb"  
      a: 
        value: "lambda c, aa: c*aa"  
      b: 1
      D:
      E:
       derived: "lambda D,c,a,aa: D*c/a+aa"      
    prior:
      pr: "lambda bb, a: bb-10*a"

    stop_at_error: True
    """
    test_info = yaml_load(info_yaml)
    test_info["likelihood"] = {"Like": TestLike}

    model = get_model(test_info)
    assert np.isclose(model.loglike({'bb': 0.5, 'aa': 2})[0], 105)
    assert np.isclose(model.logposterior({'bb': 0.5, 'aa': 2}).logpriors[1], -49.5)
    test_info['params']['b'] = {'value': 'lambda a, c, bb: a*c*bb'}
    like, derived = get_model(test_info).loglike({'bb': 0.5, 'aa': 2})
    assert np.isclose(like, 630)
    assert derived == [2.5, 5.0, 6.25, -7, -1.5]
    assert np.isclose(model.logposterior({'bb': 0.5, 'aa': 2}).logpriors[1], -49.5)
    test_info['params']['aa'] = 2
    test_info['params']['bb'] = 0.5
    like, derived = get_model(test_info).loglike()
    assert np.isclose(like, 630)
    assert derived == [2.5, 5.0, 6.25, -7, -1.5]

    test_info["prior"]["on_derived"] = "lambda f: 5*f"
    with pytest.raises(LoggedError) as e:
        get_model(test_info)
    assert "found and don't have a default value either" in str(e.value)

    # currently don't allow priors on derived parameters
    test_info["prior"]["on_derived"] = "lambda E: 5*E"
    with pytest.raises(LoggedError) as e:
        get_model(test_info)
    assert "that are output derived parameters" in str(e.value)
