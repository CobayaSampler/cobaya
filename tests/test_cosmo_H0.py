from __future__ import print_function
import sys
import numpy as np
from scipy.stats import norm
from copy import copy

from cobaya.conventions import _theory, _sampler, _likelihood, _params, _path_install
from cobaya.run import run
from install_for_tests import process_modules_path

fiducial_H0 = 70


def test_H0_riess2018a_camb(modules):
    body_of_test(modules, "H0_riess2018a", "camb")


def test_H0_riess2018b_classy(modules):
    body_of_test(modules, "H0_riess2018b", "classy")


def test_H0_docs_camb(modules):
    from cobaya.likelihoods._H0_prototype import _H0_prototype
    doc = sys.modules[_H0_prototype.__module__].__doc__
    pre = "my_H0"
    line = next(l for l in doc.split("\n") if l.strip().startswith(pre))
    line = line[line.find("lambda"):].strip("'\"")
    line = line.replace("mu_H0", "%g" % fiducial_H0)
    line = line.replace("sigma_H0", "1")
    body_of_test(modules, line, "camb")


def body_of_test(modules, lik_name, theory):
    info = {_path_install: process_modules_path(modules),
            _theory: {theory: None},
            _sampler: {"evaluate": None}}
    if lik_name.startswith("lambda"):
        line = copy(lik_name)
        lik_name = "whatever"
        info[_likelihood] = {lik_name: line}
    else:
        info[_likelihood] = {lik_name: None}
    info[_params] = {"H0": fiducial_H0}
    updated_info, products = run(info)
    # The default values for .get are for the _docs_ test
    mean = updated_info[_likelihood][lik_name].get("H0", fiducial_H0)
    std = updated_info[_likelihood][lik_name].get("H0_std", 1)
    reference_value = -2 * norm.logpdf(fiducial_H0, loc=mean, scale=std)
    computed_value = (
        products["sample"]["chi2__" + list(info[_likelihood].keys())[0]].values[0])
    assert np.allclose(computed_value, reference_value)
