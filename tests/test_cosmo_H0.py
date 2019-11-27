from __future__ import print_function, absolute_import
import sys
import numpy as np
from scipy.stats import norm
from copy import copy

from cobaya.conventions import kinds, _params, _path_install
from cobaya.run import run
from .common import process_modules_path

fiducial_H0 = 70


def test_H0_riess2018a(modules):
    body_of_test(modules, "H0.riess2018a")


def test_H0_riess201903(modules):
    body_of_test(modules, "H0.riess201903")


def test_H0_docs(modules):
    from cobaya.likelihoods._base_classes._H0_prototype import _H0_prototype
    doc = sys.modules[_H0_prototype.__module__].__doc__
    pre = "my_H0"
    line = next(l for l in doc.split("\n") if l.strip().startswith(pre))
    line = line[line.find("lambda"):].strip("'\"")
    line = line.replace("mu_H0", "%g" % fiducial_H0)
    line = line.replace("sigma_H0", "1")
    body_of_test(modules, line)


def body_of_test(modules, lik_name):
    info = {_path_install: process_modules_path(modules),
            kinds.sampler: {"evaluate": None}}
    if lik_name.startswith("lambda"):
        line = copy(lik_name)
        lik_name = "whatever"
        info[kinds.likelihood] = {lik_name: line}
    else:
        info[kinds.likelihood] = {lik_name: None}
    info[_params] = {"H0": fiducial_H0}
    updated_info, products = run(info)
    # The default values for .get are for the _docs_ test
    mean = updated_info[kinds.likelihood][lik_name].get("H0", fiducial_H0)
    std = updated_info[kinds.likelihood][lik_name].get("H0_std", 1)
    reference_value = -2 * norm.logpdf(fiducial_H0, loc=mean, scale=std)
    computed_value = (
        products["sample"]["chi2__" + list(info[kinds.likelihood].keys())[0]].values[0])
    assert np.allclose(computed_value, reference_value)
