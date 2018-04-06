from __future__ import print_function
import numpy as np
from scipy.stats import norm

from cobaya.conventions import _theory, _sampler, _likelihood, _params, _path_install
from cobaya.run import run


def test_cosmo_hst_riess2018_camb(modules):
    body_of_test(modules, "riess2018", "camb")


def test_cosmo_hst_riess2018_classy(modules):
    body_of_test(modules, "riess2018", "classy")


def body_of_test(modules, data, theory):
    assert modules, "I need a modules folder!"
    info = {_path_install: modules,
            _theory: {theory: None},
            _sampler: {"evaluate": None}}
    lik_name = "hst_"+data
    info[_likelihood] = {lik_name: None}
    fiducial_H0 = 70
    info[_params] = {"H0": fiducial_H0}
    updated_info, products = run(info)
    mean = updated_info[_likelihood][lik_name]["H0"]
    std = updated_info[_likelihood][lik_name]["H0_err"]
    reference_value = -2*norm.logpdf(fiducial_H0, loc=mean, scale=std)
    computed_value = (
        products["sample"]["chi2__"+list(info[_likelihood].keys())[0]].values[0])
    assert np.allclose(computed_value, reference_value)
