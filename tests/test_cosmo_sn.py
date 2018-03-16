from copy import deepcopy

from test_cosmo_planck_2015 import params_lowTEB_highTTTEEE
from cosmo_common import body_of_test, baseline_cosmology
from cobaya.yaml import yaml_load


# Pantheon (alpha and beta not used - no nuisance parameters), fast
def test_sn_pantheon_camb(modules):
    lik = "sn_pantheon"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit_base, info_likelihood, info_theory, chi2_sn_pantheon)


def test_sn_pantheon_classy(modules):
    lik = "sn_pantheon"
    info_likelihood = {lik: {}}
    info_theory = {"classy": {"use_camb_names": True}}
    body_of_test(modules, best_fit_base, info_likelihood, info_theory, chi2_sn_pantheon)


# JLA
def test_sn_jla_camb(modules):
    best_fit = deepcopy(best_fit_base)
    best_fit.update(best_fit_sn)
    lik = "sn_jla"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory, chi2_sn_jla)


def test_sn_jla_classy(modules):
    best_fit = deepcopy(best_fit_base)
    best_fit.update(best_fit_sn)
    lik = "sn_jla"
    info_likelihood = {lik: {}}
    info_theory = {"classy": {"use_camb_names": True}}
    body_of_test(modules, best_fit, info_likelihood, info_theory, chi2_sn_jla)


# JLA marginalized over alpha, beta
def test_sn_jla_lite_camb(modules):
    best_fit = deepcopy(best_fit_base)
    lik = "sn_jla_lite"
    info_likelihood = {lik: {"marginalize": True}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory, chi2_sn_jla_lite)


# JLA marginalized over alpha, beta (slow version!)
def test_sn_jla_lite_slow_camb(modules):
    best_fit = deepcopy(best_fit_base)
    lik = "sn_jla_lite"
    info_likelihood = {lik: {"marginalize": True, "precompute_covmats": False}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory, chi2_sn_jla_lite)


# BEST FIT AND REFERENCE VALUES ##########################################################

best_fit_base = yaml_load(baseline_cosmology)
best_fit_base.update({k:v for k,v in params_lowTEB_highTTTEEE.items()
                      if k in baseline_cosmology})
best_fit_sn = {"alpha_jla": 0.1325237, "beta_jla": 2.959805}

chi2_sn_pantheon = {"sn_pantheon": 1036.6, "tolerance": 0.1}
chi2_sn_jla = {"sn_jla": 700.582, "tolerance": 0.1}
chi2_sn_jla_lite = {"sn_jla_lite": 706.882, "tolerance": 0.1}
