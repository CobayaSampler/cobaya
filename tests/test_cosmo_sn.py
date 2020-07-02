from copy import deepcopy
from .test_cosmo_planck_2015 import params_lowTEB_highTTTEEE
from .common_cosmo import body_of_test


# Pantheon (alpha and beta not used - no nuisance parameters), fast
def test_sn_pantheon_camb(packages_path, skip_not_installed):
    lik = "sn.pantheon"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, chi2_sn_pantheon,
                 skip_not_installed=skip_not_installed)


def test_sn_pantheon_classy(packages_path, skip_not_installed):
    lik = "sn.pantheon"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, chi2_sn_pantheon,
                 skip_not_installed=skip_not_installed)


# JLA
def test_sn_jla_camb(packages_path, skip_not_installed):
    best_fit_test = deepcopy(params_lowTEB_highTTTEEE)
    best_fit_test.update(best_fit_sn)
    lik = "sn.jla"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(packages_path, best_fit_test, info_likelihood, info_theory, chi2_sn_jla,
                 skip_not_installed=skip_not_installed)


def test_sn_jla_classy(packages_path, skip_not_installed):
    best_fit_test = deepcopy(params_lowTEB_highTTTEEE)
    best_fit_test.update(best_fit_sn)
    lik = "sn.jla"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    body_of_test(packages_path, best_fit_test, info_likelihood, info_theory, chi2_sn_jla,
                 skip_not_installed=skip_not_installed)


# JLA marginalized over alpha, beta
def test_sn_jla_lite_camb(packages_path, skip_not_installed):
    lik = "sn.jla_lite"
    info_likelihood = {lik: {"marginalize": True}}
    info_theory = {"camb": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, chi2_sn_jla_lite,
                 skip_not_installed=skip_not_installed)


# JLA marginalized over alpha, beta (slow version!)
def test_sn_jla_lite_slow_camb(packages_path, skip_not_installed):
    lik = "sn.jla_lite"
    info_likelihood = {lik: {"marginalize": True, "precompute_covmats": False}}
    info_theory = {"camb": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, chi2_sn_jla_lite,
                 skip_not_installed=skip_not_installed)


# BEST FIT AND REFERENCE VALUES ##########################################################

best_fit = deepcopy(params_lowTEB_highTTTEEE)
best_fit_sn = {"alpha_jla": 0.1325237, "beta_jla": 2.959805}

chi2_sn_pantheon = {"sn.pantheon": 1035.30, "tolerance": 0.1}
chi2_sn_jla = {"sn.jla": 700.582, "tolerance": 0.1}
chi2_sn_jla_lite = {"sn.jla_lite": 706.882, "tolerance": 0.1}
