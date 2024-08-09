from copy import deepcopy
from .test_cosmo_planck_2015 import params_lowTEB_highTTTEEE
from .common_cosmo import body_of_test
from cobaya.typing import empty_dict


def _test_sn(packages_path, skip_not_installed, lik, theory='camb',
             lik_params=empty_dict):
    info_likelihood = {lik: lik_params}
    info_theory = {theory: None}
    ref_chi2 = {"tolerance": 0.1, lik: chi2_sn[lik]}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, ref_chi2,
                 skip_not_installed=skip_not_installed)


# Pantheon (alpha and beta not used - no nuisance parameters), fast
def test_sn_pantheon_camb(packages_path, skip_not_installed):
    _test_sn(packages_path, skip_not_installed, "sn.pantheon")


def test_sn_pantheon_classy(packages_path, skip_not_installed):
    _test_sn(packages_path, skip_not_installed, "sn.pantheon", "classy")


# JLA
def test_sn_jla_camb(packages_path, skip_not_installed):
    best_fit_test = deepcopy(params_lowTEB_highTTTEEE)
    best_fit_test.update(best_fit_sn)
    lik = "sn.jla"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    ref_chi2 = {"tolerance": 0.1, lik: chi2_sn[lik]}
    body_of_test(packages_path, best_fit_test, info_likelihood, info_theory, ref_chi2,
                 skip_not_installed=skip_not_installed)


def test_sn_jla_classy(packages_path, skip_not_installed):
    best_fit_test = deepcopy(params_lowTEB_highTTTEEE)
    best_fit_test.update(best_fit_sn)
    lik = "sn.jla"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    ref_chi2 = {"tolerance": 0.1, lik: chi2_sn[lik]}
    body_of_test(packages_path, best_fit_test, info_likelihood, info_theory, ref_chi2,
                 skip_not_installed=skip_not_installed)


# JLA marginalized over alpha, beta
def test_sn_jla_lite_camb(packages_path, skip_not_installed):
    _test_sn(packages_path, skip_not_installed, "sn.jla_lite", "camb",
             {"marginalize": True})


# JLA marginalized over alpha, beta (slow version!)
def test_sn_jla_lite_slow_camb(packages_path, skip_not_installed):
    _test_sn(packages_path, skip_not_installed, "sn.jla_lite", "camb",
             {"marginalize": True, "precompute_covmats": False})


def test_sn_pantheon_Mb(packages_path, skip_not_installed):
    best_fit_test = deepcopy(params_lowTEB_highTTTEEE)
    best_fit_test.update(best_fit_Mb)
    info_likelihood = {"sn.pantheon": {"use_abs_mag": True}, "H0.riess2020Mb": None}
    info_theory = {"camb": None}
    body_of_test(packages_path, best_fit_test, info_likelihood, info_theory,
                 chi2_sn_pantheon_Mb, skip_not_installed=skip_not_installed)


def test_sn_pantheonplus_camb(packages_path, skip_not_installed):
    _test_sn(packages_path, skip_not_installed, "sn.pantheonplus")


def test_sn_pantheonplusshoes_camb(packages_path, skip_not_installed):
    _test_sn(packages_path, skip_not_installed, "sn.pantheonplusshoes")


def test_sn_union3_camb(packages_path, skip_not_installed):
    _test_sn(packages_path, skip_not_installed, "sn.union3")


def test_sn_desy5_camb(packages_path, skip_not_installed):
    _test_sn(packages_path, skip_not_installed, "sn.desy5")


# BEST FIT AND REFERENCE VALUES ##########################################################

best_fit = deepcopy(params_lowTEB_highTTTEEE)
best_fit_sn = {"alpha_jla": 0.1325237, "beta_jla": 2.959805}
best_fit_Mb = {"Mb": -19.2}

chi2_sn_pantheon_Mb = {"sn.pantheon": 4025.30, "H0.riess2020Mb": 1.65, "tolerance": 0.1}
chi2_sn = {"sn.pantheon": 1035.30, "sn.jla": 700.582, "sn.jla_lite": 706.882,
           "sn.pantheonplus": 1403.69, "sn.pantheonplusshoes": 1496.97,
           "sn.union3": 26.31, "sn.desy5": 1644.94}
