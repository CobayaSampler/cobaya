from test_cosmo_planck_2015 import params_lowTEB_highTTTEEE
from cosmo_common import body_of_test, baseline_cosmology
from cobaya.yaml import yaml_load


def test_sdss_dr12_consensus_bao_camb(modules):
    lik = "sdss_dr12_consensus_bao"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit_base, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_bao)


def test_sdss_dr12_consensus_bao_classy(modules):
    lik = "sdss_dr12_consensus_bao"
    info_likelihood = {lik: {}}
    info_theory = {"classy": {"use_camb_names": True}}
    body_of_test(modules, best_fit_base, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_bao)


def test_sdss_dr12_consensus_full_shape_camb(modules):
    lik = "sdss_dr12_consensus_full_shape"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit_base, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_full_shape)


def test_sdss_dr12_consensus_full_shape_classy(modules):
    lik = "sdss_dr12_consensus_full_shape"
    info_likelihood = {lik: {}}
    info_theory = {"classy": {"use_camb_names": True}}
    body_of_test(modules, best_fit_base, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_full_shape)


def test_sdss_dr12_consensus_final_camb(modules):
    lik = "sdss_dr12_consensus_final"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit_base, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_final)


def test_sdss_dr12_consensus_final_classy(modules):
    lik = "sdss_dr12_consensus_final"
    info_likelihood = {lik: {}}
    info_theory = {"classy": {"use_camb_names": True}}
    body_of_test(modules, best_fit_base, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_final)


# BEST FIT AND REFERENCE VALUES ##########################################################

best_fit_base = yaml_load(baseline_cosmology)
best_fit_base.update({k:v for k,v in params_lowTEB_highTTTEEE.items()
                      if k in baseline_cosmology})

chi2_sdss_dr12_consensus_bao = {"sdss_dr12_consensus_bao": 40.8, "tolerance": 0.1}
chi2_sdss_dr12_consensus_full_shape = {"sdss_dr12_consensus_full_shape": 10, "tolerance": 0.1}
chi2_sdss_dr12_consensus_final = {"sdss_dr12_consensus_final": 10, "tolerance": 0.1}
