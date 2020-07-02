from copy import deepcopy
import pytest

from .test_cosmo_planck_2015 import params_lowTEB_highTTTEEE
from .common_cosmo import body_of_test


def test_sdss_dr12_consensus_bao_camb(packages_path, skip_not_installed):
    lik = "bao.sdss_dr12_consensus_bao"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_bao, skip_not_installed=skip_not_installed)


def test_sdss_dr12_consensus_bao_classy(packages_path, skip_not_installed):
    lik = "bao.sdss_dr12_consensus_bao"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_bao, skip_not_installed=skip_not_installed)


def test_sdss_dr12_consensus_full_shape_camb(packages_path, skip_not_installed):
    lik = "bao.sdss_dr12_consensus_full_shape"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_full_shape,
                 skip_not_installed=skip_not_installed)


@pytest.mark.skip
def test_sdss_dr12_consensus_full_shape_classy(packages_path, skip_not_installed):
    lik = "bao.sdss_dr12_consensus_full_shape"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    chi2_classy = deepcopy(chi2_sdss_dr12_consensus_full_shape)
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_classy, skip_not_installed=skip_not_installed)


def test_sdss_dr12_consensus_final_camb(packages_path, skip_not_installed):
    lik = "bao.sdss_dr12_consensus_final"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_final, skip_not_installed=skip_not_installed)


@pytest.mark.skip
def test_sdss_dr12_consensus_final_classy(packages_path, skip_not_installed):
    lik = "bao.sdss_dr12_consensus_final"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    chi2_classy = deepcopy(chi2_sdss_dr12_consensus_final)
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_classy, skip_not_installed=skip_not_installed)


def test_sixdf_2011_bao_camb(packages_path, skip_not_installed):
    lik = "bao.sixdf_2011_bao"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_sixdf_2011_bao, skip_not_installed=skip_not_installed)


def test_sixdf_2011_bao_classy(packages_path, skip_not_installed):
    lik = "bao.sixdf_2011_bao"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_sixdf_2011_bao, skip_not_installed=skip_not_installed)


def test_sdss_dr7_mgs_camb(packages_path, skip_not_installed):
    lik = "bao.sdss_dr7_mgs"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr7_mgs, skip_not_installed=skip_not_installed)


def test_sdss_dr7_mgs_classy(packages_path, skip_not_installed):
    lik = "bao.sdss_dr7_mgs"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr7_mgs, skip_not_installed=skip_not_installed)


# BEST FIT AND REFERENCE VALUES ##########################################################

best_fit = deepcopy(params_lowTEB_highTTTEEE)

chi2_sdss_dr12_consensus_bao = {
    "bao.sdss_dr12_consensus_bao": 5.687, "tolerance": 0.04}
chi2_sdss_dr12_consensus_full_shape = {
    "bao.sdss_dr12_consensus_full_shape": 8.154, "tolerance": 0.02}
chi2_sdss_dr12_consensus_final = {
    "bao.sdss_dr12_consensus_final": 8.051, "tolerance": 0.03}
chi2_sixdf_2011_bao = {
    "bao.sixdf_2011_bao": 0.088, "tolerance": 0.02}
chi2_sdss_dr7_mgs = {
    "bao.sdss_dr7_mgs": 0.92689, "tolerance": 0.02}
