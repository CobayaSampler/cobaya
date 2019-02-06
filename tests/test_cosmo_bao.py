from __future__ import absolute_import
from copy import deepcopy
import pytest

from .test_cosmo_planck_2015 import params_lowTEB_highTTTEEE
from .common_cosmo import body_of_test


def test_sdss_dr12_consensus_bao_camb(modules):
    lik = "sdss_dr12_consensus_bao"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_bao)


def test_sdss_dr12_consensus_bao_classy(modules):
    lik = "sdss_dr12_consensus_bao"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_bao)


def test_sdss_dr12_consensus_full_shape_camb(modules):
    lik = "sdss_dr12_consensus_full_shape"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_full_shape)


@pytest.mark.skip
def test_sdss_dr12_consensus_full_shape_classy(modules):
    lik = "sdss_dr12_consensus_full_shape"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    chi2_classy = deepcopy(chi2_sdss_dr12_consensus_full_shape)
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_classy)


def test_sdss_dr12_consensus_final_camb(modules):
    lik = "sdss_dr12_consensus_final"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr12_consensus_final)


@pytest.mark.skip
def test_sdss_dr12_consensus_final_classy(modules):
    lik = "sdss_dr12_consensus_final"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    chi2_classy = deepcopy(chi2_sdss_dr12_consensus_final)
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_classy)


def test_sixdf_2011_bao_camb(modules):
    lik = "sixdf_2011_bao"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_sixdf_2011_bao)


def test_sixdf_2011_bao_classy(modules):
    lik = "sixdf_2011_bao"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_sixdf_2011_bao)


def test_sdss_dr7_mgs_camb(modules):
    lik = "sdss_dr7_mgs"
    info_likelihood = {lik: {}}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr7_mgs)


def test_sdss_dr7_mgs_classy(modules):
    lik = "sdss_dr7_mgs"
    info_likelihood = {lik: {}}
    info_theory = {"classy": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_sdss_dr7_mgs)


# BEST FIT AND REFERENCE VALUES ##########################################################

best_fit = deepcopy(params_lowTEB_highTTTEEE)

chi2_sdss_dr12_consensus_bao = {
    "sdss_dr12_consensus_bao": 5.687, "tolerance": 0.04}
chi2_sdss_dr12_consensus_full_shape = {
    "sdss_dr12_consensus_full_shape": 8.154, "tolerance": 0.02}
chi2_sdss_dr12_consensus_final = {
    "sdss_dr12_consensus_final": 8.051, "tolerance": 0.02}
chi2_sixdf_2011_bao = {
    "sixdf_2011_bao": 0.088, "tolerance": 0.02}
chi2_sdss_dr7_mgs = {
    "sdss_dr7_mgs": 0.92689, "tolerance": 0.02}
