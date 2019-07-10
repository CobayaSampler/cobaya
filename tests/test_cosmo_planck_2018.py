# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CAMB and CLASS
from __future__ import absolute_import
import pytest
from copy import deepcopy

from .common_cosmo import body_of_test
from cobaya.cosmo_input import cmb_precision

# Generating plots in Travis
import matplotlib

matplotlib.use('agg')

# Derived parameters not understood by CLASS
# https://wiki.cosmos.esa.int/planckpla2015/images/b/b9/Parameter_tag_definitions_2015.pdf
classy_unknown = ["zstar", "rstar", "thetastar", "DAstar", "zdrag",
                  "kd", "thetad", "zeq", "keq", "thetaeq", "thetarseq",
                  "DH", "Y_p"]

# Small chi2 difference with CLASS (total still <0.5)
classy_extra_tolerance = 0.2


def test_planck_2018_t_camb(modules):
    best_fit = params_lowl_highTT_lensing
    info_likelihood = lik_info_lowl_highTT_lensing
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    best_fit_derived = derived_lowl_highTT_lensing
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_lowl_highTT_lensing, best_fit_derived)


def test_planck_2018_p_camb(modules):
    best_fit = params_lowTE_highTTTEEE_lensingcmblikes
    info_likelihood = lik_info_lowTE_highTTTEEE_lensingcmblikes
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    best_fit_derived = derived_lowTE_highTTTEEE_lensingcmblikes
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                  chi2_lowTE_highTTTEEE_lensingcmblikes, best_fit_derived)


def test_planck_2018_t_classy(modules):
    best_fit = params_lowl_highTT_lensing
    info_likelihood = lik_info_lowl_highTT_lensing
    info_theory = {"classy": {"extra_args": cmb_precision["classy"]}}
    best_fit_derived = deepcopy(derived_lowl_highTT_lensing)
    for p in classy_unknown:
        best_fit_derived.pop(p, None)
    chi2_lowl_highTT_classy = deepcopy(chi2_lowl_highTT_lensing)
    chi2_lowl_highTT_classy["tolerance"] += classy_extra_tolerance
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_lowl_highTT_classy, best_fit_derived)


def test_planck_2018_p_classy(modules):
    best_fit = params_lowTE_highTTTEEE_lensingcmblikes
    info_likelihood = lik_info_lowTE_highTTTEEE_lensingcmblikes
    info_theory = {"classy": {"extra_args": cmb_precision["classy"]}}
    best_fit_derived = deepcopy(derived_lowTE_highTTTEEE_lensingcmblikes)
    for p in classy_unknown:
        best_fit_derived.pop(p, None)
    chi2_lowl_highTT_classy = deepcopy(chi2_lowTE_highTTTEEE_lensingcmblikes)
    chi2_lowl_highTT_classy["tolerance"] += classy_extra_tolerance
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_lowl_highTT_classy, best_fit_derived)


# Temperature only #######################################################################

lik_info_lowl_highTT_lensing = {
    "planck_2018_lowl": None, "planck_2018_plikHM_TT": None, "planck_2018_lensing": None}

chi2_lowl_highTT_lensing = {
    "planck_2018_lowl": 22.92,
    "planck_2018_plikHM_TT": 757.77,
    "planck_2018_lensing": 9.11,
    "tolerance": 0.11}

params_lowl_highTT_lensing = {
    # Sampled
    "omegabh2": 0.02240,
    "omegach2": 0.1172,
    # only one of the next two is finally used!
    "H0": 68.45,  # will be ignored in the CAMB case
    "theta": 1.04117,  # will be ignored in the CLASS case
    "tau": 0.0862,
    "logA": 3.100,
    "ns": 0.9733,
    # Planck likelihood
    "A_planck": 1.00008,
    "A_cib_217": 45.6,
    "xi_sz_cib": 0.73,
    "A_sz": 6.92,
    "ps_A_100_100": 246,
    "ps_A_143_143": 51.3,
    "ps_A_143_217": 54.7,
    "ps_A_217_217": 122.1,
    "ksz_norm": 0.01,
    "gal545_A_100": 8.87,
    "gal545_A_143": 10.79,
    "gal545_A_143_217": 19.8,
    "gal545_A_217": 95.4,
    "calib_100T": 0.99965,
    "calib_217T": 0.99822}

derived_lowl_highTT_lensing = {
    # param: [best_fit, sigma]
    "H0": [params_lowl_highTT_lensing["H0"], 1.2],
    "omegal": [0.7006, 0.015],
    "omegam": [0.2994, 0.016],
    "sigma8": [0.8271, 0.013],
    "zrei": [10.61, 2.2],
    "YHe": [0.245408, 0.0001],
    "Y_p": [0.246735, 0.0001],
    "DH": [2.5794e-5, 0.05e-5],
    "age": [13.7744, 0.047],
    "zstar": [1089.63, 0.52],
    "rstar": [145.13, 0.55],
    "thetastar": [1.04136, 0.00050],
    "DAstar": [13.9364, 0.050],
    "zdrag": [1059.818, 0.51],
    "rdrag": [147.79, 0.53],
    "kd": [0.14015, 0.00053],
    "thetad": [0.160844, 0.00029],
    "zeq": [3337, 57],
    "keq": [0.010184, 0.00017],
    "thetaeq": [0.8255, 0.011],
    "thetarseq": [0.4557, 0.0057]}

# Best fit Polarization ##################################################################

lik_info_lowTE_highTTTEEE_lensingcmblikes = {
    "planck_2018_lowl": None, "planck_2018_lowE": None, "planck_2018_plikHM_TTTEEE": None,
    "planck_2018_lensing": None}

chi2_lowTE_highTTTEEE_lensingcmblikes = {
    "planck_2018_lowl": 23.25, "planck_2018_lowE": 396.05,
    "planck_2018_plikHM_TTTEEE": 2344.93, "planck_2018_lensing": 8.87,
    "tolerance": 0.11}

params_lowTE_highTTTEEE_lensingcmblikes = {
    # Sampled
    "omegabh2": 0.022383,
    "omegach2": 0.12011,
    # only one of the next two is finally used!
    "H0": 67.32,  # will be ignored in the CAMB case
    "theta": 1.040909,  # will be ignored in the CLASS case
    "logA": 3.0448,
    "ns": 0.96605,
    "tau": 0.0543,
    # Planck likelihood
    "A_planck": 1.00044,
    "A_cib_217": 46.1,
    "xi_sz_cib": 0.66,
    "A_sz": 7.08,
    "ps_A_100_100": 248.2,
    "ps_A_143_143": 50.7,
    "ps_A_143_217": 53.3,
    "ps_A_217_217": 121.9,
    "ksz_norm": 0.00,
    "gal545_A_100": 8.80,
    "gal545_A_143": 11.01,
    "gal545_A_143_217": 20.16,
    "gal545_A_217": 95.5,
    "galf_TE_A_100": 0.1138,
    "galf_TE_A_100_143": 0.1346,
    "galf_TE_A_100_217": 0.479,
    "galf_TE_A_143": 0.225,
    "galf_TE_A_143_217": 0.665,
    "galf_TE_A_217": 2.082,
    "calib_100T": 0.99974,
    "calib_217T": 0.99819}

derived_lowTE_highTTTEEE_lensingcmblikes = {
    # param: [best_fit, sigma]
    "H0": [params_lowTE_highTTTEEE_lensingcmblikes["H0"], 0.54],
    "omegal": [0.6842, 0.0073],
    "omegam": [0.3158, 0.0073],
    "sigma8": [0.8120, 0.0060],
    "zrei": [7.68, 7.67],
    "YHe": [0.245401, 0.000057],
    "Y_p": [0.246727, 0.000058],
    "DH": [2.5831e-5, 0.027e-5],
    "age": [13.7971, 0.023],
    "zstar": [1089.914, 0.25],
    "rstar": [144.394, 0.26],
    "thetastar": [1.041085, 0.00031],
    "DAstar": [13.8696, 0.025],
    "zdrag": [1059.971, 0.30],
    "rdrag": [147.049, 0.26],
    "kd": [0.140922, 0.00030],
    "thetad": [0.160734, 0.00017],
    "zeq": [3405.1, 26],
    "keq": [0.010393, 0.000081],
    "thetaeq": [0.81281, 0.0050],
    "thetarseq": [0.44912, 0.0026]}
