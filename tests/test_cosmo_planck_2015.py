# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CAMB and CLASS

import pytest
from copy import deepcopy

from cosmo_common import body_of_test


def test_planck_2015_t_camb(modules):
    best_fit = params_lowl_highTT
    info_likelihood = lik_info_lowl_highTT
    info_theory = {"camb": None}
    best_fit_derived = derived_lowl_highTT
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_lowl_highTT, best_fit_derived)


def test_planck_2015_p_camb(modules):
    best_fit = params_lowTEB_highTTTEEE
    info_likelihood = lik_info_lowTEB_highTTTEEE
    info_theory = {"camb": None}
    best_fit_derived = derived_lowTEB_highTTTEEE
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_lowTEB_highTTTEEE, best_fit_derived)


def test_planck_2015_l_camb(modules):
    best_fit = params_lensing
    info_likelihood = lik_info_lensing
    info_theory = {"camb": None}
    best_fit_derived = derived_lensing
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_lensing, best_fit_derived)


@pytest.mark.skip
def test_planck_2015_t_classy(modules):
    best_fit = params_lowl_highTT
    info_likelihood = lik_info_lowl_highTT
    info_theory = {"classy": None}
    best_fit_derived = derived_lowl_highTT
    chi2_lowl_highTT_classy = deepcopy(chi2_lowl_highTT)
    chi2_lowl_highTT_classy["tolerance"] += 2.1
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_lowl_highTT_classy, best_fit_derived)


@pytest.mark.skip
def test_planck_2015_p_classy(modules):
    best_fit = params_lowTEB_highTTTEEE
    info_likelihood = lik_info_lowTEB_highTTTEEE
    info_theory = {"classy": None}
    best_fit_derived = derived_lowTEB_highTTTEEE
    chi2_lowTEB_highTTTEEE_classy = deepcopy(chi2_lowTEB_highTTTEEE)
    chi2_lowTEB_highTTTEEE_classy["tolerance"] += 2.1
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_lowTEB_highTTTEEE_classy, best_fit_derived)


@pytest.mark.skip
def test_planck_2015_l_classy(modules):
    best_fit = params_lensing
    info_likelihood = lik_info_lensing
    info_theory = {"classy": None}
    best_fit_derived = derived_lensing
    chi2_lensing_classy = deepcopy(chi2_lensing)
#    chi2_lowTEB_lensing_classy["tolerance"] += 2.1
    body_of_test(modules, best_fit, info_likelihood, info_theory,
                 chi2_lensing, best_fit_derived)


# Temperature only #######################################################################

lik_info_lowl_highTT = {"planck_2015_lowl": None, "planck_2015_plikHM_TT": None}

chi2_lowl_highTT = {"planck_2015_lowl": 15.39,
                    "planck_2015_plikHM_TT": 761.09,
                    "tolerance": 0.1}

params_lowl_highTT = {
    # Sampled
    "ombh2": 0.02249139,
    "omch2": 0.1174684,
    # only one of the next two is finally used!
    "H0": 68.43994,  # will be ignored in the CAMB case
    "cosmomc_theta_100": 1.041189,  # will be ignored in the CLASS case
    "tau": 0.1249913,
    "logAs1e10": 3.179,
    "ns": 0.9741693,
    # Planck likelihood
    "A_planck": 1.00027,
    "A_cib_217": 61.1,
    "xi_sz_cib": 0.56,
    "A_sz": 6.84,
    "ps_A_100_100": 242.9,
    "ps_A_143_143":  43.0,
    "ps_A_143_217":  46.1,
    "ps_A_217_217": 104.1,
    "ksz_norm": 0.00,
    "gal545_A_100":      7.31,
    "gal545_A_143":      9.07,
    "gal545_A_143_217": 17.99,
    "gal545_A_217":     82.9,
    "calib_100T": 0.99796,
    "calib_217T": 0.99555}

derived_lowl_highTT = {
    # param: [best_fit, sigma]
    "H0": [68.44, 1.2],
    "omegav": [0.6998, 0.016],
    "omegam": [0.3002, 0.016],
    "omegamh2": [0.1406, 0.0024],
    "omegamh3": [0.09623, 0.00046],
    "sigma8": [0.8610, 0.023],
    "s8omegamp5": [0.472, 0.014],
    "s8omegamp25":[0.637, 0.016],
    "s8h5":       [1.041, 0.025],
    "zre":   [13.76, 2.5],
    "As1e9":  [2.40, 0.15],
    "clamp":  [1.870468, 0.01535354],
    "YHe":    [0.2454462, 0.0001219630],
    "Y_p":    [0.2467729, 0.0001224069],
    "DH":     [2.568606e-5, 0.05098625e-5],
    "age":    [13.7664, 0.048],
    "zstar":  [1089.55, 0.52],
    "rstar":  [145.00, 0.55],
    "thetastar": [1.041358, 0.0005117986],
    "DAstar":  [13.924, 0.050],
    "zdrag":   [1060.05, 0.52],
    "rdrag":   [147.63, 0.53],
    "kd":      [0.14039, 0.00053],
    "thetad":  [0.160715, 0.00029],
    "zeq":     [3345, 58],
    "keq":     [0.010208, 0.00018],
    "thetaeq":  [0.8243, 0.011],
    "thetarseq": [0.4550, 0.0058],
    }


# Best fit Polarization ##################################################################

lik_info_lowTEB_highTTTEEE = {"planck_2015_lowTEB": None,
                              "planck_2015_plikHM_TTTEEE": None}

chi2_lowTEB_highTTTEEE = {"planck_2015_lowTEB": 10496.93,
                          "planck_2015_plikHM_TTTEEE": 2431.65,
                          "tolerance": 0.1}

params_lowTEB_highTTTEEE = {
    # Sampled
    "ombh2": 0.02225203,
    "omch2": 0.1198657,
    # only one of the next two is finally used!
    "H0": 67.25,  # will be ignored in the CAMB case
    "cosmomc_theta_100": 1.040778,  # will be ignored in the CLASS case
    "logAs1e10": 3.0929,
    "ns": 0.9647522,
    "tau": 0.07888604,
    # Planck likelihood
    "A_planck": 1.00029,
    "A_cib_217": 66.4,
    "xi_sz_cib": 0.13,
    "A_sz": 7.17,
    "ps_A_100_100": 255.0,
    "ps_A_143_143": 40.1,
    "ps_A_143_217": 36.4,
    "ps_A_217_217": 98.7,
    "ksz_norm": 0.00,
    "gal545_A_100": 7.34,
    "gal545_A_143": 8.97,
    "gal545_A_143_217": 17.56,
    "gal545_A_217": 81.9,
    "galf_EE_A_100": 0.0813,
    "galf_EE_A_100_143": 0.0488,
    "galf_EE_A_100_217": 0.0995,
    "galf_EE_A_143": 0.1002,
    "galf_EE_A_143_217": 0.2236,
    "galf_EE_A_217": 0.645,
    "galf_TE_A_100": 0.1417,
    "galf_TE_A_100_143": 0.1321,
    "galf_TE_A_100_217": 0.307,
    "galf_TE_A_143": 0.155,
    "galf_TE_A_143_217": 0.338,
    "galf_TE_A_217": 1.667,
    "calib_100T": 0.99818,
    "calib_217T": 0.99598}

derived_lowTEB_highTTTEEE = {
    # param: [best_fit, sigma]
    "H0": [67.25, 0.66],
    "omegav": [0.6844, 0.0091],
    "omegam": [0.3156, 0.0091],
    "omegamh2": [0.14276, 0.0014],
    "omegamh3": [0.096013, 0.00029],
    "sigma8": [0.8310, 0.013],
    "s8omegamp5": [0.4669, 0.0098],
    "s8omegamp25":[0.6228, 0.011],
    "s8h5":       [1.0133, 0.017],
    "zre":   [10.07, 1.6],
    "As1e9":  [2.204, 0.074],
    "clamp":  [1.8824, 0.012],
    "YHe":    [0.2453409, 0.000072],
    "Y_p":    [0.2466672, 0.000072],
    "DH":     [2.6136e-5, 0.030e-5],
    "age":    [13.8133, 0.026],
    "zstar":  [1090.057, 0.30],
    "rstar":  [144.556, 0.32],
    "thetastar": [1.040967, 0.00032],
    "DAstar":  [13.8867, 0.030],
    "zdrag":   [1059.666, 0.31],
    "rdrag":   [147.257, 0.31],
    "kd":      [0.140600, 0.00032],
    "thetad":  [0.160904, 0.00018],
    "zeq":     [3396.2, 33],
    "keq":     [0.010365, 0.00010],
    "thetaeq":  [0.8139, 0.0063],
    "thetarseq": [0.44980, 0.0032],
    }


# Best fit lensing (from best combination with lowTEB+_highTTTEEE ########################

lik_info_lensing = {"planck_2015_lensing": None}

chi2_lensing = {"planck_2015_lensing": 9.78, "tolerance": 0.1}

params_lensing = {
    # Sampled
    "ombh2": 0.022274,
    "omch2": 0.11913,
    # only one of the next two is finally used!
    "H0": 67.56,  # will be ignored in the CAMB case
    "cosmomc_theta_100": 1.040867,  # will be ignored in the CLASS case
    "logAs1e10": 3.0600,
    "ns": 0.96597,
    "tau": 0.0639,
    # Planck likelihood
    "A_planck": 0.99995}

derived_lensing = {
    # param: [best_fit, sigma]
    "H0": [67.56, 0.64],
    "omegav": [0.6888, 0.0087],
    "omegam": [0.3112, 0.0087],
    "omegamh2": [0.14205, 0.0013],
    "omegamh3": [0.095971, 0.00030],
    "sigma8": [0.8153, 0.0087],
    "s8omegamp5": [0.4548, 0.0068],
    "s8omegamp25":[0.6089, 0.0067],
    "s8h5":       [0.9919, 0.010],
    "zre":    [8.64, 1.3],
    "As1e9":  [2.133, 0.053],
    "clamp":  [1.8769, 0.011],
    "YHe":    [0.245350, 0.000071],
    "Y_p":    [0.246677, 0.000072],
    "DH":     [2.6095e-5, 0.030e-5],
    "age":    [13.8051, 0.026],
    "zstar":  [1089.966, 0.29],
    "rstar":  [144.730, 0.31],
    "thetastar": [1.041062, 0.00031],
    "DAstar":  [13.9022, 0.029],
    "zdrag":   [1059.666, 0.31],
    "rdrag":   [147.428, 0.30],
    "kd":      [0.140437, 0.00032],
    "thetad":  [0.160911, 0.00018],
    "zeq":     [3379.1, 32],
    "keq":     [0.010313, 0.000096],
    "thetaeq":  [0.8171, 0.0060],
    "thetarseq": [0.45146, 0.0031],
    }
