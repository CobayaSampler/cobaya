# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CAMB and CLASS
from copy import deepcopy

from .common_cosmo import body_of_test
from cobaya.cosmo_input import cmb_precision

# Generating plots in Travis
import matplotlib

matplotlib.use('agg')

# Downgrade of Planck 2018 precision/model

cmb_precision = deepcopy(cmb_precision)
cmb_precision["camb"].update({
    "halofit_version": "takahashi",
    "bbn_predictor": "BBN_fitting_parthenope"
})
cmb_precision["classy"].update({
    "non linear": "halofit",
})

# Derived parameters not understood by CLASS
# https://wiki.cosmos.esa.int/planckpla2015/images/b/b9/Parameter_tag_definitions_2015.pdf
classy_unknown = ["zstar", "rstar", "thetastar", "DAstar", "zdrag",
                  "kd", "thetad", "zeq", "keq", "thetaeq", "thetarseq",
                  "DH", "Y_p"]

# Small chi2 difference with CLASS (total still <0.5)
classy_extra_tolerance = 0.2


def test_planck_2015_t_camb(packages_path, skip_not_installed):
    best_fit = deepcopy(params_lowl_highTT)
    best_fit.pop("H0")
    info_likelihood = lik_info_lowl_highTT
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    best_fit_derived = derived_lowl_highTT
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_lowl_highTT, best_fit_derived,
                 skip_not_installed=skip_not_installed)


def test_planck_2015_p_camb(packages_path, skip_not_installed):
    best_fit = deepcopy(params_lowTEB_highTTTEEE)
    best_fit.pop("H0")
    info_likelihood = lik_info_lowTEB_highTTTEEE
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    best_fit_derived = derived_lowTEB_highTTTEEE
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_lowTEB_highTTTEEE, best_fit_derived,
                 skip_not_installed=skip_not_installed)


def test_planck_2015_l_camb(packages_path, skip_not_installed):
    best_fit = deepcopy(params_lensing)
    best_fit.pop("H0")
    info_likelihood = lik_info_lensing
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    best_fit_derived = derived_lensing
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_lensing, best_fit_derived,
                 skip_not_installed=skip_not_installed)


def test_planck_2015_l2_camb(packages_path, skip_not_installed):
    best_fit = deepcopy(params_lensing)
    best_fit.pop("H0")
    lik_name = "planck_2015_lensing_cmblikes"
    clik_name = "planck_2015_lensing"
    info_likelihood = {lik_name: lik_info_lensing[clik_name]}
    chi2_lensing_cmblikes = deepcopy(chi2_lensing)
    chi2_lensing_cmblikes[lik_name] = chi2_lensing[clik_name]
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    best_fit_derived = derived_lensing
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_lensing_cmblikes, best_fit_derived,
                 skip_not_installed=skip_not_installed)


def test_planck_2015_t_classy(packages_path, skip_not_installed):
    best_fit = deepcopy(params_lowl_highTT)
    best_fit.pop("theta_MC_100")
    info_likelihood = lik_info_lowl_highTT
    info_theory = {"classy": {"extra_args": cmb_precision["classy"]}}
    best_fit_derived = deepcopy(derived_lowl_highTT)
    for p in classy_unknown:
        best_fit_derived.pop(p, None)
    chi2_lowl_highTT_classy = deepcopy(chi2_lowl_highTT)
    chi2_lowl_highTT_classy["tolerance"] += classy_extra_tolerance
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_lowl_highTT_classy, best_fit_derived,
                 skip_not_installed=skip_not_installed)


def test_planck_2015_p_classy(packages_path, skip_not_installed):
    best_fit = deepcopy(params_lowTEB_highTTTEEE)
    best_fit.pop("theta_MC_100")
    info_likelihood = lik_info_lowTEB_highTTTEEE
    info_theory = {"classy": {"extra_args": cmb_precision["classy"]}}
    best_fit_derived = deepcopy(derived_lowTEB_highTTTEEE)
    for p in classy_unknown:
        best_fit_derived.pop(p, None)
    chi2_lowTEB_highTTTEEE_classy = deepcopy(chi2_lowTEB_highTTTEEE)
    chi2_lowTEB_highTTTEEE_classy["tolerance"] += classy_extra_tolerance
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_lowTEB_highTTTEEE_classy, best_fit_derived,
                 skip_not_installed=skip_not_installed)


def test_planck_2015_l_classy(packages_path, skip_not_installed):
    best_fit = deepcopy(params_lensing)
    best_fit.pop("theta_MC_100")
    info_likelihood = lik_info_lensing
    info_theory = {"classy": {"extra_args": cmb_precision["classy"]}}
    best_fit_derived = deepcopy(derived_lensing)
    for p in classy_unknown:
        best_fit_derived.pop(p, None)
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_lensing, best_fit_derived,
                 skip_not_installed=skip_not_installed)


# Temperature only #######################################################################

lik_info_lowl_highTT = {"planck_2015_lowl": None, "planck_2015_plikHM_TT": None}

chi2_lowl_highTT = {"planck_2015_lowl": 15.39,
                    "planck_2015_plikHM_TT": 761.09,
                    "tolerance": 0.1}

params_lowl_highTT = {
    # Sampled
    "omegabh2": 0.02249139,
    "omegach2": 0.1174684,
    # only one of the next two is finally used!
    "H0": 68.43994,  # will be ignored in the CAMB case
    "theta_MC_100": 1.041189,  # will be ignored in the CLASS case
    "tau": 0.1249913,
    "logA": 3.179,
    "ns": 0.9741693,
    # Planck likelihood
    "A_planck": 1.00027,
    "A_cib_217": 61.1,
    "xi_sz_cib": 0.56,
    "A_sz": 6.84,
    "ps_A_100_100": 242.9,
    "ps_A_143_143": 43.0,
    "ps_A_143_217": 46.1,
    "ps_A_217_217": 104.1,
    "ksz_norm": 0.00,
    "gal545_A_100": 7.31,
    "gal545_A_143": 9.07,
    "gal545_A_143_217": 17.99,
    "gal545_A_217": 82.9,
    "calib_100T": 0.99796,
    "calib_217T": 0.99555}

derived_lowl_highTT = {
    # param: [best_fit, sigma]
    "H0": [params_lowl_highTT["H0"], 1.2],
    "omegal": [0.6998, 0.016],
    "omegam": [0.3002, 0.016],
    "sigma8": [0.8610, 0.023],
    "zrei": [13.76, 2.5],
    # "YHe": [0.2454462, 0.0001219630],
    # "Y_p": [0.2467729, 0.0001224069],
    # "DH": [2.568606e-5, 0.05098625e-5],
    "age": [13.7664, 0.048],
    "zstar": [1089.55, 0.52],
    "rstar": [145.00, 0.55],
    "thetastar": [1.041358, 0.0005117986],
    "DAstar": [13.924, 0.050],
    "zdrag": [1060.05, 0.52],
    "rdrag": [147.63, 0.53],
    "kd": [0.14039, 0.00053],
    "thetad": [0.160715, 0.00029],
    "zeq": [3345, 58],
    "keq": [0.010208, 0.00018],
    "thetaeq": [0.8243, 0.011],
    "thetarseq": [0.4550, 0.0058]}

# Best fit Polarization ##################################################################

lik_info_lowTEB_highTTTEEE = {"planck_2015_lowTEB": None,
                              "planck_2015_plikHM_TTTEEE": None}

chi2_lowTEB_highTTTEEE = {"planck_2015_lowTEB": 10496.93,
                          "planck_2015_plikHM_TTTEEE": 2431.65,
                          "tolerance": 0.15}

params_lowTEB_highTTTEEE = {
    # Sampled
    "omegabh2": 0.02225203,
    "omegach2": 0.1198657,
    # only one of the next two is finally used!
    "H0": 67.25,  # will be ignored in the CAMB case
    "theta_MC_100": 1.040778,  # will be ignored in the CLASS case
    "logA": 3.0929,
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
    "H0": [params_lowTEB_highTTTEEE["H0"], 0.66],
    "omegal": [0.6844, 0.0091],
    "omegam": [0.3156, 0.0091],
    "sigma8": [0.8310, 0.013],
    "zrei": [10.07, 1.6],
    # "YHe": [0.2453409, 0.000072],
    # "Y_p": [0.2466672, 0.000072],
    # "DH": [2.6136e-5, 0.030e-5],
    "age": [13.8133, 0.026],
    "zstar": [1090.057, 0.30],
    "rstar": [144.556, 0.32],
    "thetastar": [1.040967, 0.00032],
    "DAstar": [13.8867, 0.030],
    "zdrag": [1059.666, 0.31],
    "rdrag": [147.257, 0.31],
    "kd": [0.140600, 0.00032],
    "thetad": [0.160904, 0.00018],
    "zeq": [3396.2, 33],
    "keq": [0.010365, 0.00010],
    "thetaeq": [0.8139, 0.0063],
    "thetarseq": [0.44980, 0.0032]}

# Best fit lensing (from best combination with lowTEB+_highTTTEEE ########################

lik_info_lensing = {"planck_2015_lensing": None}

chi2_lensing = {"planck_2015_lensing": 9.78, "tolerance": 0.09}

params_lensing = {
    # Sampled
    "omegabh2": 0.022274,
    "omegach2": 0.11913,
    # only one of the next two is finally used!
    "H0": 67.56,  # will be ignored in the CAMB case
    "theta_MC_100": 1.040867,  # will be ignored in the CLASS case
    "logA": 3.0600,
    "ns": 0.96597,
    "tau": 0.0639,
    # Planck likelihood
    "A_planck": 0.99995}

derived_lensing = {
    # param: [best_fit, sigma]
    "H0": [params_lensing["H0"], 0.64],
    "omegal": [0.6888, 0.0087],
    "omegam": [0.3112, 0.0087],
    "sigma8": [0.8153, 0.0087],
    "zrei": [8.64, 1.3],
    # "YHe": [0.245350, 0.000071],
    # "Y_p": [0.246677, 0.000072],
    # "DH": [2.6095e-5, 0.030e-5],
    "age": [13.8051, 0.026],
    "zstar": [1089.966, 0.29],
    "rstar": [144.730, 0.31],
    "thetastar": [1.041062, 0.00031],
    "DAstar": [13.9022, 0.029],
    "zdrag": [1059.666, 0.31],
    "rdrag": [147.428, 0.30],
    "kd": [0.140437, 0.00032],
    "thetad": [0.160911, 0.00018],
    "zeq": [3379.1, 32],
    "keq": [0.010313, 0.000096],
    "thetaeq": [0.8171, 0.0060],
    "thetarseq": [0.45146, 0.0031]}
