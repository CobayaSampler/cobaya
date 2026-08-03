"""
Testing some quantities not used yet by any internal likelihood.
"""

from copy import deepcopy

import numpy as np

from cobaya.cosmo_input import cmb_lss_precision, create_input, planck_base_model
from cobaya.model import get_model
from cobaya.tools import check_2d, recursive_update

from .common import process_packages_path
from .conftest import install_test_wrapper
from .test_cosmo_planck_2018 import planck_2018_precision

lss_tests_precision = deepcopy(cmb_lss_precision)
lss_tests_precision["camb"].update(planck_2018_precision["camb"])
lss_tests_precision["classy"].update(planck_2018_precision["classy"])

# Best fit Planck 2015 as test point
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
    "calib_217T": 0.99598,
}

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
    "thetarseq": [0.44980, 0.0032],
}

fiducial_parameters = deepcopy(params_lowTEB_highTTTEEE)
redshifts = [100, 10, 1, 0]


def _get_model_with_requirements_and_eval(theo, reqs, packages_path, skip_not_installed):
    planck_base_model_prime = deepcopy(planck_base_model)
    planck_base_model_prime["hubble"] = "H"  # intercompatibility CAMB/CLASS
    info_theory = {theo: {"extra_args": lss_tests_precision[theo]}}
    info = create_input(planck_names=True, theory=theo, **planck_base_model_prime)
    info = recursive_update(info, {"theory": info_theory, "likelihood": {"one": None}})
    info["packages_path"] = process_packages_path(packages_path)
    info["debug"] = True
    model = install_test_wrapper(skip_not_installed, get_model, info)
    eval_parameters = {
        p: v
        for p, v in fiducial_parameters.items()
        if p in model.parameterization.sampled_params()
    }
    model.add_requirements(reqs)
    model.logposterior(eval_parameters)
    return model


# sigma8(z), fsgima8(z) ##################################################################

sigma8_values = [0.01072142, 0.09646278, 0.50453064, 0.83075029]
fsigma8_values = [0.01063181, 0.09639325, 0.44312554, 0.43910440]


def _test_cosmo_sigma8_fsigma8(theo, packages_path, skip_not_installed):
    reqs = {"sigma8_z": {"z": redshifts}, "fsigma8": {"z": redshifts}}
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed
    )
    assert np.allclose(
        model.theory[theo].get_sigma8_z(redshifts),
        sigma8_values,
        rtol=1e-5 if theo.lower() == "camb" else 5e-4,
    )
    # NB: classy tolerance quite high for fsigma8!
    # (see also test of bao.sdss_dr16_baoplus_qso)
    assert np.allclose(
        model.theory[theo].get_fsigma8(redshifts),
        fsigma8_values,
        rtol=1e-5 if theo.lower() == "camb" else 1e-2,
    )


def test_cosmo_sigma8_fsigma8_camb(packages_path, skip_not_installed):
    _test_cosmo_sigma8_fsigma8("camb", packages_path, skip_not_installed)


def test_cosmo_sigma8_fsigma8_classy(packages_path, skip_not_installed):
    _test_cosmo_sigma8_fsigma8("classy", packages_path, skip_not_installed)


# sigma(R, z) ############################################################################

z_sigma_R = [0, 2, 5]
R_sigma_R = np.arange(1, 20, 1)
sigma_R_values = {
    ("delta_tot", "delta_tot"): [
        [
            2.91641254,
            2.21110647,
            1.83788033,
            1.59298658,
            1.41544943,
            1.27836629,
            1.16813442,
            1.07708900,
            1.00020233,
            0.93412840,
            0.87661110,
            0.82601676,
            0.78104794,
            0.74078511,
            0.70446867,
            0.67147677,
            0.64137639,
            0.61379492,
            0.58841445,
        ],
        [
            1.21913884,
            0.92428288,
            0.76825251,
            0.66587180,
            0.59164992,
            0.53434021,
            0.48825600,
            0.45019296,
            0.41804923,
            0.39042598,
            0.36638001,
            0.34522835,
            0.32642858,
            0.30959628,
            0.29441387,
            0.28062138,
            0.26803778,
            0.25650726,
            0.24589699,
        ],
        [
            0.61857867,
            0.46896789,
            0.38979721,
            0.33784844,
            0.30018757,
            0.27110800,
            0.24772433,
            0.22841066,
            0.21210050,
            0.19808409,
            0.18588283,
            0.17515017,
            0.16561091,
            0.15706996,
            0.14936621,
            0.14236772,
            0.13598264,
            0.13013193,
            0.12474814,
        ],
    ],
    ("delta_nonu", "delta_nonu"): [
        [
            2.92953280,
            2.22099638,
            1.84605528,
            1.60003361,
            1.42167739,
            1.28396104,
            1.17321957,
            1.08175304,
            1.00451067,
            0.93813107,
            0.88034777,
            0.82951954,
            0.78434299,
            0.74389431,
            0.70741039,
            0.67426650,
            0.64402762,
            0.61631940,
            0.59082246,
        ],
        [
            1.22465067,
            0.92845005,
            0.77170679,
            0.66885762,
            0.59429572,
            0.53672319,
            0.49042745,
            0.45218958,
            0.41989813,
            0.39214788,
            0.36799132,
            0.34674236,
            0.32785610,
            0.31094635,
            0.29569410,
            0.28183817,
            0.26919669,
            0.25761318,
            0.24695410,
        ],
        [
            0.62138393,
            0.47109121,
            0.39155925,
            0.33937320,
            0.30154015,
            0.27232755,
            0.24883683,
            0.22943470,
            0.21304980,
            0.19896912,
            0.18671191,
            0.17593002,
            0.16634699,
            0.15776685,
            0.15002773,
            0.14299711,
            0.13658272,
            0.13070515,
            0.12529665,
        ],
    ],
}


def _test_cosmo_sigma_R(theo, packages_path, skip_not_installed):
    vars_pairs = (("delta_tot", "delta_tot"), ("delta_nonu", "delta_nonu"))
    reqs = {"sigma_R": {"z": z_sigma_R, "R": R_sigma_R, "vars_pairs": vars_pairs}}
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed
    )
    for pair in vars_pairs:
        z_out, R_out, sigma_R_out = model.theory[theo].get_sigma_R(pair)
        assert np.allclose(R_out, R_sigma_R)
        assert np.allclose(z_out, z_sigma_R)
        assert np.allclose(
            sigma_R_out,
            np.array(sigma_R_values[pair]),
            rtol=1e-5 if theo.lower() == "camb" else 2e-3,
        )


def test_cosmo_sigma_R_camb(packages_path, skip_not_installed):
    _test_cosmo_sigma_R("camb", packages_path, skip_not_installed)


def test_cosmo_sigma_R_classy(packages_path, skip_not_installed):
    _test_cosmo_sigma_R("classy", packages_path, skip_not_installed)


# Omega_X(z) #############################################################################

Omega_b_values = [0.15172485, 0.15517809, 0.12258897, 0.04920226]
Omega_cdm_values = [0.81730093, 0.83590262, 0.66035382, 0.26503934]
Omega_nu_massive_values = [0.0060864, 0.00452457, 0.00355337, 0.00142596]


def _test_cosmo_omega(theo, packages_path, skip_not_installed):
    reqs = {
        "Omega_b": {"z": redshifts},
        "Omega_cdm": {"z": redshifts},
        "Omega_nu_massive": {"z": redshifts},
    }
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed
    )
    assert np.allclose(
        model.theory[theo].get_Omega_b(redshifts),
        Omega_b_values,
        rtol=1e-5 if theo.lower() == "camb" else 5e-4,
    )
    assert np.allclose(
        model.theory[theo].get_Omega_cdm(redshifts),
        Omega_cdm_values,
        rtol=1e-5 if theo.lower() == "camb" else 5e-4,
    )
    assert np.allclose(
        model.theory[theo].get_Omega_nu_massive(redshifts),
        Omega_nu_massive_values,
        rtol=1e-4 if theo.lower() == "camb" else 2e-3,
    )


def test_cosmo_omega_camb(packages_path, skip_not_installed):
    _test_cosmo_omega("camb", packages_path, skip_not_installed)


def test_cosmo_omega_classy(packages_path, skip_not_installed):
    _test_cosmo_omega("classy", packages_path, skip_not_installed)


# angular_diameter_distance_2 ############################################################

ang_diam_dist_2_values = [
    31.59567987,
    93.34513188,
    127.08027199,
    566.97224099,
    876.72216398,
    1703.62457558,
]


def _test_cosmo_ang_diam_dist_2(theo, packages_path, skip_not_installed):
    reqs = {"angular_diameter_distance_2": {"z_pairs": redshifts}}
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed
    )
    redshift_pairs = check_2d(redshifts)
    assert np.allclose(
        model.theory[theo].get_angular_diameter_distance_2(redshift_pairs),
        ang_diam_dist_2_values,
        rtol=1e-5,
    )


def test_cosmo_ang_diam_dist_2_camb(packages_path, skip_not_installed):
    _test_cosmo_ang_diam_dist_2("camb", packages_path, skip_not_installed)


def test_cosmo_ang_diam_dist_2_classy(packages_path, skip_not_installed):
    _test_cosmo_ang_diam_dist_2("classy", packages_path, skip_not_installed)


# Weyl power spectrum ####################################################################

var_pair = ("Weyl", "Weyl")
zs = [0, 0.5, 1, 1.5, 2, 3.5]
ks = np.logspace(-4, np.log10(15), 5)
Pkz_values = [
    [-27.43930515, -24.66945900, -24.57379175, -28.00104480, -33.24475090],
    [-27.15442496, -24.38433192, -24.28441132, -28.05596695, -33.27034018],
    [-27.05300137, -24.28273060, -24.17938062, -28.25166740, -33.31505946],
    [-27.01121008, -24.24079436, -24.13487392, -28.46594161, -33.37422198],
    [-26.99148442, -24.22094148, -24.11307122, -28.65841066, -33.44075857],
    [-26.97141453, -24.20052285, -24.08890040, -29.05072967, -33.70090126],
]


def _test_cosmo_weyl_pkz(theo, packages_path, skip_not_installed):
    # Similar to what"s requested by DES (but not used by default)
    reqs = {
        "Pk_interpolator": {
            "z": zs,
            "k_max": max(ks),
            "nonlinear": (False, True),
            "vars_pairs": var_pair,
        }
    }
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed
    )
    interp = model.theory[theo].get_Pk_interpolator(var_pair=var_pair, nonlinear=True)
    assert np.allclose(
        np.array(Pkz_values),
        interp.logP(zs, ks),
        rtol=1e-5 if theo.lower() == "camb" else 5e-4,
    )


def test_cosmo_weyl_pkz_camb(packages_path, skip_not_installed):
    _test_cosmo_weyl_pkz("camb", packages_path, skip_not_installed)


def test_cosmo_weyl_pkz_classy(packages_path, skip_not_installed):
    _test_cosmo_weyl_pkz("classy", packages_path, skip_not_installed)
