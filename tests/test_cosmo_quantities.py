"""
Testing some quantities not used yet by any internal likelihood.
"""

import pytest
import numpy as np
from copy import deepcopy

from cobaya.cosmo_input import planck_lss_precision, planck_base_model, create_input
from cobaya.model import get_model
from cobaya.tools import recursive_update, check_2d

from .test_cosmo_planck_2015 import params_lowTEB_highTTTEEE
from .conftest import install_test_wrapper
from .common import process_packages_path

fiducial_parameters = deepcopy(params_lowTEB_highTTTEEE)
redshifts = [100, 10, 1, 0]


def _get_model_with_requirements_and_eval(theo, reqs, packages_path, skip_not_installed):
    planck_base_model_prime = deepcopy(planck_base_model)
    planck_base_model_prime["hubble"] = "H"  # intercompatibility CAMB/CLASS
    info_theory = {theo: {"extra_args": planck_lss_precision[theo]}}
    info = create_input(planck_names=True, theory=theo, **planck_base_model_prime)
    info = recursive_update(info, {"theory": info_theory, "likelihood": {"one": None}})
    info["packages_path"] = process_packages_path(packages_path)
    info["debug"] = True
    model = install_test_wrapper(skip_not_installed, get_model, info)
    eval_parameters = {p: v for p, v in fiducial_parameters.items()
                       if p in model.parameterization.sampled_params()}
    model.add_requirements(reqs)
    model.logposterior(eval_parameters)
    return model


# sigma8(z), fsgima8(z) ##################################################################

sigma8_values = [0.0107213, 0.09646127, 0.5045227, 0.83073719]
fsigma8_values = [0.01063159, 0.09639173, 0.44311857, 0.43909749]


def _test_cosmo_sigma8_fsigma8(theo, packages_path, skip_not_installed):
    reqs = {"sigma8_z": {"z": redshifts}, "fsigma8": {"z": redshifts}}
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed)
    assert np.allclose(model.theory[theo].get_sigma8_z(redshifts),
                       sigma8_values, rtol=1e-5 if theo.lower() == "camb" else 5e-4)
    # NB: classy tolerance quite high for fsigma8!
    # (see also test of bao.sdss_dr16_baoplus_qso)
    assert np.allclose(model.theory[theo].get_fsigma8(redshifts),
                       fsigma8_values, rtol=1e-5 if theo.lower() == "camb" else 1e-2)


def test_cosmo_sigma8_fsigma8_camb(packages_path, skip_not_installed):
    _test_cosmo_sigma8_fsigma8("camb", packages_path, skip_not_installed)


def test_cosmo_sigma8_fsigma8_classy(packages_path, skip_not_installed):
    _test_cosmo_sigma8_fsigma8("classy", packages_path, skip_not_installed)


# sigma(R, z) ############################################################################

z_sigma_R = [0, 2, 5]
R_sigma_R = np.arange(1, 20, 1)
sigma_R_values = {
    ("delta_tot", "delta_tot"): [
        [2.91641221, 2.21110361, 1.83787574, 1.59298059, 1.41544224,
         1.27835804, 1.16812519, 1.07707886, 1.00019134, 0.93411662,
         0.87659859, 0.82600355, 0.78103409, 0.74077065, 0.70445364,
         0.67146121, 0.64136035, 0.61377842, 0.58839752],
        [1.21913872, 0.92428173, 0.76825062, 0.66586932, 0.59164694,
         0.53433678, 0.48825215, 0.45018873, 0.41804465, 0.39042106,
         0.36637479, 0.34522284, 0.3264228, 0.30959024, 0.2944076,
         0.28061489, 0.26803108, 0.2565004, 0.24588992],
        [0.61857867, 0.46896734, 0.38979628, 0.33784721, 0.30018608,
         0.27110628, 0.24772239, 0.22840853, 0.21209819, 0.19808161,
         0.18588019, 0.17514739, 0.16560799, 0.15706691, 0.14936304,
         0.14236443, 0.13597925, 0.13012844, 0.12474456]],
    ("delta_nonu", "delta_nonu"): [
        [2.92953248, 2.22099356, 1.84605073, 1.60002765, 1.42167022,
         1.2839528, 1.17321033, 1.08174289, 1.00449967, 0.93811928,
         0.88033523, 0.82950631, 0.7843291, 0.74387981, 0.70739533,
         0.67425091, 0.64401153, 0.61630285, 0.59080548],
        [1.22465058, 0.92844891, 0.77170492, 0.66885515, 0.59429274,
         0.53671975, 0.49042359, 0.45218534, 0.41989354, 0.39214295,
         0.36798608, 0.34673683, 0.3278503, 0.3109403, 0.29568781,
         0.28183165, 0.26918997, 0.25760626, 0.24694701],
        [0.62138387, 0.47109062, 0.39155828, 0.33937194, 0.30153863,
         0.27232581, 0.24883487, 0.22943255, 0.21304747, 0.19896663,
         0.18670925, 0.17592722, 0.16634405, 0.15776378, 0.15002454,
         0.14299381, 0.13657932, 0.13070165, 0.12529306]]}


def _test_cosmo_sigma_R(theo, packages_path, skip_not_installed):
    vars_pairs = (("delta_tot", "delta_tot"), ("delta_nonu", "delta_nonu"))
    reqs = {"sigma_R": {"z": z_sigma_R, "R": R_sigma_R, "vars_pairs": vars_pairs}}
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed)
    for pair in vars_pairs:
        z_out, R_out, sigma_R_out = model.theory[theo].get_sigma_R(pair)
        assert np.allclose(R_out, R_sigma_R)
        assert np.allclose(z_out, z_sigma_R)
        assert np.allclose(sigma_R_out, np.array(sigma_R_values[pair]),
                           rtol=1e-5 if theo.lower() == "camb" else 2e-3)


def test_cosmo_sigma_R_camb(packages_path, skip_not_installed):
    _test_cosmo_sigma_R("camb", packages_path, skip_not_installed)


def test_cosmo_sigma_R_classy(packages_path, skip_not_installed):
    _test_cosmo_sigma_R("classy", packages_path, skip_not_installed)


# Omega_X(z) #############################################################################

Omega_b_values = [0.15172485, 0.15517809, 0.12258897, 0.04920226]
Omega_cdm_values = [0.81730093, 0.83590262, 0.66035382, 0.26503934]
Omega_nu_massive_values = [0.00608623, 0.0045243, 0.00355319, 0.00142589]


def _test_cosmo_omega(theo, packages_path, skip_not_installed):
    reqs = {"Omega_b": {"z": redshifts}, "Omega_cdm": {"z": redshifts},
            "Omega_nu_massive": {"z": redshifts}}
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed)
    assert np.allclose(model.theory[theo].get_Omega_b(redshifts),
                       Omega_b_values, rtol=1e-5 if theo.lower() == "camb" else 5e-4)
    assert np.allclose(model.theory[theo].get_Omega_cdm(redshifts),
                       Omega_cdm_values, rtol=1e-5 if theo.lower() == "camb" else 5e-4)
    assert np.allclose(model.theory[theo].get_Omega_nu_massive(redshifts),
                       Omega_nu_massive_values,
                       rtol=1e-5 if theo.lower() == "camb" else 2e-3)


def test_cosmo_omega_camb(packages_path, skip_not_installed):
    _test_cosmo_omega("camb", packages_path, skip_not_installed)


def test_cosmo_omega_classy(packages_path, skip_not_installed):
    _test_cosmo_omega("classy", packages_path, skip_not_installed)


# angular_diameter_distance_2 ############################################################

ang_diam_dist_2_values = [
    31.59567987, 93.34513188, 127.08027199, 566.97224099, 876.72216398, 1703.62457558]


def _test_cosmo_ang_diam_dist_2(theo, packages_path, skip_not_installed):
    reqs = {"angular_diameter_distance_2": {"z_pairs": redshifts}}
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed)
    redshift_pairs = check_2d(redshifts)
    assert np.allclose(model.theory[theo].get_angular_diameter_distance_2(redshift_pairs),
                       ang_diam_dist_2_values, rtol=1e-5)


def test_cosmo_ang_diam_dist_2_camb(packages_path, skip_not_installed):
    _test_cosmo_ang_diam_dist_2("camb", packages_path, skip_not_installed)


def test_cosmo_ang_diam_dist_2_classy(packages_path, skip_not_installed):
    _test_cosmo_ang_diam_dist_2("classy", packages_path, skip_not_installed)


# Weyl power spectrum ####################################################################

var_pair = ("Weyl", "Weyl")
zs = [0, 0.5, 1, 1.5, 2, 3.5]
ks = np.logspace(-4, np.log10(15), 5)
Pkz_values = [[-27.43930515, -24.66945975, -24.57476343, -28.00109372, -33.24474829],
              [-27.15442496, -24.3843334, -24.28538885, -28.05600287, -33.27033893],
              [-27.05300137, -24.28273214, -24.1803611, -28.25169186, -33.31505885],
              [-27.01121008, -24.24079765, -24.13585631, -28.46595848, -33.3742216],
              [-26.99148443, -24.22094486, -24.11405468, -28.65842103, -33.44075841],
              [-26.97141436, -24.20052295, -24.08988483, -29.05073038, -33.70090064]]


def _test_cosmo_weyl_pkz(theo, packages_path, skip_not_installed):
    # Similar to what"s requested by DES (but not used by default)
    reqs = {"Pk_interpolator": {"z": zs, "k_max": max(ks), "nonlinear": (False, True),
                                "vars_pairs": var_pair}}
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed)
    interp = model.theory[theo].get_Pk_interpolator(var_pair=var_pair, nonlinear=True)
    assert np.allclose(np.array(Pkz_values), interp.logP(zs, ks),
                       rtol=1e-5 if theo.lower() == "camb" else 5e-4)


def test_cosmo_weyl_pkz_camb(packages_path, skip_not_installed):
    _test_cosmo_weyl_pkz("camb", packages_path, skip_not_installed)


@pytest.mark.skip
def test_cosmo_weyl_pkz_classy(packages_path, skip_not_installed):
    _test_cosmo_weyl_pkz("classy", packages_path, skip_not_installed)
