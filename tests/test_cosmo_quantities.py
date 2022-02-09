"""
Testing some quantities not used yet by any internal likelihood.
"""

import pytest
import numpy as np
from copy import deepcopy

from cobaya.cosmo_input import base_precision, planck_base_model, create_input
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
    info_theory = {theo: {"extra_args": base_precision[theo]}}
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

sigma8_values = [0.01072007, 0.0964498, 0.50446177, 0.83063667]
fsigma8_values = [0.01063036, 0.09638032, 0.44306551, 0.43904513]


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


# Omega_X(z) #############################################################################

Omega_b_values = [0.15172485, 0.15517809, 0.12258897, 0.04920226]
Omega_cdm_values = [0.81730093, 0.83590262, 0.66035382, 0.26503934]
Omega_nu_massive_values = [0.00608926, 0.00452621, 0.00355468, 0.00142649]


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
Pkz_values = [[-27.43930416, -24.66947574, -24.57497701, -28.00145562, -33.24503304],
              [-27.15442457, -24.38434986, -24.28560236, -28.05638781, -33.27063402],
              [-27.05300137, -24.28274865, -24.18057448, -28.25209712, -33.31536626],
              [-27.01121008, -24.24081427, -24.13606913, -28.46637302, -33.37454288],
              [-26.99148444, -24.22096153, -24.11426728, -28.65883515, -33.44109139],
              [-26.97141437, -24.20053966, -24.09009681, -29.05111258, -33.70130814]]


def _test_cosmo_weyl_pkz(theo, packages_path, skip_not_installed):
    # Similar to what"s requested by DES (but not used by default)
    reqs = {"Pk_interpolator": {"z": zs, "k_max": max(ks), "nonlinear": (False, True),
                                "vars_pairs": var_pair}}
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed)
    interp = model.theory[theo].get_Pk_interpolator(var_pair=var_pair, nonlinear=True)
    assert np.allclose(Pkz_values, interp.logP(zs, ks),
                       rtol=1e-5 if theo.lower() == "camb" else 5e-4)


def test_cosmo_weyl_pkz_camb(packages_path, skip_not_installed):
    _test_cosmo_weyl_pkz("camb", packages_path, skip_not_installed)


def test_cosmo_weyl_pkz_classy(packages_path, skip_not_installed):
    _test_cosmo_weyl_pkz("classy", packages_path, skip_not_installed)
