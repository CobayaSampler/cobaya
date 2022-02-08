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
    model = install_test_wrapper(skip_not_installed, get_model, info)
    eval_parameters = {p: v for p, v in fiducial_parameters.items()
                       if p in model.parameterization.sampled_params()}
    model.add_requirements(reqs)
    model.logposterior(eval_parameters)
    return model


# sigma8(z) ##############################################################################

sigma8_values = [0.01072007, 0.0964498, 0.50446177, 0.83063667]


def _test_cosmo_sigma8(theo, packages_path, skip_not_installed):
    reqs = {"sigma8_z": {"z": redshifts}}
    model = _get_model_with_requirements_and_eval(
        theo, reqs, packages_path, skip_not_installed)
    assert np.allclose(model.theory[theo].get_sigma8_z(redshifts),
                       sigma8_values, rtol=1e-5 if theo.lower() == "camb" else 5e-4)


def test_cosmo_sigma8_camb(packages_path, skip_not_installed):
    _test_cosmo_sigma8("camb", packages_path, skip_not_installed)


def test_cosmo_sigma8_classy(packages_path, skip_not_installed):
    _test_cosmo_sigma8("classy", packages_path, skip_not_installed)


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
