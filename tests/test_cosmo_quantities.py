"""
Testing some quantities not used yet by any internal likelihood.
"""

import numpy as np
from copy import deepcopy

from cobaya.cosmo_input import base_precision, planck_base_model, create_input
from cobaya.model import get_model
from cobaya.tools import recursive_update

from .test_cosmo_planck_2015 import params_lowTEB_highTTTEEE
from .conftest import install_test_wrapper
from .common import process_packages_path

fiducial_parameters = deepcopy(params_lowTEB_highTTTEEE)
sigma8_redshifts = [100, 10, 1, 0]
sigma8_values = [0.01072007, 0.0964498, 0.50446177, 0.83063667]


def _test_cosmo_sigma8(theo, packages_path, skip_not_installed):
    planck_base_model_prime = deepcopy(planck_base_model)
    planck_base_model_prime["hubble"] = "H"
    info_theory = {theo: {"extra_args": base_precision[theo]}}
    info = create_input(planck_names=True, theory=theo, **planck_base_model_prime)
    info = recursive_update(info, {"theory": info_theory, "likelihood": {"one": None}})
    info["packages_path"] = process_packages_path(packages_path)
    model = install_test_wrapper(skip_not_installed, get_model, info)
    eval_parameters = {p: v for p, v in fiducial_parameters.items()
                       if p in model.parameterization.sampled_params()}
    model.add_requirements({"sigma8_z": {"z": sigma8_redshifts}})
    model.logposterior(eval_parameters)
    assert np.allclose(model.theory[theo].get_sigma8_z(sigma8_redshifts),
                       sigma8_values, rtol=1e-5 if theo.lower() == "camb" else 5e-4)


def test_cosmo_sigma8_camb(packages_path, skip_not_installed):
    _test_cosmo_sigma8("camb", packages_path, skip_not_installed)


def test_cosmo_sigma8_classy(packages_path, skip_not_installed):
    _test_cosmo_sigma8("classy", packages_path, skip_not_installed)
