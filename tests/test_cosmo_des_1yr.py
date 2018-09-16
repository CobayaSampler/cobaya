from copy import deepcopy
import pytest

from test_cosmo_planck_2015 import params_lowTEB_highTTTEEE
from common_cosmo import body_of_test

best_fit = deepcopy(params_lowTEB_highTTTEEE)


@pytest.mark.skip
def test_cosmo_des_1yr_shear_camb(modules):
    like = "des_1yr_shear"
    info_likelihood = {like: {}}
    best_fit_shear = deepcopy(best_fit)
    best_fit_shear.update(test_params_shear)
    info_theory = {"camb": None}
    body_of_test(modules, best_fit_shear, info_likelihood, info_theory,
                 {like: 0, "tolerance": 5})


@pytest.mark.skip
def test_cosmo_des_1yr_clustering_camb(modules):
    like = "des_1yr_clustering"
    info_likelihood = {like: {}}
    best_fit_clustering = deepcopy(best_fit)
    best_fit_clustering.update(test_params_clustering)
    info_theory = {"camb": None}
    body_of_test(modules, best_fit_clustering, info_likelihood, info_theory,
                 {like: 0, "tolerance": 5})


@pytest.mark.skip
def test_cosmo_des_1yr_galaxy_galaxylensing_camb(modules):
    like = "des_1yr_galaxy_galaxylensing"
    info_likelihood = {like: {}}
    best_fit_galaxy_galaxylensing = deepcopy(best_fit)
    best_fit_galaxy_galaxylensing.update(test_params_shear)
    best_fit_galaxy_galaxylensing.update(test_params_clustering)
    info_theory = {"camb": None}
    body_of_test(modules, best_fit_galaxy_galaxylensing, info_likelihood, info_theory,
                 {like: 0, "tolerance": 5})


@pytest.mark.skip
def test_cosmo_des_1yr_shear_classy(modules):
    like = "des_1yr_shear"
    info_likelihood = {like: {}}
    best_fit_shear = deepcopy(best_fit)
    best_fit_shear.update(test_params_shear)
    info_theory = {"classy": None}
    body_of_test(modules, best_fit_shear, info_likelihood, info_theory,
                 {like: 0, "tolerance": 5})


@pytest.mark.skip
def test_cosmo_des_1yr_clustering_classy(modules):
    like = "des_1yr_clustering"
    info_likelihood = {like: {}}
    best_fit_clustering = deepcopy(best_fit)
    best_fit_clustering.update(test_params_clustering)
    info_theory = {"classy": None}
    body_of_test(modules, best_fit_clustering, info_likelihood, info_theory,
                 {like: 0, "tolerance": 5})


@pytest.mark.skip
def test_cosmo_des_1yr_galaxy_galaxylensing_classy(modules):
    like = "des_1yr_galaxy_galaxylensing"
    info_likelihood = {like: {}}
    best_fit_galaxy_galaxylensing = deepcopy(best_fit)
    best_fit_galaxy_galaxylensing.update(test_params_shear)
    best_fit_galaxy_galaxylensing.update(test_params_clustering)
    info_theory = {"classy": None}
    body_of_test(modules, best_fit_galaxy_galaxylensing, info_likelihood, info_theory,
                 {like: 0, "tolerance": 5})


test_params_shear = {
    # wl_photoz_errors
    "DES_DzS1": 0.002,
    "DES_DzS2": -0.015,
    "DES_DzS3": 0.007,
    "DES_DzS4": -0.018,
    # shear_calibration_parameters
    "DES_m1": 0.012,
    "DES_m2": 0.012,
    "DES_m3": 0.012,
    "DES_m4": 0.012,
    # Intrinsic Alignment
    "DES_AIA": 1.0,
    "DES_alphaIA": 1.0,
    "DES_z0IA": 0.62}

test_params_clustering = {
    # lens_photoz_errors
    "DES_DzL1": 0.002,
    "DES_DzL2": 0.001,
    "DES_DzL3": 0.003,
    "DES_DzL4": 0,
    "DES_DzL5": 0,
    # bin_bias
    "DES_b1": 1.45,
    "DES_b2": 1.55,
    "DES_b3": 1.65,
    "DES_b4": 1.8,
    "DES_b5": 2.0}
