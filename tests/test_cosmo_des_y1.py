from copy import deepcopy
from types import MappingProxyType
from .test_cosmo_planck_2015 import params_lowTEB_highTTTEEE
from .common_cosmo import body_of_test
from cobaya.cosmo_input import base_precision

best_fit = deepcopy(params_lowTEB_highTTTEEE)

info_camb = MappingProxyType({"camb": {"extra_args": base_precision["camb"]}})
info_classy = MappingProxyType({"classy": {"extra_args": base_precision["classy"]}})


def test_cosmo_des_y1_shear_camb(packages_path, skip_not_installed,
                                 info_theory=info_camb):
    like = "des_y1.shear"
    info_likelihood = {like: {}}
    best_fit_shear = deepcopy(best_fit)
    best_fit_shear.update(test_params_shear)
    body_of_test(packages_path, best_fit_shear, info_likelihood, info_theory,
                 {like: ref_chi2["shear"], "tolerance": tolerance},
                 skip_not_installed=skip_not_installed)


def test_cosmo_des_y1_clustering_camb(packages_path, skip_not_installed,
                                      info_theory=info_camb):
    like = "des_y1.clustering"
    info_likelihood = {like: {}}
    best_fit_clustering = deepcopy(best_fit)
    best_fit_clustering.update(test_params_clustering)
    body_of_test(packages_path, best_fit_clustering, info_likelihood, info_theory,
                 {like: ref_chi2["clustering"], "tolerance": tolerance},
                 skip_not_installed=skip_not_installed)


def test_cosmo_des_y1_galaxy_galaxy_camb(packages_path, skip_not_installed,
                                         info_theory=info_camb):
    like = "des_y1.galaxy_galaxy"
    info_likelihood = {like: {}}
    best_fit_galaxy_galaxy = deepcopy(best_fit)
    best_fit_galaxy_galaxy.update(test_params_shear)
    best_fit_galaxy_galaxy.update(test_params_clustering)
    body_of_test(packages_path, best_fit_galaxy_galaxy, info_likelihood, info_theory,
                 {like: ref_chi2["galaxy_galaxy"], "tolerance": tolerance},
                 skip_not_installed=skip_not_installed)


def test_cosmo_des_y1_joint_camb(packages_path, skip_not_installed,
                                 info_theory=info_camb):
    like = "des_y1.joint"
    info_likelihood = {like: {}}
    best_fit_joint = deepcopy(best_fit)
    best_fit_joint.update(test_params_shear)
    best_fit_joint.update(test_params_clustering)
    body_of_test(packages_path, best_fit_joint, info_likelihood, info_theory,
                 {like: ref_chi2["joint"], "tolerance": tolerance},
                 skip_not_installed=skip_not_installed)


def test_cosmo_des_y1_shear_classy(packages_path, skip_not_installed,
                                   info_theory=info_classy):
    test_cosmo_des_y1_shear_camb(packages_path, info_theory=info_theory,
                                 skip_not_installed=skip_not_installed)


def test_cosmo_des_y1_clustering_classy(packages_path, skip_not_installed,
                                        info_theory=info_classy):
    test_cosmo_des_y1_clustering_camb(packages_path, info_theory=info_theory,
                                      skip_not_installed=skip_not_installed)


def test_cosmo_des_y1_galaxy_galaxy_classy(packages_path, skip_not_installed,
                                           info_theory=info_classy):
    test_cosmo_des_y1_galaxy_galaxy_camb(packages_path, info_theory=info_theory,
                                         skip_not_installed=skip_not_installed)


ref_chi2 = {"shear": 242.825, "clustering": 100.997,
            "galaxy_galaxy": 208.005, "joint": 570.428}
tolerance = 0.2

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
