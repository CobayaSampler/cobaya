from copy import deepcopy

import pytest

from .common_cosmo import body_of_test
from .test_cosmo_planck_2018 import planck_2018_precision

info_likelihood = {
    "planck_2018_lowl.TT": None,
    "planck_2018_lowl.EE": None,
    "planck_NPIPE_highl_CamSpec.TTTEEE": None,
    "planckpr4lensing": {
        "package_install": {
            "github_repository": "carronj/planck_PR4_lensing",
            "min_version": "1.0.2",
        },
    },
}

cosmo_params = {
    "logA": 3.04920413,
    "ns": 0.96399503,
    "theta_MC_100": 1.04240171,
    "H0": 67.39679063149518,
    "omegabh2": 0.02235048,
    "omegach2": 0.12121379,
    "tau": 0.05,
}

nuisance_params = {
    "A_planck": 0.99818025,  # calPlanck
    "amp_143": 10.35947284,
    "amp_217": 18.67072461,
    "amp_143x217": 7.54932654,
    "n_143": 0.83715482,
    "n_217": 0.94987418,
    "n_143x217": 1.23385364,
    "calTE": 0.98781552,
    "calEE": 1.013345,
}

chi2_planck_NPIPE = {
    "planck_2018_lowl.TT": 24.81,
    "planck_2018_lowl.EE": 395.72,
    "planck_NPIPE_highl_CamSpec.TTTEEE": 11341.17,
    "planckpr4lensing": 13.76,
    "tolerance": 0.10,
}

# Fixing some precision and modelling parameters to guarantee test is future-proof.

# We keep the 2017 sBBN tables, because is the only one both in CLASS and CAMB atm.
# But we update HMCode

planck_NPIPE_precision = deepcopy(planck_2018_precision)
planck_NPIPE_precision["camb"].update({"halofit_version": "mead2020"})
planck_NPIPE_precision["classy"].update({"hmcode_version": "2020"})


def test_planck_NPIPE_p_CamSpec_camb(packages_path, skip_not_installed):
    best_fit = deepcopy(cosmo_params)
    best_fit.pop("H0")
    best_fit.update(nuisance_params)
    chi2 = chi2_planck_NPIPE.copy()
    info_theory = {"camb": {"extra_args": planck_NPIPE_precision["camb"]}}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


# TODO: Skipped. Same as 2018 CamSpec tests, there is a DeltaChi2 ~ 7 of unknown origin.
@pytest.mark.skip
def test_planck_NPIPE_p_CamSpec_classy(packages_path, skip_not_installed):
    best_fit = deepcopy(cosmo_params)
    best_fit.pop("theta_MC_100")
    best_fit.update(nuisance_params)
    chi2 = chi2_planck_NPIPE.copy()
    info_theory = {"classy": {"extra_args": planck_NPIPE_precision["classy"]}}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        None,
        skip_not_installed=skip_not_installed,
    )
