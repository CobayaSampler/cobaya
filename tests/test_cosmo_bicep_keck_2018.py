# Tries to evaluate the BK18 likelihood at a reference point

from copy import deepcopy

from cobaya.cosmo_input import planck_precision

from .common_cosmo import body_of_test

camb_extra = deepcopy(planck_precision["camb"])
camb_extra.update({"halofit_version": "takahashi"})
classy_extra = deepcopy(planck_precision["classy"])
classy_extra.update({"non linear": "halofit"})
classy_extra.update({"nonlinear_min_k_max": 20})


def test_bicep_keck_2018_camb(packages_path, skip_not_installed):
    info_theory = {"camb": {"extra_args": camb_extra}}
    body_of_test(
        packages_path,
        test_point,
        lik_info,
        info_theory,
        chi2,
        extra_model={"primordial": "SFSR_t"},
        skip_not_installed=skip_not_installed,
    )


def test_bicep_keck_2018_classy(packages_path, skip_not_installed):
    info_theory = {"classy": {"extra_args": classy_extra}}
    body_of_test(
        packages_path,
        test_point,
        lik_info,
        info_theory,
        chi2,
        extra_model={"primordial": "SFSR_t"},
        skip_not_installed=skip_not_installed,
    )


lik_info = {"bicep_keck_2018": {}}

# NB: chi2 obtained using CAMB w HMcode
chi2 = {"bicep_keck_2018": 543.25, "tolerance": 0.16}

test_point = {
    "omegabh2": 0.2235620e-01,
    "omegach2": 0.1204042e00,
    "theta_MC_100": 0.1040871e01,  # for CAMB
    "H0": 0.6718506e02,  # for CLASS
    "tau": 0.5454114e-01,
    "logA": 0.3046322e01,
    "ns": 0.9654113e00,
    "r": 0.1451578e-01,
    "calPlanck": 0.1000689e01,
    "BBdust": 0.4648994e01,
    "BBsync": 0.1542620e01,
    "BBalphadust": -0.5338430e00,
    "BBbetadust": 0.1576173e01,
    "BBalphasync": -0.1915241e00,
    "BBbetasync": -0.3040606e01,
    "BBdustsynccorr": -0.3441905e00,
}
