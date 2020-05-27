# Tries to evaluate the BK15 likelihood at a reference point

from copy import deepcopy
from .common_cosmo import body_of_test
from cobaya.cosmo_input import cmb_precision

camb_extra = deepcopy(cmb_precision["camb"])
camb_extra.update({"halofit_version": "takahashi"})
classy_extra = deepcopy(cmb_precision["classy"])
classy_extra.update({"non linear": "halofit"})
classy_extra.update({"halofit_min_k_max": 20})


def test_bicep_keck_2015_camb(packages_path, skip_not_installed):
    info_theory = {"camb": {"extra_args": camb_extra}}
    body_of_test(packages_path, test_point, lik_info, info_theory, chi2,
                 extra_model={"primordial": "SFSR_t"},
                 skip_not_installed=skip_not_installed)


def test_bicep_keck_2015_classy(packages_path, skip_not_installed):
    info_theory = {"classy": {"extra_args": classy_extra}}
    # extra tolerance for CLASS
    chi2_classy = deepcopy(chi2)
    chi2_classy["tolerance"] *= 2
    body_of_test(packages_path, test_point, lik_info, info_theory, chi2_classy,
                 extra_model={"primordial": "SFSR_t"},
                 skip_not_installed=skip_not_installed)


lik_info = {"bicep_keck_2015": {}}

# NB: chi2 obtained using CAMB w HMcode
chi2 = {"bicep_keck_2015": 735.187, "tolerance": 0.16}

test_point = {
    "omegabh2": 0.2235620E-01,
    "omegach2": 0.1204042E+00,
    "theta_MC_100": 0.1040871E+01,  # for CAMB
    "H0": 0.6718506E+02,  # for CLASS
    "tau": 0.5454114E-01,
    "logA": 0.3046322E+01,
    "ns": 0.9654113E+00,
    "r": 0.1451578E-01,
    "calPlanck": 0.1000689E+01,
    "BBdust": 0.4648994E+01,
    "BBsync": 0.1542620E+01,
    "BBalphadust": -0.5338430E+00,
    "BBbetadust": 0.1576173E+01,
    "BBalphasync": -0.1915241E+00,
    "BBbetasync": -0.3040606E+01,
    "BBdustsynccorr": -0.3441905E+00
}
