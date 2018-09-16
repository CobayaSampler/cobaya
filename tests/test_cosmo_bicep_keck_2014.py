# Tries to evaluate the BK14 likelihood at a reference point

from copy import deepcopy

from common_cosmo import body_of_test
from cobaya.cosmo_input import cmb_precision

# Small chi2 difference with CLASS (total still <0.8)
classy_extra_tolerance = 0.7


def test_bicep_keck_2014_camb(modules):
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    body_of_test(modules, test_point, lik_info, info_theory, chi2,
                 extra_model={"primordial": "SFSR_t"})


def test_bicep_keck_2014_classy(modules):
    info_theory = {"classy": {"extra_args": cmb_precision["classy"]}}
    chi2_classy = deepcopy(chi2)
    chi2_classy["tolerance"] += classy_extra_tolerance
    body_of_test(modules, test_point, lik_info, info_theory, chi2_classy,
                 extra_model={"primordial": "SFSR_t"})


# Best fit ###############################################################################
# From non-public chains

lik_info = {"bicep_keck_2014": {}}

chi2 = {"bicep_keck_2014": 642.244, "tolerance": 0.15}

test_point = {
    "omegabh2": 0.2236168E-01,
    "omegach2": 0.1204742E+00,
    "theta": 0.1040852E+01,  # for CAMB
    "H0": 0.6716064E+02,  # for CLASS
    "tau": 0.5453638E-01,
    "logA": 0.3046414E+01,
    "ns": 0.9657577E+00,
    "r": 0.1647227E-01,
    "calPlanck": 0.1000649E+01,
    "BBdust": 0.4064084E+01,
    "BBsync": 0.1484951E+01,
    "BBalphadust": -0.1290001E+00,
    "BBbetadust": 0.1581963E+01,
    "BBalphasync": -0.5495452E+00,
    "BBbetasync": -0.3056026E+01,
    "BBdustsynccorr": 0.2681231E-04,
}
