# Tries to evaluate the BK2015 likelihood at a reference point

from copy import deepcopy

from cosmo_common import body_of_test


def test_bicep_keck_2015_camb(modules):
    info_theory = {"camb": None}
    body_of_test(modules, params, lik_info, info_theory, chi2, derived)


def test_bicep_keck_2015_classy(modules):
    info_theory = {"classy": {"use_camb_names": True}}
    chi2_classy = deepcopy(chi2)
    chi2_classy["tolerance"] += 2.0
    body_of_test(modules, params, lik_info, info_theory, chi2_classy, derived)


# Best fit ###############################################################################

lik_info = {"bicep_keck_2015": None}

chi2 = {"bicep_keck_2015": 650.872548, "tolerance": 0.1}

params = {
    # Theory
    "ombh2": 0.02224017,
    "omch2": 0.1192851,
    "H0": 67.44495,  # IGNORED BY CAMB
    "cosmomc_theta_100": 1.040761,  # IGNORED BY CLASSY
    "tau": 0.7602569E-01,
    "logAs1e10": 3.081122,
    "ns": 0.9633217E+00,
    # Experimental
    'BBdust': 3,
    'BBsync': 1,
    'BBalphadust': -0.42,
    'BBalphasync': -0.6,
    'BBbetadust': 1.59,
    'BBbetasync': -3.1,
    'BBdustsynccorr': 0.2,
    'EEtoBB_dust': 2,
    'EEtoBB_sync': 2,
    'BBTdust': 19.6}

# Not a real 1d posterior for H0, just for tests
derived = {"H0": [params["H0"], 0.001]}
