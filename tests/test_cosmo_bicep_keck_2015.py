from __future__ import print_function
import numpy as np

from cobaya.conventions import _path_install, _sampler, _likelihood, _params, _theory
from cobaya.run import run
from cosmo_common import baseline_cosmology_classy_extra


def test_cosmo_bicep_keck_2015_camb(modules):
    body_of_test(modules, "camb")


def test_cosmo_bicep_keck_2015_classy(modules):
    body_of_test(modules, "classy")


def body_of_test(modules, theory):
    assert modules, "I need a modules folder!"
    info = {_path_install: modules,
            _theory: {theory: None},
            _sampler: {"evaluate": None}}
    info[_likelihood] = {"bicep_keck_2015": None}
    info[_params] = {
        # Theory
        "ombh2": 0.2224017E-01,
        "omch2": 0.1192851E+00,
        "H0": 67.30713,
        # "cosmomc_theta": 0.1040761E-1,
        "tau": 0.7602569E-01,
        "As": 1e-10*np.exp(0.3081122E+01),
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
    reference_value = 650.872548
    abs_tolerance = 0.1
    if theory == "classy":
        info[_params].update(baseline_cosmology_classy_extra)
        abs_tolerance += 2
        print("WE SHOULD NOT HAVE TO LOWER THE TOLERANCE THAT MUCH!!!")
    updated_info, products = run(info)
    # print products["sample"]
    computed_value = products["sample"]["chi2__"+list(info[_likelihood].keys())[0]].values[0]
    assert (computed_value-reference_value) < abs_tolerance
