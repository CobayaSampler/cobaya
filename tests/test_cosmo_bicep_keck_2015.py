import numpy as np

from cobaya.conventions import _path_install, _theory, _sampler, _likelihood, _params
from cobaya.run import run


def test_cosmo_bicep_keck_2015_camb(modules, theory="camb"):
    assert modules, "I need a modules folder!"
    info = {_path_install: modules,
            _theory: {theory: None},
            _sampler: {"evaluate": None}}
    info[_likelihood] = {"bicep_keck_2015": None}
    info[_params] = {
        "theory": {
            "ombh2": 0.2224017E-01,
            "omch2": 0.1192851E+00,
            "cosmomc_theta": 0.1040761E-1,
            "tau": 0.7602569E-01,
            "As": 1e-10*np.exp(0.3081122E+01),
            "ns": 0.9633217E+00,
        },
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
    updated_info, products = run(info)
    # print products["sample"]
    reference_value = 650.872548
    abs_tolerance = 0.1
    computed_value = products["sample"]["chi2__"+info[_likelihood].keys()[0]].values[0]
    assert (computed_value-reference_value) < abs_tolerance
