import numpy as np

from cobaya.conventions import _path_install, _theory, _sampler, _likelihood, _params
from cobaya.yaml_custom import yaml_load
from cobaya.run import run
from cosmo_common import baseline_cosmology


def test_cosmo_bicep_keck_2015_camb(modules, x, theory="camb"):
    assert modules, "I need a modules folder!"
    info = {_path_install: modules,
            _theory: {theory: None},
            _sampler: {"evaluate": None}}
#    if x == "dust":
    info[_likelihood] = {"bicep_keck_2015": None}#, "planck_2015_lowTEB": None}
    info.update(yaml_load(baseline_cosmology))
    # Some values for nuisance params
    info[_params].update({
        "theory": {
            "ombh2": 0.2224017E-01,
            "omch2": 0.1192851E+00,
            "cosmomc_theta": 0.1040761E-1,
            "tau": 0.7602569E-01,
            "As": 10**-10*np.exp(0.3081122E+01),
            "ns": 0.9633217E+00},
        'BBdust': 3,
        'BBsync': 1,
        'BBalphadust': -0.42,
        'BBalphasync': -0.6,
        'BBbetadust': 1.59,
        'BBbetasync': -3.1,
        'BBdustsynccorr': 0.2,
        'EEtoBB_dust': 2,
        'EEtoBB_sync': 2,
        'BBTdust': 19.6})
    from cobaya.yaml_custom import yaml_dump
    print yaml_dump(info)#["params"])
    updated_info, products = run(info)
    # print products["sample"]
