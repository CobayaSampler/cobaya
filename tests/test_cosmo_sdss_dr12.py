from __future__ import print_function
import numpy as np
import pytest

from cobaya.conventions import _path_install, _theory, _sampler, _likelihood, _params
from cobaya.run import run
from cobaya.yaml import yaml_load
from cosmo_common import baseline_cosmology_classy_extra


@pytest.mark.slow
def test_cosmo_sdss_dr12_consensus_bao(modules):
    body_of_test(modules, "consensus_bao", "camb")


def body_of_test(modules, data, theory):
    assert modules, "I need a modules folder!"
    info = {_path_install: modules,
            _theory: {theory: None},
            _sampler: {"evaluate": None}}
    info[_likelihood] = {"sdss_dr12_"+data: None}
    info[_params] = {
        _theory: {
            "H0": 68.81,
            "ombh2": 0.0468 * 0.6881 ** 2,
            "omch2": (0.295 - 0.0468) * 0.6881 ** 2 - 0.0006155,
            "YHe": 0.245341,
            "tau": 0.08,
            "As": 2.260574e-09,
            "ns": 0.9676
        }}
#    if data in ["shear", "galaxy_galaxylensing"]:
#        info[_params].update(yaml_load(test_params_shear)[_params])
#    if data in ["clustering", "galaxy_galaxylensing"]:
#        info[_params].update(yaml_load(test_params_clustering)[_params])


    # UPDATE WITH BOTH ANYWAY FOR NOW!!!!!
#    info[_params].update(yaml_load(test_params_shear)[_params])
#    info[_params].update(yaml_load(test_params_clustering)[_params])
    
    
    reference_value = 650.872548
    abs_tolerance = 0.1
#    if theory == "classy":
#        info[_params][_theory].update(baseline_cosmology_classy_extra)
#        abs_tolerance += 2
#        print("WE SHOULD NOT HAVE TO LOWER THE TOLERANCE THAT MUCH!!!")
    updated_info, products = run(info)
    print(products["sample"])
    computed_value = products["sample"]["chi2__"+list(info[_likelihood].keys())[0]].values[0]
    assert (computed_value-reference_value) < abs_tolerance
