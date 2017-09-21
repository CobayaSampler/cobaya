# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CAMB

import pytest

from cobaya.conventions import _theory, _likelihood, _sampler, _params, _path_install
from cobaya.yaml_custom import yaml_load, yaml_dump
from cobaya.run import run

from cosmo_common import baseline_cosmology, derived
from cosmo_common import lik_info_lowl_highTT, lik_info_lowTEB_highTTTEEE

@pytest.mark.slow
def test_camb_planck_slow(modules):
    body_of_test(modules, "p")
    
def body_of_test(modules, x):
    assert modules, "I need a modules folder!"
    info = yaml_load(baseline_cosmology)
    # Add derived
    info[_params][_theory].update(derived)
    print "FOR NOW, POPPING THE BBN PARAMETERS!!!!!!!"
    info[_params][_theory].pop("DH")
    info[_params][_theory].pop("YHe")
    info[_params][_theory].pop("Y_p")
    info.update({_path_install: modules, _theory: {"camb": {"speed": 0.5}}})
    info[_sampler] = {"mcmc": {
        "burn_in": 100,
        "learn_proposal": True,
        "learn_proposal_Rminus1_max": 3.,
        "learn_proposal_Rminus1_max_early": 50.,
        "learn_proposal_Rminus1_min": 0.,
        "Rminus1_stop": 0.01,
        "Rminus1_cl_stop": 0.2,
        "Rminus1_cl_level": 0.95,
        "drag_interp_steps": 3,
        "max_speed_slow": 0.5
    }}
    if x == "t":
        info[_likelihood] = lik_info_lowl_highTT
        info[_sampler]["mcmc"]["covmat"] = "./base_plikHM_TT_lowTEB_0.01theta.covmat"
    elif x == "p":
        info[_likelihood] = lik_info_lowTEB_highTTTEEE
        info[_sampler]["mcmc"]["covmat"] = "./base_plikHM_TTTEEE_lowTEB_0.01theta.covmat"
    else:
        raise ValueError("Test not recognised: %r"%x)
    info["output_prefix"] = "./test_planck/%s_"%x
    info["debug"] = False
#    info["debug_file"] = "test_planck_slow.log"
    print "Input info (dumped to yaml) -------------------------------"
    print yaml_dump(info)
    print "-----------------------------------------------------------"
    updated_info, products = run(info)
