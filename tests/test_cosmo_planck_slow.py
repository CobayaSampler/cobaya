# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CAMB

from __future__ import division
import pytest
import numpy as np

from cobaya.conventions import _theory, _likelihood, _sampler, _params, _path_install
from cobaya.yaml_custom import yaml_load, yaml_dump
from cobaya.run import run

from cosmo_common import baseline_cosmology, baseline_cosmology_classy_extra, derived
from cosmo_common import lik_info_lowl_highTT, lik_info_lowTEB_highTTTEEE, adapt_covmat

@pytest.mark.slow
def test_camb_planck_slow(modules, tmpdir, debug=False):
    body_of_test(modules, tmpdir, "p", theory="camb", debug=debug)

@pytest.mark.slow
def test_classy_planck_slow(modules, tmpdir, debug=False):
    body_of_test(modules, tmpdir, "p", theory="classy", debug=debug)
    
def body_of_test(modules, tmpdir, x, theory, debug=False):
    assert modules, "I need a modules folder!"
    info = {_path_install: modules}
    info.update(yaml_load(baseline_cosmology))
    # Add derived
    info[_params][_theory].update(derived)
    print "FOR NOW, POPPING THE BBN PARAMETERS!!!!!!!"
    for p in ("YHe", "Y_p", "DH"):
        info[_params][_theory].pop(p, None)
    if theory == "camb":
        info.update({_theory: {"camb": None}})
    elif theory == "classy":
        info.update({_theory: {"classy": None}})
        info[_params][_theory].update(baseline_cosmology_classy_extra)
        info[_params][_theory].pop("cosmomc_theta")
        info[_params][_theory]["theta_s_100"] = info[_params][_theory].pop("cosmomc_theta_100")
        info[_params][_theory]["theta_s_100"]["ref"]["loc"] = 1.0418
        info[_params][_theory]["theta_s_100"]["latex"] = r"100*\theta_s"
        info[_params][_theory]["100*theta_s"] = "lambda theta_s_100: theta_s_100"
        info[_params][_theory]["omegam"].pop("derived")
        info[_params][_theory]["omegamh2"]["derived"] = "lambda omegam, H0: omegam*(H0/100)**2"
        info[_params][_theory]["omegamh3"]["derived"] = "lambda omegam, H0: omegam*(H0/100)**3"
        info[_params][_theory]["s8omegamp5"]["derived"] = "lambda sigma8, omegam: sigma8*omegam**0.5"
        info[_params][_theory]["s8omegamp25"]["derived"] = "lambda sigma8, omegam: sigma8*omegam**0.25"
        # Not yet implemented
        for p in ["zstar", "rstar", "thetastar", "DAstar", "zdrag", "rdrag",
                  "kd", "thetad", "zeq", "keq", "thetaeq", "thetarseq"]:
            info[_params][_theory].pop(p)
    info[_theory][info[_theory].keys()[0]] = {"speed": 0.5}
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
        covmat_file = "./base_plikHM_TT_lowTEB.covmat"
    elif x == "p":
        info[_likelihood] = lik_info_lowTEB_highTTTEEE
        covmat_file = "./base_plikHM_TTTEEE_lowTEB.covmat"
    else:
        raise ValueError("Test not recognised: %r"%x)
    # Change Planck's official CosmoMC covmat
    info[_sampler]["mcmc"]["covmat"] = adapt_covmat(covmat_file, tmpdir, theory)
    info["output_prefix"] = "./chains_planck_%s_%s/"%(theory, x)
    info["debug"] = debug
#    info["debug_file"] = "test_planck_slow.log"
    print "Input info (dumped to yaml) -------------------------------"
    print yaml_dump(info)
    print "-----------------------------------------------------------"
    updated_info, products = run(info)
