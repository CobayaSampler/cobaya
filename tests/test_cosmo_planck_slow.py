# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CAMB

import pytest

from cobaya.conventions import input_theory, input_likelihood, input_sampler
from cobaya.conventions import input_params
from cobaya.yaml_custom import yaml_custom_load, yaml_custom_dump
from cobaya.run import run

from cosmo_common import baseline_cosmology, derived

@pytest.mark.slow
def test_camb_planck_slow(camb_path, planck_path):
    body_of_test(camb_path, planck_path, "p")
    
def body_of_test(camb_path, planck_path, x):
    assert camb_path, "I need CAMB's folder!"
    info = yaml_custom_load(baseline_cosmology)
    info[input_params][input_theory].update(derived)
    info[input_theory] = {"camb": {"path": camb_path}}
    info[input_sampler] = {"mcmc": {
        "burn_in": 100,
        "learn_proposal": True,
        "learn_proposal_Rminus1_max": 3.,
        "learn_proposal_Rminus1_max_early": 50.,
        "learn_proposal_Rminus1_min": 0.,
        "Rminus1_stop": 0.01,
        "Rminus1_cl_stop": 0.2,
        "Rminus1_cl_level": 0.95,
        "drag_interp_steps": 3
    }}
    if x == "t":
        info[input_likelihood] = {"planck_2015_lowl": {"path": planck_path},
                                  "planck_2015_plikHM_TT": {"path": planck_path}}
        info[input_sampler]["mcmc"]["covmat"] = "./base_plikHM_TT_lowTEB_0.01theta.covmat"
    elif x == "p":
        info[input_likelihood] = {"planck_2015_lowTEB": {"path": planck_path, "speed": 1},
                                  "planck_2015_plikHM_TTTEEE": {"path": planck_path, "speed": 4}}
        info[input_sampler]["mcmc"]["covmat"] = "./base_plikHM_TTTEEE_lowTEB_0.01theta.covmat"
    else:
        raise ValueError("Test not recognised: %r"%x)
#    for lik in info[input_likelihood]:
#        info[input_likelihood][lik].update({"speed": 4})
    info["output_prefix"] = "./test_planck/%s_"%x
    info["debug"] = False
#    info["debug_file"] = "test_planck_slow.log"
    print "Input info (dumped to yaml) -------------------------------"
    print yaml_custom_dump(info)
    print "-----------------------------------------------------------"
    updated_info, products = run(info)

# DELETE ME!!!
if __name__ == "__main__":
    import os
    root = "/lustre/scratch/astro/jt386/projects/sampler/"
    root = "/home/jesus/scratch/sampler"
    test_camb_planck_slow(os.path.join(root, "CAMB"), os.path.join(root, "likelihoods/planck_2015"))
