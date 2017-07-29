# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CLASS

from cobaya.conventions import input_theory, input_likelihood, input_sampler
from cobaya.conventions import input_params, _chi2, separator
from cobaya.yaml_custom import yaml_custom_load
from cobaya.run import run

from cosmo_common import params_lowl_highTT, chi2_lowl_highTT
from cosmo_common import params_lowTEB_highTTTEEE, chi2_lowTEB_highTTTEEE
from cosmo_common import derived, tolerance_abs

def test_classy_planck_t(classy_path, planck_path):
    body_of_test(classy_path, planck_path, "t")

def test_classy_planck_p(classy_path, planck_path):
    body_of_test(classy_path, planck_path, "p")

def body_of_test(classy_path, planck_path, x):
    assert classy_path, "I need CLASSY's folder!"
    info = {input_theory: {"classy": {"path": classy_path}},
            input_sampler: {"evaluate": None}}
    if x == "t":
        info[input_likelihood] = {"planck_2015_lowl": {"path": planck_path},
                                  "planck_2015_plikHM_TT": {"path": planck_path}}
        info.update(yaml_custom_load(params_lowl_highTT))
        ref_chi2 = chi2_lowl_highTT
    elif x == "p":
        info[input_likelihood] = {"planck_2015_lowTEB": {"path": planck_path},
                                  "planck_2015_plikHM_TTTEEE": {"path": planck_path}}
        info.update(yaml_custom_load(params_lowTEB_highTTTEEE))
        ref_chi2 = chi2_lowTEB_highTTTEEE
    else:
        raise ValueError("Test not recognised: %r"%x)
    # Remove cosmomc_theta in favour of H0
    info[input_params][input_theory].pop("cosmomc_theta")
    # Add derived
### TODO    info[input_params][input_theory].update(derived)
    # CLASS' specific stuff to compute Planck's baseline LCDM
    info[input_params][input_theory].update({
        "N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06, "T_ncdm": 0.71611})
    updated_info, products = run(info)
    # print products["sample"]
    # Check value of likelihoods
    for lik in info[input_likelihood]:
        chi2 = products["sample"][_chi2+separator+lik][0]
        assert abs(chi2-ref_chi2[lik]) < tolerance_abs, (
            "Likelihood value for '%s' off by more than %f!"%(lik, tolerance_abs))
    # Check value of derived parameters
    ###################
    ###################
    ###################
