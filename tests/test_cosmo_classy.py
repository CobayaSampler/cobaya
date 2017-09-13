# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CLASS

from __future__ import division

from cobaya.conventions import _theory, _likelihood, _sampler
from cobaya.conventions import _params, _chi2, separator, _path_install
from cobaya.yaml_custom import yaml_custom_load
from cobaya.run import run

from cosmo_common import params_lowl_highTT, chi2_lowl_highTT
from cosmo_common import params_lowTEB_highTTTEEE, chi2_lowTEB_highTTTEEE
from cosmo_common import derived, tolerance_chi2_abs

def test_classy_planck_t(modules):
    body_of_test(modules, "t")

def test_classy_planck_p(modules):
    body_of_test(modules, "p")

def body_of_test(modules, x):
    assert modules, "I need a modules folder!"
    info = {_path_install: modules,
            _theory: {"classy": None},
            _sampler: {"evaluate": None}}
    if x == "t":
        info[_likelihood] = {"planck_2015_lowl": None,
                                  "planck_2015_plikHM_TT": None}
        info.update(yaml_custom_load(params_lowl_highTT))
        ref_chi2 = chi2_lowl_highTT
    elif x == "p":
        info[_likelihood] = {"planck_2015_lowTEB": None,
                                  "planck_2015_plikHM_TTTEEE": None}
        info.update(yaml_custom_load(params_lowTEB_highTTTEEE))
        ref_chi2 = chi2_lowTEB_highTTTEEE
    else:
        raise ValueError("Test not recognised: %r"%x)
    # Remove cosmomc_theta in favour of H0
    info[_params][_theory].pop("cosmomc_theta")
    # Add derived
    derived.pop("H0")
    # Aboundances disabled for now!
    for p in ["YHe", "Y_p", "DH",
              "zstar",
              "rstar",
              "thetastar",
              "DAstar",
              "zdrag",
              "rdrag",
              "kd",
              "thetad",
              "zeq",
              "keq",
              "thetaeq",
              "thetarseq"]:
        derived.pop(p)
    info[_params][_theory].update(derived)
    # CLASS' specific stuff to compute Planck's baseline LCDM
    info[_params][_theory].update({
        "N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06, "T_ncdm": 0.71611,
        # Seems not to be necessary (but clarify, and add basestring to the fixed param check:
        # "sBBN file": modules+"/theories/CLASS/bbn/sBBN.dat",
    })
    print info[_params][_theory]
    updated_info, products = run(info)
    # print products["sample"]
    # Check value of likelihoods
    for lik in info[_likelihood]:
        chi2 = products["sample"][_chi2+separator+lik][0]
        assert abs(chi2-ref_chi2[lik]) < tolerance_chi2_abs, (
            "Likelihood value for '%s' off by more than %f!"%(lik, tolerance_chi2_abs))
    # Check value of derived parameters
    ###################
    ###################
    ###################
