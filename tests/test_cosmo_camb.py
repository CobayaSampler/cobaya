# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CAMB

from __future__ import division

from cobaya.conventions import _theory, _likelihood, _sampler, _derived_pre
from cobaya.conventions import _params, _chi2, separator, _path_install
from cobaya.yaml_custom import yaml_custom_load
from cobaya.run import run

from cosmo_common import params_lowl_highTT, chi2_lowl_highTT, derived_lowl_highTT
from cosmo_common import params_lowTEB_highTTTEEE, chi2_lowTEB_highTTTEEE#, derived_lowTEB_highTTTEEE
from cosmo_common import derived, tolerance_chi2_abs, tolerance_derived


def test_camb_planck_t(modules):
    body_of_test(modules, "t")
    
def test_camb_planck_p(modules):
    body_of_test(modules, "p")

def body_of_test(modules, x):
    assert modules, "I need a modules folder!"
    info = {_path_install: modules,
            _theory: {"camb": None},
            _sampler: {"evaluate": None}}

    ## TEST!!!!!! update for BBN!!!!!
    info["theory"]["camb"] = {"path":"/home/jesus/scratch/CAMB"}

    
    if x == "t":
        info[_likelihood] = {"planck_2015_lowl": None,
                                  "planck_2015_plikHM_TT": None}
        info.update(yaml_custom_load(params_lowl_highTT))
        ref_chi2 = chi2_lowl_highTT
        derived_values = derived_lowl_highTT
    elif x == "p":
        info[_likelihood] = {"planck_2015_lowTEB": None,
                                  "planck_2015_plikHM_TTTEEE": None}
        info.update(yaml_custom_load(params_lowTEB_highTTTEEE))
        ref_chi2 = chi2_lowTEB_highTTTEEE
        derived_values = derived_lowl_highTT
        print "TODO: ~~~~~~ change previous line to correct TTTEEE derived ~~~~~~"
    else:
        raise ValueError("Test not recognised: %r"%x)
    use_H0_instead_of_theta = False
    if use_H0_instead_of_theta:
        info[_params][_theory].pop("cosmomc_theta")
        derived.pop("H0")
        derived_values.pop("H0")
    else:
        info[_params][_theory].pop("H0")
    # Add derived
    info[_params][_theory].update(derived)
    updated_info, products = run(info)
    # print products["sample"]
    # Check value of likelihoods
    for lik in info[_likelihood]:
        chi2 = products["sample"][_chi2+separator+lik][0]
        assert abs(chi2-ref_chi2[lik]) < tolerance_chi2_abs, (
            "Likelihood value for '%s' off by more than %f!"%(lik, tolerance_chi2_abs))
    # Check value of derived parameters
    not_tested = []
    not_passed = []
    for p in derived_values:
        if derived_values[p][0] == None:
            not_tested += [p]
            continue
        rel = (abs(products["sample"][ _derived_pre+p][0]-derived_values[p][0])
               /derived_values[p][1])
        if rel > tolerance_derived:
            not_passed += [(p, rel)]
    print "Derived parameters not tested because not implemented: %r"%not_tested
    assert not(not_passed), "Some derived parameters were off: %r"%not_passed
