from __future__ import division
import numpy as np
import os
from copy import deepcopy

from cobaya.conventions import _theory, _likelihood, _params, _derived_pre
from cobaya.conventions import _sampler, _chi2, separator, _path_install
from cobaya.yaml_custom import yaml_load
from cobaya.run import run

# Tolerance for the tests
tolerance_chi2_abs = 0.1
tolerance_derived = 0.03
# this last one cannot be smaller: rounding errors in BBN abundances, bc new interp tables


# Converting 100cosmomc_theta to cosmomc_theta in Planck's covmats #######################

def adapt_covmat(filename, tmpdir, theory="camb", theta_factor=100):
    with open(filename, "r") as original:
        params = original.readline()[1:].split()
        covmat = np.loadtxt(filename)
    i_logA = params.index("logA")
    params[i_logA] = "logAs1e10"
    i_theta = params.index("cosmomc_theta")
    if theory == "camb":
        params[i_theta] = "cosmomc_theta_100"
    elif theory == "classy":
        params[i_theta] = "theta_s_100"
    # if used for cosmomc_theta or theta_s, not their multiples
    if theta_factor != 100:
        covmat[i_theta, :] /= (100/theta_factor)
        covmat[:, i_theta] /= (100/theta_factor)
    filename_new = os.path.join(str(tmpdir),"covmat.dat")
    np.savetxt(filename_new, covmat, fmt="%.8g", header=" ".join(params))
    return filename_new


# Body of the best-fit test ##############################################################

def body_of_test(modules, x, theory):
    assert modules, "I need a modules folder!"
    info = {_path_install: modules,
            _theory: {theory: None},
            _sampler: {"evaluate": None}}
    if x == "t":
        info[_likelihood] = lik_info_lowl_highTT
        info.update(yaml_load(params_lowl_highTT))
        ref_chi2 = chi2_lowl_highTT
        derived_values = derived_lowl_highTT
    elif x == "p":
        info[_likelihood] = lik_info_lowTEB_highTTTEEE
        info.update(yaml_load(params_lowTEB_highTTTEEE))
        ref_chi2 = chi2_lowTEB_highTTTEEE
        derived_values = derived_lowTEB_highTTTEEE
    else:
        raise ValueError("Test not recognised: %r"%x)
    derived_bestfit_test = deepcopy(derived)
    # Adjustments for Classy
    if theory == "classy":
        # Remove "cosmomc_theta" in favour of "H0" (remove it from derived then!)
        info[_params][_theory].pop("cosmomc_theta")
        derived_bestfit_test.pop("H0")
        derived_values.pop("H0")
        # Don't test those that have not been implemented yet
        for p in ["zstar", "rstar", "thetastar", "DAstar", "zdrag", "rdrag",
                  "kd", "thetad", "zeq", "keq", "thetaeq", "thetarseq"]:
            derived_bestfit_test.pop(p)
        # Adapt the definitions of some derived parameters
        derived_bestfit_test["omegam"].pop("derived")
        derived_bestfit_test["omegamh2"]["derived"] = "lambda omegam, H0: omegam*(H0/100)**2"
        derived_bestfit_test["omegamh3"]["derived"] = "lambda omegam, H0: omegam*(H0/100)**3"
        derived_bestfit_test["s8omegamp5"]["derived"] = "lambda sigma8, omegam: sigma8*omegam**0.5"
        derived_bestfit_test["s8omegamp25"]["derived"] = "lambda sigma8, omegam: sigma8*omegam**0.25"
        # More stuff that CLASS needs for the Planck model
        info[_params][_theory].update(baseline_cosmology_classy_extra)
    # Add derived
    info[_params][_theory].update(derived_bestfit_test)
    print "FOR NOW, POPPING THE BBN PARAMETERS!!!!!!!"
    for p in ("YHe", "Y_p", "DH"):
        info[_params][_theory].pop(p, None)
        derived_values.pop(p, None)
    updated_info, products = run(info)
    # print products["sample"]
    # Check value of likelihoods
    for lik in info[_likelihood]:
        chi2 = products["sample"][_chi2+separator+lik][0]
        tolerance = tolerance_chi2_abs + (2.1 if theory == "classy" else 0)
        assert abs(chi2-ref_chi2[lik]) < tolerance, (
            "Likelihood value for '%s' off by more than %f!"%(lik, tolerance_chi2_abs))
    # Check value of derived parameters
    not_tested = []
    not_passed = []
    for p in derived_values:
        if derived_values[p][0] is None or p not in derived_bestfit_test:
            not_tested += [p]
            continue
        rel = (abs(products["sample"][_derived_pre+p][0]-derived_values[p][0])
               / derived_values[p][1])
        if rel > tolerance_derived*(
                2 if p in ("YHe", "Y_p", "DH", "sigma8", "s8omegamp5") else 1):
            not_passed += [(p, rel)]
    print "Derived parameters not tested because not implemented: %r"%not_tested
    assert not(not_passed), "Some derived parameters were off: %r"%not_passed


# Baseline priors ########################################################################

baseline_cosmology = r"""
%s:
  %s:
    # dummy prior for ombh2 so that the sampler does not complain
    ombh2:
      prior:
        min: 0.005
        max: 0.1
      ref:
        dist: norm
        loc: 0.0221
        scale: 0.0001
      proposal: 0.0001
      latex: \Omega_\mathrm{b} h^2
    omch2:
      prior:
        min: 0.001
        max: 0.99
      ref:
        dist: norm
        loc: 0.12
        scale: 0.001
      proposal: 0.0005
      latex: \Omega_\mathrm{c} h^2
    # If using CLASS, rename to "100*theta_s"!!!
    cosmomc_theta: "lambda cosmomc_theta_100: 1.e-2*cosmomc_theta_100"
    cosmomc_theta_100:
      prior:
        min: 0.5
        max: 100
      ref:
        dist: norm
        loc: 1.0411
        scale: 0.0004
      proposal: 0.0002
      latex: 100\theta_\mathrm{MC}
      drop:
    tau:
      prior:
        min: 0.01
        max: 0.8
      ref:
        dist: norm
        loc: 0.09
        scale: 0.01
      proposal: 0.005
      latex: \tau_\mathrm{reio}
    logAs1e10:
      prior:
        min: 2
        max: 4
      ref:
        dist: norm
        loc:   3.1
        scale: 0.001
      proposal: 0.001
      latex: \log(10^{10} A_s)
      drop:
    As: "lambda logAs1e10: 1e-10*np.exp(logAs1e10)"
    ns:
      prior:
        min: 0.8
        max: 1.2
      ref:
        dist: norm
        loc: 0.96
        scale: 0.004
      proposal: 0.002
      latex: n_\mathrm{s}
"""%(_params, _theory)

baseline_cosmology_classy_extra = {"N_ur": 2.0328, "N_ncdm": 1,
                                   "m_ncdm": 0.06, "T_ncdm": 0.71611}

# Derived parameters, described in
# https://wiki.cosmos.esa.int/planckpla2015/images/b/b9/Parameter_tag_definitions_2015.pdf
derived = {
    "H0":          {"latex": r"H_0"},
    "omegav":      {"latex": r"\Omega_\Lambda"},
    "omegam":      {"derived": "lambda omegab, omegac, omegan: omegab+omegac+omegan", "latex": r"\Omega_m"},
    "omegamh2":    {"derived": "lambda omegab, omegac, omegan, H0: (omegab+omegac+omegan)*(H0/100)**2", "latex": r"\Omega_m h^2"},
    "omegamh3":    {"derived": "lambda omegab, omegac, omegan, H0: (omegab+omegac+omegan)*(H0/100)**3", "latex": r"\Omega_m h^3"},
#    "omeganuh2":   {"derived": "lambda omegan, H0: omegan*(H0*1e-2)**2", "latex": r"\Omega_\nu h^2"},
    "sigma8":      {"latex": r"\sigma_8"},
    "s8h5":        {"derived": "lambda sigma8, H0: sigma8*(H0*1e-2)**(-0.5)", "latex": r"\sigma_8/h^{0.5}"},
    "s8omegamp5":  {"derived": "lambda sigma8, omegab, omegac, omegan: sigma8*(omegab+omegac+omegan)**0.5", "latex": r"\sigma_8 \Omega_m^{0.5}"},
    "s8omegamp25": {"derived": "lambda sigma8, omegab, omegac, omegan: sigma8*(omegab+omegac+omegan)**0.25", "latex": r"\sigma_8 \Omega_m^{0.25}"},
    "zre":         {"latex": r"z_\mathrm{re}"},
    "As1e9":       {"derived": "lambda As: 1e9*As", "latex": r"10^9 A_s"},
    "clamp":       {"derived": "lambda As, tau: 1e9*As*np.exp(-2*tau)", "latex": r"10^9 A_s e^{-2\tau}"},
    "YHe":         {"latex": r"Y_P"},
    "Y_p":         {"latex": r"Y_P^\mathrm{BBN}"},
    "DH":          {"latex": r"10^5D/H"},
    "age":         {"latex": r"{\rm{Age}}/\mathrm{Gyr}"},
    "zstar":       {"latex": r"z_*"},
    "rstar":       {"latex": r"r_*"},
    "thetastar":   {"latex": r"100\theta_*"},
    "DAstar":      {"latex": r"D_\mathrm{A}/\mathrm{Gpc}"},
    "zdrag":       {"latex": r"z_\mathrm{drag}"},
    "rdrag":       {"latex": r"r_\mathrm{drag}"},
    "kd":          {"latex": r"k_\mathrm{D}"},
    "thetad":      {"latex": r"100\theta_\mathrm{D}"},
    "zeq":         {"latex": r"z_\mathrm{eq}"},
    "keq":         {"latex": r"k_\mathrm{eq}"},
    "thetaeq":     {"latex": r"100\theta_\mathrm{eq}"},
    "thetarseq":   {"latex": r"100\theta_\mathrm{s,eq}"},
    }


# Temperature only #######################################################################

# NB: A_sz and ksz_norm need to have a prior defined, though "evaluate" will ignore it in
# favour of the fixed "ref" value. This needs to be so since, at the time of writing this
# test, explicit multiparameter priors need to be functions of *sampled* parameters.
# Hopefully this requirement can be dropped in the near future.

lik_info_lowl_highTT = {"planck_2015_lowl": None, "planck_2015_plikHM_TT": None}

chi2_lowl_highTT = {"planck_2015_lowl": 15.39,
                      "planck_2015_plikHM_TT": 761.09}

params_lowl_highTT = """
%s:
  %s:
    # Sampled
    # dummy prior for ombh2 so that the sampler does not complain
    ombh2:
      prior:
        min: 0.005
        max: 0.1
      ref: 0.02249139
    omch2: 0.1174684
    # only one of the next two is finally used!
    H0: 68.43994
    cosmomc_theta: 0.01041189
    tau: 0.1249913
    As: 2.401687e-9
    ns: 0.9741693
    # Derived
  # Planck likelihood
  A_planck: 1.00027
  A_cib_217: 61.1
  xi_sz_cib: 0.56
  A_sz:
    prior:
      dist: uniform
      min: 0
      max: 10
    ref: 6.84
  ps_A_100_100: 242.9
  ps_A_143_143:  43.0
  ps_A_143_217:  46.1
  ps_A_217_217: 104.1
  ksz_norm:
    prior:
      dist: uniform
      min: 0
      max: 10
    ref: 0.00
  gal545_A_100:      7.31
  gal545_A_143:      9.07
  gal545_A_143_217: 17.99
  gal545_A_217:     82.9
  calib_100T: 0.99796
  calib_217T: 0.99555
"""%(_params, _theory)

derived_lowl_highTT = {
    # param: [best_fit, sigma]
    "H0": [68.44, 1.2],
    "omegav": [0.6998, 0.016],
    "omegam": [0.3002, 0.016],
    "omegamh2": [0.1406, 0.0024],
    "omegamh3": [0.09623, 0.00046],
    "sigma8": [0.8610, 0.023],
    "s8omegamp5": [0.472, 0.014],
    "s8omegamp25":[0.637, 0.016],
    "s8h5":       [1.041, 0.025],
    "zre":   [13.76, 2.5],
    "As1e9":  [2.40, 0.15],
    "clamp":  [1.870468, 0.01535354],
    "YHe":    [0.2454462, 0.0001219630],
    "Y_p":    [0.2467729, 0.0001224069],
    "DH":     [2.568606e-5, 0.05098625e-5],
    "age":    [13.7664, 0.048],
    "zstar":  [1089.55, 0.52],
    "rstar":  [145.00, 0.55],
    "thetastar": [1.041358, 0.0005117986],
    "DAstar":  [13.924, 0.050],
    "zdrag":   [1060.05, 0.52],
    "rdrag":   [147.63, 0.53],
    "kd":      [0.14039, 0.00053],
    "thetad":  [0.160715, 0.00029],
    "zeq":     [3345, 58],
    "keq":     [0.010208, 0.00018],
    "thetaeq":  [0.8243, 0.011],
    "thetarseq": [0.4550, 0.0058],
    }


# Best fit Polarisation ##################################################################

# NB: A_sz and ksz_norm need to have a prior defined, though "evaluate" will ignore it in
# favour of the fixed "ref" value. This needs to be so since, at the time of writing this
# test, explicit multiparameter priors need to be functions of *sampled* parameters.
# Hopefully this requirement can be dropped in the near future.

lik_info_lowTEB_highTTTEEE = {"planck_2015_lowTEB": {"speed": 0.25},
                              "planck_2015_plikHM_TTTEEE": None}

chi2_lowTEB_highTTTEEE = {"planck_2015_lowTEB": 10496.93,
                          "planck_2015_plikHM_TTTEEE": 2431.65}

params_lowTEB_highTTTEEE = """
%s:
  %s:
    # Sampled
    # dummy prior for ombh2 so that the sampler does not complain
    ombh2:
      prior:
        min: 0.005
        max: 0.1
      ref: 0.02225203
    omch2: 0.1198657
    # only one of the next two is finally used!
    H0: 67.25
    cosmomc_theta: 0.01040778
    As: 2.204051e-9
    ns: 0.9647522
    tau: 0.07888604
    # Derived
  # Planck likelihood
  A_planck: 1.00029
  A_cib_217: 66.4
  xi_sz_cib: 0.13
  A_sz:
    prior:
      dist: uniform
      min: 0
      max: 10
    ref: 7.17
  ps_A_100_100: 255.0
  ps_A_143_143: 40.1
  ps_A_143_217: 36.4
  ps_A_217_217: 98.7
  ksz_norm:
    prior:
      dist: uniform
      min: 0
      max: 10
    ref: 0.00
  gal545_A_100: 7.34
  gal545_A_143: 8.97
  gal545_A_143_217: 17.56
  gal545_A_217: 81.9
  galf_EE_A_100: 0.0813
  galf_EE_A_100_143: 0.0488
  galf_EE_A_100_217: 0.0995
  galf_EE_A_143: 0.1002
  galf_EE_A_143_217: 0.2236
  galf_EE_A_217: 0.645
  galf_TE_A_100: 0.1417
  galf_TE_A_100_143: 0.1321
  galf_TE_A_100_217: 0.307
  galf_TE_A_143: 0.155
  galf_TE_A_143_217: 0.338
  galf_TE_A_217: 1.667
  calib_100T: 0.99818
  calib_217T: 0.99598
"""%(_params, _theory)

derived_lowTEB_highTTTEEE = {
    # param: [best_fit, sigma]
    "H0": [67.25, 0.66],
    "omegav": [0.6844, 0.0091],
    "omegam": [0.3156, 0.0091],
    "omegamh2": [0.14276, 0.0014],
    "omegamh3": [0.096013, 0.00029],
    "sigma8": [0.8310, 0.013],
    "s8omegamp5": [0.4669, 0.0098],
    "s8omegamp25":[0.6228, 0.011],
    "s8h5":       [1.0133, 0.017],
    "zre":   [10.07, 1.6],
    "As1e9":  [2.204, 2.207],
    "clamp":  [1.8824, 0.012],
    "YHe":    [0.2453409, 0.000072],
    "Y_p":    [0.2466672, 0.000072],
    "DH":     [2.6136e-5, 0.030e-5],
    "age":    [13.8133, 0.026],
    "zstar":  [1090.057, 0.30],
    "rstar":  [144.556, 0.32],
    "thetastar": [1.040967, 0.00032],
    "DAstar":  [13.8867, 0.030],
    "zdrag":   [1059.666, 0.31],
    "rdrag":   [147.257, 0.31],
    "kd":      [0.140600, 0.00032],
    "thetad":  [0.160904, 0.00018],
    "zeq":     [3396.2, 33],
    "keq":     [0.010365, 0.00010],
    "thetaeq":  [0.8139, 0.0063],
    "thetarseq": [0.44980, 0.0032],
    }
