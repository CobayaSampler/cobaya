from __future__ import division
import numpy as np
import os
from collections import OrderedDict as odict

from cobaya.conventions import _theory, _likelihood, _params
from cobaya.conventions import _sampler, _chi2, separator

# Tolerance for the tests
tolerance_chi2_abs = 0.1
tolerance_derived = 0.025
# this last one cannot be smaller: rounding errors in BBN abundances, bc new interp tables


# Converting 100cosmomc_theta to cosmomc_theta in Planck's covmats ########################

def convert_cosmomc_theta(filename):
    with open(filename, "r") as original:
        params = original.readline()[1:].split()
        i_theta = params.index("cosmomc_theta")
    covmat = np.loadtxt(filename)
    covmat[i_theta, :] /= 100
    covmat[:, i_theta] /= 100
    filename_new = "_0.01theta".join(os.path.splitext(filename))
    np.savetxt(filename_new, covmat, fmt="%.8g", header=" ".join(params))
    return filename_new


# Baseline priors #########################################################################

baseline_cosmology = r"""
%s:
  %s:
    # Sampled
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
    cosmomc_theta:
      prior:
        min: 0.005
        max: 0.1
      ref:
        dist: norm
        loc: 0.010411
        scale: 0.000004
      proposal: 0.000002
      latex: \theta_\mathrm{MC}
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
### Change A for log(1e10*A)! ---> param[logA] = 3.1 2 4 0.001 0.001
    As:
      prior:
        min: 0.74e-9
        max: 5.5e-9
      ref:
        dist: norm
        loc: 2.22e-9
        scale: 0.5e-9
      proposal: 0.5e-9
      latex: A_\mathrm{s}
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

# Derived parameters, described in
# https://wiki.cosmos.esa.int/planckpla2015/images/b/b9/Parameter_tag_definitions_2015.pdf
derived = {
    "H0":          {"latex": r"H_0"},
    "omegav":      {"latex": r"\Omega_\Lambda"},
#func    "omegam":      {"latex": r"\Omega_m"}, <-- b+cdm+nu
#func "omegamh2":    {"latex": r"\Omega_m h^2"},
#func    "omeganuh2":   {"latex": r"\Omega_\nu h^2"},
#func    "omegamh3":    {"latex": r"\Omega_m h^3"},
    "sigma8":      {"latex": r"\sigma_8"},
#func    "s8omegamp5":  {"latex": r"\sigma_8 \Omega_m^{0.5}"},
#func    "s8omegamp25": {"latex": r"\sigma_8 \Omega_m^{0.25}"},
#func    "s8h5":        {"latex": r"\sigma_8/h^{0.5}"},
#----------???    "rmsdeflect":  {"latex": r"\langle d^2\rangle^{1/2}"},
    "zre":         {"latex": r"z_\mathrm{re}"},
#func??    "A":           {"latex": r"10^9 A_s"},
#func    "clamp":       {"latex": r"10^9 A_s e^{-2\tau}"},
#func??    "ns02":        {"latex": r"n_{s,0.002}"}, <------ invert from analytic P(k)
    "YHe":         {"latex": r"Y_P"},
    "Y_p":       {"latex": r"Y_P^\mathrm{BBN}"},
    "DH":       {"latex": r"10^5D/H"},
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


# Best fit Temperature ####################################################################

# NB: A_sz and ksz_norm need to have a prior defined, though "evaluate" will ignore it in
# favour of the fixed "ref" value. This needs to be so since, at the time of writing this
# test, explicit multiparameter priors need to be functions of *sampled* parameters.
# Hopefully this requirement can be dropped in the near future.

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
      ref: 0.022491
    omch2: 0.11747
    # only one of the next two is finally used!
    H0: 68.44
    cosmomc_theta: 0.0104119
    tau: 0.1250
    As: 2.402e-9
    ns: 0.9742
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
    "H0":     [68.44, 1.2],
    "omegav": [0.6998, 0.016],
    "omegam":     [None, None], ### [0.3002, 0.016],
    "omegamh2":   [None, None],
    "omeganuh2":  [None, None],
    "omegamh3":   [None, None],
    "sigma8":     [0.8610, 0.023],
    "s8omegamp5": [None, None],
    "s8omegamp25":[None, None],
    "s8h5":       [None, None],
    "rmsdeflect": [None, None],
    "zre":   [13.76, 2.5],
    "A":      [None, None],
    "clamp":  [None, None],
    "ns02":   [None, None],
    "YHe":    [0.245446, 0.00012],
    "Y_p":  [0.246773, 0.00012],
    "DH":  [2.569e-5, 0.051e-5],
    "age":    [13.7664, 0.048],
    "zstar":  [1089.55, 0.52],
    "rstar":  [145.00, 0.55],
    "thetastar": [1.04136, 0.00051],
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


# Best fit Polarisation ###################################################################

# NB: A_sz and ksz_norm need to have a prior defined, though "evaluate" will ignore it in
# favour of the fixed "ref" value. This needs to be so since, at the time of writing this
# test, explicit multiparameter priors need to be functions of *sampled* parameters.
# Hopefully this requirement can be dropped in the near future.

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
      ref: 0.022252
    omch2: 0.11987
    # only one of the next two is finally used!
    H0: 67.25
    cosmomc_theta: 0.01040778
    As: 2.204e-9
    ns: 0.96475
    tau: 0.0789
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
