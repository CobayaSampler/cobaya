from __future__ import division
import numpy as np
import os
from collections import OrderedDict as odict

from cobaya.conventions import input_theory, input_likelihood, input_params
from cobaya.conventions import input_sampler, _chi2, separator

# Expected values and tolerance
tolerance_abs = 0.1

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
"""%(input_params, input_theory)

derived = {
    "H0":          {"latex": r"H_0"},
    # was "omegal"
    "omegav":      {"latex": r"\Omega_\Lambda"},
#    "omegam":      {"latex": r"\Omega_m"},
#    "omegamh2":    {"latex": r"\Omega_m h^2"},
#    "omeganuh2":   {"latex": r"\Omega_\nu h^2"},
#    "omegamh3":    {"latex": r"\Omega_m h^3"},
#    "sigma8":      {"latex": r"\sigma_8"},
#    "s8omegamp5":  {"latex": r"\sigma_8 \Omega_m^{0.5}"},
#    "s8omegamp25": {"latex": r"\sigma_8 \Omega_m^{0.25}"},
#    "s8h5":        {"latex": r"\sigma_8/h^{0.5}"},
#    "rmsdeflect":  {"latex": r"\langle d^2\rangle^{1/2}"},
    # was "zrei"
    # "Reion/redshift":        {"latex": r"z_{\rm re}"},
#    "A":           {"latex": r"10^9 A_s"},
#    "clamp":       {"latex": r"10^9 A_s e^{-2\tau}"},
#    "DL40":        {"latex": r"D_{40}"},
#    "DL220":       {"latex": r"D_{220}"},
#    "DL810":       {"latex": r"D_{810}"},
#    "DL1420":      {"latex": r"D_{1420}"},
#    "DL2000":      {"latex": r"D_{2000}"},
#    "ns02":        {"latex": r"n_{s,0.002}"},
#    "yheused":     {"latex": r"Y_P"},
#    "YpBBN":       {"latex": r"Y_P^{\rm{BBN}}"},
#    "DHBBN":       {"latex": r"10^5D/H"},
#    "age":         {"latex": r"{\rm{Age}}/{\rm{Gyr}}"},
#    "zstar":       {"latex": r"z_*"},
#    "rstar":       {"latex": r"r_*"},
#    "thetastar":   {"latex": r"100\theta_*"},
#    "DAstar":      {"latex": r"D_{\rm{A}}/{\rm{Gpc}}"},
#    "zdrag":       {"latex": r"z_{\rm{drag}}"},
#    "rdrag":       {"latex": r"r_{\rm{drag}}"},
#    "kd":          {"latex": r"k_{\rm D}"},
#    "thetad":      {"latex": r"100\theta_{\rm{D}}"},
#    "zeq":         {"latex": r"z_{\rm{eq}}"},
#    "keq":         {"latex": r"k_{\rm{eq}}"},
#    "thetaeq":     {"latex": r"100\theta_{\rm{eq}}"},
#    "thetarseq":   {"latex": r"100\theta_{\rm{s,eq}}"},
#    "rsDv057":     {"latex": r"r_{\rm{drag}}/D_V(0.57)"},
#    "Hubble057":   {"latex": r"H(0.57)"},
#    "DA057":       {"latex": r"D_A(0.57)"},
#    "FAP057":      {"latex": r"F_{\rm AP}(0.57)"},
#    "fsigma8z057": {"latex": r"f\sigma_8(0.57)"},
#    "sigma8z057":  {"latex": r"\sigma_8(0.57)"},
#    "f2000_143":   {"latex": r"f_{2000}^{143}"},
#    "f2000_x":     {"latex": r"f_{2000}^{143\times217}"},
#    "f2000_217":   {"latex": r"f_{2000}^{217}"}
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
"""%(input_params, input_theory)

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
"""%(input_params, input_theory)
