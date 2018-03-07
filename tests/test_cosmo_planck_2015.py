# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CAMB and CLASS

from cobaya.yaml import yaml_load

from cosmo_common import body_of_test


def test_planck_2015_t_camb(modules):
    info_best_fit = yaml_load(params_lowl_highTT)
    info_likelihood = lik_info_lowl_highTT
    info_theory = {"camb": None}
    derived_values = derived_lowl_highTT
    body_of_test(modules, info_best_fit, info_likelihood, info_theory,
                 chi2_lowl_highTT, derived_values)


def test_planck_2015_p_camb(modules):
    info_best_fit = yaml_load(params_lowTEB_highTTTEEE)
    info_likelihood = lik_info_lowTEB_highTTTEEE
    info_theory = {"camb": None}
    derived_values = derived_lowTEB_highTTTEEE
    body_of_test(modules, info_best_fit, info_likelihood, info_theory,
                 chi2_lowTEB_highTTTEEE, derived_values)


def test_camb_planck_p(modules):
    body_of_test(modules, "p", "camb")


def test_classy_planck_t(modules):

###        tolerance = tolerance_chi2_abs + (2.1 if theory == "classy" else 0)
    
    body_of_test(modules, "t", "classy")


def test_classy_planck_p(modules):
    body_of_test(modules, "p", "classy")


## QUITAR EL DUMMY PRIOR!!!!!

    

# Temperature only #######################################################################

# NB: A_sz and ksz_norm need to have a prior defined, though "evaluate" will ignore it in
# favour of the fixed "ref" value. This needs to be so since, at the time of writing this
# test, explicit multiparameter priors need to be functions of *sampled* parameters.
# Hopefully this requirement can be dropped in the near future.

lik_info_lowl_highTT = {"planck_2015_lowl": None, "planck_2015_plikHM_TT": None}

chi2_lowl_highTT = {"planck_2015_lowl": 15.39,
                    "planck_2015_plikHM_TT": 761.09,
                    "tolerance": 0.1}

params_lowl_highTT = """
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
"""

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


# Best fit Polarization ##################################################################

# NB: A_sz and ksz_norm need to have a prior defined, though "evaluate" will ignore it in
# favour of the fixed "ref" value. This needs to be so since, at the time of writing this
# test, explicit multiparameter priors need to be functions of *sampled* parameters.
# Hopefully this requirement can be dropped in the near future.

lik_info_lowTEB_highTTTEEE = {"planck_2015_lowTEB": {"speed": 0.25},
                              "planck_2015_plikHM_TTTEEE": None}

chi2_lowTEB_highTTTEEE = {"planck_2015_lowTEB": 10496.93,
                          "planck_2015_plikHM_TTTEEE": 2431.65,
                          "tolerance": 0.1}

params_lowTEB_highTTTEEE = """
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
"""

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
