# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CAMB and CLASS
from copy import deepcopy

from .common_cosmo import body_of_test
from cobaya.cosmo_input import cmb_precision

# Generating plots in Travis
try:
    import matplotlib

    matplotlib.use('agg')
except:
    pass

# Derived parameters not understood by CLASS
# https://wiki.cosmos.esa.int/planckpla2015/images/b/b9/Parameter_tag_definitions_2015.pdf
classy_unknown = ["zstar", "rstar", "thetastar", "DAstar", "zdrag",
                  "kd", "thetad", "zeq", "keq", "thetaeq", "thetarseq",
                  "DH", "Y_p"]

# Small chi2 difference with CLASS (total still <0.5)
classy_extra_tolerance = 0.4


# STANDARD ###############################################################################

def test_planck_2018_t_camb(packages_path, skip_not_installed):
    best_fit = deepcopy(params_lowl_highTT_lensing)
    best_fit.pop("H0")
    info_likelihood = lik_info_lowl_highTT_lensing
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    best_fit_derived = derived_lowl_highTT_lensing
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_lowl_highTT_lensing, best_fit_derived,
                 skip_not_installed=skip_not_installed)


def test_planck_2018_p_camb(packages_path, skip_not_installed):
    best_fit = deepcopy(params_lowTE_highTTTEEE_lensingcmblikes)
    best_fit.pop("H0")
    info_likelihood = lik_info_lowTE_highTTTEEE_lensingcmblikes.copy()
    chi2 = chi2_lowTE_highTTTEEE_lensingcmblikes.copy()
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    best_fit_derived = derived_lowTE_highTTTEEE_lensingcmblikes
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, chi2,
                 best_fit_derived, skip_not_installed=skip_not_installed)


# LITES ##################################################################################

def test_planck_2018_t_lite_camb(packages_path, skip_not_installed, native=False):
    best_fit = deepcopy(params_lowl_highTT_lite_lensing)
    best_fit.pop("H0")
    like_name = "planck_2018_highl_plik.TT_lite" + ("_native" if native else "")
    info_likelihood = {like_name: None}
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    chi2 = {like_name: chi2_planck_2018_plikHM_highTT_lite, "tolerance": 0.01}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, chi2,
                 skip_not_installed=skip_not_installed)


def test_planck_2018_t_lite_native_camb(packages_path, skip_not_installed):
    test_planck_2018_t_lite_camb(packages_path, native=True,
                                 skip_not_installed=skip_not_installed)


def test_planck_2018_p_lite_camb(packages_path, skip_not_installed, native=False):
    best_fit = deepcopy(params_lowTE_highTTTEEE_lite_lensingcmblikes)
    best_fit.pop("H0")
    like_name = "planck_2018_highl_plik.TTTEEE_lite" + ("_native" if native else "")
    info_likelihood = {like_name: None}
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    chi2 = {like_name: chi2_planck_2018_plikHM_highTTTEEE_lite, "tolerance": 0.01}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, chi2,
                 skip_not_installed=skip_not_installed)


def test_planck_2018_p_lite_native_camb(packages_path, skip_not_installed):
    test_planck_2018_p_lite_camb(packages_path, native=True,
                                 skip_not_installed=skip_not_installed)


# UNBINNED ###############################################################################

def test_planck_2018_t_unbinned_camb(packages_path, skip_not_installed, native=False):
    best_fit = deepcopy(params_lowl_highTT_lensing)
    best_fit.pop("H0")
    like_name = "planck_2018_highl_plik.TT_unbinned"
    info_likelihood = {like_name: None}
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    chi2 = {like_name: 8275.93, "tolerance": 0.01}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, chi2,
                 skip_not_installed=skip_not_installed)


def test_planck_2018_p_unbinned_camb(packages_path, skip_not_installed, native=False):
    best_fit = deepcopy(params_lowTE_highTTTEEE_lensingcmblikes)
    best_fit.pop("H0")
    like_name = "planck_2018_highl_plik.TTTEEE_unbinned"
    info_likelihood = {like_name: None}
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    chi2 = {like_name: 24125.92, "tolerance": 0.01}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, chi2,
                 skip_not_installed=skip_not_installed)


# CamSpec ################################################################################

def test_planck_2018_t_CamSpec_native_camb(packages_path, skip_not_installed, plik=False):
    # TODO: sort out calPlacnk vs A_planck
    name = "planck_2018_highl_CamSpec.TT" if plik else "planck_2018_highl_CamSpec.TT_native"
    info_likelihood = {name: None}
    chi2 = {name: 7060.04, 'tolerance': 0.2}
    best_fit = params_lowTE_highTTTEEE_lite_lensingcmblikes.copy()
    best_fit['calPlanck'] = best_fit['A_planck']
    best_fit.update(
        {'aps100': 238.7887, 'aps143': 41.31762, 'aps217': 100.6226, 'acib217': 44.96003, 'asz143': 5.886124,
         'psr': 0.5820399, 'cibr': 0.7912195, 'ncib': 0.0, 'cibrun': 0.0, 'xi': 0.1248677, 'aksz': 1.153473,
         'dust100': 1.010905, 'dust143': 0.9905765, 'dust217': 0.9658913, 'dust143x217': 0.9946434,
         'cal0': 0.9975484, 'cal2': 1.00139, 'calTE': 1.0, 'calEE': 1.0})
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, chi2,
                 skip_not_installed=skip_not_installed)


def test_planck_2018_t_CamSpec_clik_camb(packages_path, skip_not_installed):
    test_planck_2018_t_CamSpec_native_camb(packages_path, plik=True,
                                           skip_not_installed=skip_not_installed)


def test_planck_2018_p_CamSpec_native_camb(packages_path, skip_not_installed, plik=False):
    # TODO: sort out calPlacnk vs A_planck
    name = "planck_2018_highl_CamSpec.TTTEEE" if plik else "planck_2018_highl_CamSpec.TTTEEE_native"
    info_likelihood = {name: None}
    chi2 = {name: 11513.53, 'tolerance': 0.2}
    best_fit = params_lowTE_highTTTEEE_lite_lensingcmblikes.copy()
    if plik:
        best_fit['calPlanck'] = best_fit.pop('A_planck')
    best_fit.update(
        {'aps100': 238.7887, 'aps143': 41.31762, 'aps217': 100.6226, 'acib217': 44.96003, 'asz143': 5.886124,
         'psr': 0.5820399, 'cibr': 0.7912195, 'ncib': 0.0, 'cibrun': 0.0, 'xi': 0.1248677, 'aksz': 1.153473,
         'dust100': 1.010905, 'dust143': 0.9905765, 'dust217': 0.9658913, 'dust143x217': 0.9946434,
         'cal0': 0.9975484, 'cal2': 1.00139, 'calTE': 1.0, 'calEE': 1.0})
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory, chi2,
                 skip_not_installed=skip_not_installed)


def test_planck_2018_p_CamSpec_clik_camb(packages_path, skip_not_installed):
    test_planck_2018_p_CamSpec_native_camb(packages_path, plik=True,
                                           skip_not_installed=skip_not_installed)


# CMB-marged lensing #####################################################################

def test_planck_2018_lcmbmarged_camb(packages_path, skip_not_installed):
    best_fit = params_lensing_cmbmarged
    info_likelihood = lik_info_lensing_cmbmarged
    info_theory = {"camb": {"extra_args": cmb_precision["camb"]}}
    best_fit_derived = {}
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_lensing_cmbmarged, best_fit_derived,
                 skip_not_installed=skip_not_installed)


# with CLASS #############################################################################

def test_planck_2018_t_classy(packages_path, skip_not_installed):
    best_fit = deepcopy(params_lowl_highTT_lensing)
    best_fit.pop("theta_MC_100")
    best_fit = params_lowl_highTT_lensing
    info_likelihood = lik_info_lowl_highTT_lensing
    info_theory = {"classy": {"extra_args": cmb_precision["classy"]}}
    best_fit_derived = deepcopy(derived_lowl_highTT_lensing)
    for p in classy_unknown:
        best_fit_derived.pop(p, None)
    chi2_lowl_highTT_classy = deepcopy(chi2_lowl_highTT_lensing)
    chi2_lowl_highTT_classy["tolerance"] += classy_extra_tolerance
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_lowl_highTT_classy, best_fit_derived,
                 skip_not_installed=skip_not_installed)


def test_planck_2018_p_classy(packages_path, skip_not_installed):
    best_fit = deepcopy(params_lowTE_highTTTEEE_lensingcmblikes)
    best_fit.pop("theta_MC_100")
    info_likelihood = lik_info_lowTE_highTTTEEE_lensingcmblikes
    info_theory = {"classy": {"extra_args": cmb_precision["classy"]}}
    best_fit_derived = deepcopy(derived_lowTE_highTTTEEE_lensingcmblikes)
    for p in classy_unknown:
        best_fit_derived.pop(p, None)
    chi2_lowl_highTT_classy = deepcopy(chi2_lowTE_highTTTEEE_lensingcmblikes)
    chi2_lowl_highTT_classy["tolerance"] += classy_extra_tolerance
    body_of_test(packages_path, best_fit, info_likelihood, info_theory,
                 chi2_lowl_highTT_classy, best_fit_derived,
                 skip_not_installed=skip_not_installed)


# Best fit temperature only ##############################################################

lik_info_lowl_highTT_lensing = {
    "planck_2018_lowl.TT": None, "planck_2018_highl_plik.TT": None,
    "planck_2018_lensing.clik": None}

chi2_lowl_highTT_lensing = {
    "planck_2018_lowl.TT": 22.92,
    "planck_2018_highl_plik.TT": 757.77,
    "planck_2018_lensing.clik": 9.11,
    "tolerance": 0.11}

chi2_planck_2018_plikHM_highTT_lite = 204.45

params_lowl_highTT_lite_lensing = {
    # Sampled
    "omegabh2": 0.02240,
    "omegach2": 0.1172,
    # only one of the next two is finally used!
    "H0": 68.45,  # will be ignored in the CAMB case
    "theta_MC_100": 1.04117,  # will be ignored in the CLASS case
    "tau": 0.0862,
    "logA": 3.100,
    "ns": 0.9733,
    # Planck likelihood
    "A_planck": 1.00008}

params_lowl_highTT_lensing = params_lowl_highTT_lite_lensing.copy()
params_lowl_highTT_lensing.update({
    "A_cib_217": 45.6,
    "xi_sz_cib": 0.73,
    "A_sz": 6.92,
    "ps_A_100_100": 246,
    "ps_A_143_143": 51.3,
    "ps_A_143_217": 54.7,
    "ps_A_217_217": 122.1,
    "ksz_norm": 0.01,
    "gal545_A_100": 8.87,
    "gal545_A_143": 10.79,
    "gal545_A_143_217": 19.8,
    "gal545_A_217": 95.4,
    "calib_100T": 0.99965,
    "calib_217T": 0.99822})

derived_lowl_highTT_lensing = {
    # param: [best_fit, sigma]
    "H0": [params_lowl_highTT_lensing["H0"], 1.2],
    "omegal": [0.7006, 0.015],
    "omegam": [0.2994, 0.016],
    "sigma8": [0.8271, 0.013],
    "zrei": [10.61, 2.2],
    "YHe": [0.245408, 0.0001],
    "Y_p": [0.246735, 0.0001],
    "DH": [2.5794e-5, 0.05e-5],
    "age": [13.7744, 0.047],
    "zstar": [1089.63, 0.52],
    "rstar": [145.13, 0.55],
    "thetastar": [1.04136, 0.00050],
    "DAstar": [13.9364, 0.050],
    "zdrag": [1059.818, 0.51],
    "rdrag": [147.79, 0.53],
    "kd": [0.14015, 0.00053],
    "thetad": [0.160844, 0.00029],
    "zeq": [3337, 57],
    "keq": [0.010184, 0.00017],
    "thetaeq": [0.8255, 0.011],
    "thetarseq": [0.4557, 0.0057]}

# Best fit polarization ##################################################################

lik_info_lowTE_highTTTEEE_lensingcmblikes = {
    "planck_2018_lowl.TT": None, "planck_2018_lowl.EE": None,
    "planck_2018_highl_plik.TTTEEE": None, "planck_2018_lensing.native": None}

chi2_lowTE_highTTTEEE_lensingcmblikes = {
    "planck_2018_lowl.TT": 23.25, "planck_2018_lowl.EE": 396.05,
    "planck_2018_highl_plik.TTTEEE": 2344.93, "planck_2018_lensing.native": 8.87,
    "tolerance": 0.11}

chi2_planck_2018_plikHM_highTTTEEE_lite = 584.65

params_lowTE_highTTTEEE_lite_lensingcmblikes = {
    # Sampled
    "omegabh2": 0.022383,
    "omegach2": 0.12011,
    # only one of the next two is finally used!
    "H0": 67.32,  # will be ignored in the CAMB case
    "theta_MC_100": 1.040909,  # will be ignored in the CLASS case
    "logA": 3.0448,
    "ns": 0.96605,
    "tau": 0.0543,
    # Planck likelihood
    "A_planck": 1.00044}

params_lowTE_highTTTEEE_lensingcmblikes = params_lowTE_highTTTEEE_lite_lensingcmblikes.copy()
params_lowTE_highTTTEEE_lensingcmblikes.update(
    {"A_cib_217": 46.1,
     "xi_sz_cib": 0.66,
     "A_sz": 7.08,
     "ps_A_100_100": 248.2,
     "ps_A_143_143": 50.7,
     "ps_A_143_217": 53.3,
     "ps_A_217_217": 121.9,
     "ksz_norm": 0.00,
     "gal545_A_100": 8.80,
     "gal545_A_143": 11.01,
     "gal545_A_143_217": 20.16,
     "gal545_A_217": 95.5,
     "galf_TE_A_100": 0.1138,
     "galf_TE_A_100_143": 0.1346,
     "galf_TE_A_100_217": 0.479,
     "galf_TE_A_143": 0.225,
     "galf_TE_A_143_217": 0.665,
     "galf_TE_A_217": 2.082,
     "calib_100T": 0.99974,
     "calib_217T": 0.99819})

derived_lowTE_highTTTEEE_lensingcmblikes = {
    # param: [best_fit, sigma]
    "H0": [params_lowTE_highTTTEEE_lensingcmblikes["H0"], 0.54],
    "omegal": [0.6842, 0.0073],
    "omegam": [0.3158, 0.0073],
    "sigma8": [0.8120, 0.0060],
    "zrei": [7.68, 7.67],
    "YHe": [0.245401, 0.000057],
    "Y_p": [0.246727, 0.000058],
    "DH": [2.5831e-5, 0.027e-5],
    "age": [13.7971, 0.023],
    "zstar": [1089.914, 0.25],
    "rstar": [144.394, 0.26],
    "thetastar": [1.041085, 0.00031],
    "DAstar": [13.8696, 0.025],
    "zdrag": [1059.971, 0.30],
    "rdrag": [147.049, 0.26],
    "kd": [0.140922, 0.00030],
    "thetad": [0.160734, 0.00017],
    "zeq": [3405.1, 26],
    "keq": [0.010393, 0.000081],
    "thetaeq": [0.81281, 0.0050],
    "thetarseq": [0.44912, 0.0026]}

# Best fit CMB-marged lensing ############################################################

lik_info_lensing_cmbmarged = {"planck_2018_lensing.CMBMarged": None}

chi2_lensing_cmbmarged = {
    "planck_2018_lensing.CMBMarged": 7.51, "tolerance": 0.11}

params_lensing_cmbmarged = {
    "omegabh2": 2.2219050E-02,
    "omegach2": 1.1726920E-01,
    "theta_MC_100": 1.1180650E+00,
    "tau": 0.055,
    "logA": 3.2528000E+00,
    "ns": 9.6135180E-01}
