from __future__ import division, print_function
from collections import OrderedDict as odict

from cobaya.conventions import _theory, _params, _likelihood, _sampler
from cobaya.conventions import _prior, _p_ref, _p_proposal, _p_label, _p_dist, _p_drop

_camb = "camb"
_classy = "classy"
_desc = "desc"
_extra_args = "extra_args"

# Theory codes
theory = odict([[_camb, None], [_classy, None]])

# Primordial perturbations
primordial = odict([
    ["SFSR", {
        _desc: "Vanilla Single-field Slow-roll Inflation (no tensors)",
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["logA", {_prior: {"min": 2, "max": 4},
                      _p_ref: {_p_dist: "norm", "loc": 3.1, "scale": 0.001},
                      _p_proposal: 0.001, _p_label: r"\log(10^{10} A_s",
                      _p_drop: True}],
            ["As", "lambda logA: 1e-10*np.exp(logA)"],
            ["ns", {_prior: {"min": 0.8, "max": 1.2},
                    _p_ref: {_p_dist: "norm", "loc": 0.96, "scale": 0.004},
                    _p_proposal: 0.002, _p_label: r"n_s"}]])}]])
primordial.update(odict([
    ["SFSR_run", {
        _desc: "Vanilla Single-field Slow-roll Inflation w running (no tensors)",
        _theory: {_camb: None, _classy: None},
        _params: odict(
            (list(primordial["SFSR"][_params].items()) +
             [["nrun", {_prior: {"min": -1, "max": 1},
                        _p_ref: {_p_dist: "norm", "loc": 0, "scale": 0.005},
                        _p_proposal: 0.001, _p_label: r"n_\mathrm{run}"}]]))}],
    ["SFSR_t", {
        _desc: "Vanilla Single-field Slow-roll Inflation w tensors",
        _theory: {_camb: None, _classy: None},
        _params: odict(
            (list(primordial["SFSR"][_params].items()) +
             [["r", {_prior: {"min": 0, "max": 3},
                     _p_ref: {_p_dist: "norm", "loc": 0, "scale": 0.03},
                     _p_proposal: 0.03, _p_label: r"r_{0.05}"}]]))}]]))

# Geometry
geometry = odict([
    ["flat", {
        _desc: "Flat FLRW universe",
        _theory: {_camb: None, _classy: None}}],
    ["omegak", {
        _desc: "FLRW model with varying curvature (prior on Omega_k)",
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["omegak", {_prior: {"min": -0.3, "max": 0.3},
                        _p_ref: {_p_dist: "norm", "loc": -0.0008, "scale": 0.001},
                        _p_proposal: 0.001, _p_label: r"\Omega_k"}]])}],])

# Hubble parameter constraints
hubble = odict([
    ["H", {
        _desc: "Hubble parameter",
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["H0", {
                _prior: {"min": 40, "max": 100},
                _p_ref: {_p_dist: "norm", "loc": 70, "scale": 2},
                _p_proposal: 2, _p_label: r"H_0"}]])}],
    ["cosmomc_theta", {
        _desc: "CosmoMC's approx. angular size of sound horizon (CAMB only)",
        _theory: {_camb: None},
        _params: odict([
            ["cosmomc_theta", "lambda theta: 1.e-2*theta"],
            ["theta", {
                _prior: {"min": 0.5, "max": 10}, _p_drop: True,
                _p_ref: {_p_dist: "norm", "loc": 1.0411, "scale": 0.0004},
                _p_proposal: 0.0002, _p_label: r"100\theta_\mathrm{MC}"}],
            ["H0", {"latex": r"H_0"}]])}],
    ["theta_s", {
        _desc: "Angular size of sound horizon (CLASS only)",
        _theory: {_classy: None},
        _params: odict([
            ["100*theta_s", "lambda theta_s_100: theta_s_100"],
            ["theta_s_100", {
                _prior: {"min": 0.5, "max": 10}, _p_drop: True,
                _p_ref: {_p_dist: "norm", "loc": 1.0418, "scale": 0.0004},
                _p_proposal: 0.0002, _p_label: r"100\theta_s"}],
            ["H0", {"latex": r"H_0"}]])}]])

# Baryons
baryons = odict([
    ["omegab_h2", {
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["ombh2", {
                _prior: {"min": 0.005, "max": 0.1},
                _p_ref: {_p_dist: "norm", "loc": 0.0221, "scale": 0.0001},
                _p_proposal: 0.0001, _p_label: r"\Omega_\mathrm{b} h^2"}]])}]])

# Dark matter
dark_matter = odict([
    ["omegac_h2", {
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["omch2", {
                _prior: {"min": 0.001, "max": 0.99},
                _p_ref: {_p_dist: "norm", "loc": 0.12, "scale": 0.001},
                _p_proposal: 0.0005, _p_label: r"\Omega_\mathrm{c} h^2"}]])}]])

# Dark Energy
dark_energy = odict([
    ["lambda", {
        _desc: "Cosmological constant (w=-1)",
        _theory: {_camb: None, _classy: None},
        _params: {"omegav": {"latex": r"\Omega_\Lambda"}}}],
    ["de_w", {
        _desc: "Varying constant eq of state",
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["w", {
                _prior: {"min": -3, "max": 1},
                _p_ref: {_p_dist: "norm", "loc": -0.99, "scale": 0.02},
                _p_proposal: 0.02, _p_label: r"w_\Lambda"}]])}]])
# w_a only for CAMB devel?
# - w+wa -- params['wa'] = '0 -3 2 0.05 0.05'


# Neutrinos and other extra matter
neutrinos = odict([
    ["one_heavy_planck", {
        _desc: "Two massless nu and one with m=0.06. Neff=3.046",
        _theory: {
            _camb: {_extra_args:
                {"num_massive_neutrinos": 1, "mnu": 0.06, "nnu": 3.046}},
            _classy: {_extra_args:
                {"N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06, "T_ncdm": 0.71611}}}}],
    ["varying_mnu", {
        _desc: "Varying m_nu of 3 degenerate nu's, with N_eff=3.046",
        _theory: {
            _camb: {_extra_args:
                {"num_massive_neutrinos": 1, "nnu": 3.046}}},
        _params: odict([
            ["mnu", {
                _prior: {"min": 0, "max": 5},
                _p_ref: {_p_dist: "norm", "loc": 0.02, "scale": 0.1},
                _p_proposal: 0.03, _p_label: r"m_\nu"}]])}],
    ["varying_Neff", {
        _desc: "Varying Neff with two massless nu and one with m=0.06",
        _theory: {
            _camb: {_extra_args:
                {"num_massive_neutrinos": 1, "mnu": 0.06}}},
        _params: odict([
            ["nnu", {
                _prior: {"min": 0.05, "max": 10},
                _p_ref: {_p_dist: "norm", "loc": 3.046, "scale": 0.05},
                _p_proposal: 0.05, _p_label: r"N_\mathrm{eff}"}]])}],
    ["varying_Neff+1sterile", {
        _desc: "Varying Neff plus 1 sterile neutrino (SM nu's with m=0,0,0.06)",
        _theory: {
            _camb: {_extra_args:
                {"num_massive_neutrinos": 1, "mnu": 0.06, "accuracy_level": 1.2}}},
        _params: odict([
            ["nnu", {
                _prior: {"min": 3.046, "max": 10},
                _p_ref: {_p_dist: "norm", "loc": 3.046, "scale": 0.05},
                _p_proposal: 0.05, _p_label: r"N_\mathrm{eff}"}],
            ["meffsterile", {
                _prior: {"min": 0, "max": 3},
                _p_ref: {_p_dist: "norm", "loc": 0.1, "scale": 0.1},
                _p_proposal: 0.03,
                _p_label: r"m_{\nu,\mathrm{sterile}}}^{\mathrm{eff}}"}]])}],])

# BBN
bbn = odict([
    ["consistency", {
        _desc: "Primordial Helium fraction inferred from BBN consistency",
        _theory: {_camb: None, _classy: None},
        _params: {"YHe": {"latex": r"Y_P"}}}],
    ["YHe", {
        _desc: "Varying primordial Helium fraction",
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["YHe", {
                _prior: {"min": 0.1, "max": 0.5},
                _p_ref: {_p_dist: "norm", "loc": 0.245, "scale": 0.006},
                _p_proposal: 0.006, _p_label: r"y_\mathrm{He}"}]])}],])

# Reionization
reionization = odict([
    ["std", {
        _desc: "Standard reio, lasting delta_z=0.5",
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["tau", {
                _prior: {"min": 0.01, "max": 0.8},
                _p_ref: {_p_dist: "norm", "loc": 0.09, "scale": 0.01},
                _p_proposal: 0.005, _p_label: r"\tau_\mathrm{reio}"}],
            ["zre", {"latex": r"z_\mathrm{re}"}]])}],
    ["gauss_prior", {
        _desc: "Standard reio, lasting delta_z=0.5, gaussian prior around tau=0.07",
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["tau", {
                _prior: {_p_dist: "norm", "loc": 0.07, "scale": 0.02},
                _p_ref: {_p_dist: "norm", "loc": 0.07, "scale": 0.01},
                _p_proposal: 0.005, _p_label: r"\tau_\mathrm{reio}"}],
            ["zre", {"latex": r"z_\mathrm{re}"}]])}],])

# CMB lensing
cmb_lensing = odict([
    ["consistency", {
        _desc: "Standard CMB lensing",
        _theory: {_camb: None, _classy: None}}],
    ["ALens", {
        _desc: "Varying CMB lensing potential (scaled by sqrt(ALens)",
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["ALens", {
                _prior: {"min": 0, "max": 10},
                _p_ref: {_p_dist: "norm", "loc": 1, "scale": 0.05},
                _p_proposal: 0.05, _p_label: r"A_\mathrm{L}"}]])}],])

# EXPERIMENTS ############################################################################
cmb = odict([
    ["planck_2015_lensing", {
        _desc: "",
        _likelihood: odict([
            ["planck_2015_lowTEB", None],
            ["planck_2015_plikHM_TTTEEE", None],
            ["planck_2015_lensing", None]])}],
    ["planck_2015_lensing_bkp", {
        _desc: "",
        _likelihood: odict([
            ["planck_2015_lowTEB", None],
            ["planck_2015_plikHM_TTTEEE", None],
            ["planck_2015_lensing", None],
            ["bicep_keck_2015", None]])}],
])

# SAMPLERS ###############################################################################
sampler = odict([
    ["MCMC", {
        _desc: "MCMC sampler with covmat learning, and fast dragging.",
        _sampler: {"mcmc": {"drag": True, "learn_proposal": True}}}],
    ["PolyChord", {
        _desc: "Nested sampler, affine invariant and multi-modal.",
        _sampler: {"polychord": None}}],])

# DERIVED ################################################################################

derived = {
    "null": {_theory: {_camb: None, _classy: None},},
    "planck": {  # just the safe (re CLASS) ones
        _theory: {_camb: None, _classy: None},
        _params: odict([
        ["omegam", {"latex": r"\Omega_m"}],
        ["omegamh2",
         {"derived": "lambda omegam, H0: omegam*(H0/100)**2", "latex": r"\Omega_m h^2"}],
        ["omegamh3",
         {"derived": "lambda omegam, H0: omegam*(H0/100)**3", "latex": r"\Omega_m h^3"}],
        ["sigma8", {"latex": r"\sigma_8"}],
        ["s8h5",
         {"derived": "lambda sigma8, H0: sigma8*(H0*1e-2)**(-0.5)",
          "latex": r"\sigma_8/h^{0.5}"}],
        ["s8omegamp5",
         {"derived": "lambda sigma8, omegam: sigma8*omegam**0.5",
          "latex": r"\sigma_8 \Omega_m^{0.5}"}],
        ["s8omegamp25",
         {"derived": "lambda sigma8, omegam: sigma8*omegam**0.25",
          "latex": r"\sigma_8 \Omega_m^{0.25}"}],
        ["A", {"derived": "lambda As: 1e9*As", "latex": r"10^9 A_s"}],
        ["clamp",
         {"derived": "lambda As, tau: 1e9*As*np.exp(-2*tau)",
          "latex": r"10^9 A_s e^{-2\tau}"}],
        ["age", {"latex": r"{\rm{Age}}/\mathrm{Gyr}"}],
        ["rdrag", {"latex": r"r_\mathrm{drag}"}]])}}

# PRESETS ################################################################################
preset = odict([
    ["planck_2015_lensing_camb", {
        _desc: "Planck 2015 (Polarised CMB + lensing) with CAMB",
        "theory": "camb",
        "primordial": "SFSR",
        "geometry": "flat",
        "hubble": "cosmomc_theta",
        "baryons": "omegab_h2",
        "dark_matter": "omegac_h2",
        "dark_energy": "lambda",
        "neutrinos": "one_heavy_planck",
        "bbn": "consistency",
        "reionization": "std",
        "cmb_lensing": "consistency",
        "cmb": "planck_2015_lensing",
        "sampler": "MCMC",
        "derived": "planck"}],
    ["planck_2015_lensing_classy", {
        _desc: "Planck 2015 (Polarised CMB + lensing) with CLASS",
        "theory": "classy",
        "primordial": "SFSR",
        "geometry": "flat",
        "hubble": "theta_s",
        "baryons": "omegab_h2",
        "dark_matter": "omegac_h2",
        "dark_energy": "lambda",
        "neutrinos": "one_heavy_planck",
        "bbn": "consistency",
        "reionization": "std",
        "cmb_lensing": "consistency",
        "cmb": "planck_2015_lensing",
        "sampler": "MCMC",
        "derived": "planck"}],
    ["planck_2015_lensing_bicep_camb", {
        _desc: "Planck 2015 + lensing + BKP with CAMB",
        "theory": "camb",
        "primordial": "SFSR_t",
        "geometry": "flat",
        "hubble": "cosmomc_theta",
        "baryons": "omegab_h2",
        "dark_matter": "omegac_h2",
        "dark_energy": "lambda",
        "neutrinos": "one_heavy_planck",
        "bbn": "consistency",
        "reionization": "std",
        "cmb_lensing": "consistency",
        "cmb": "planck_2015_lensing_bkp",
        "sampler": "MCMC",
        "derived": "planck"}],
    ])
