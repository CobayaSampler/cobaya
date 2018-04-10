from __future__ import division, print_function
from collections import OrderedDict as odict

from cobaya.conventions import _theory, _params, _likelihood, _sampler
from cobaya.conventions import _prior, _p_ref, _p_proposal, _p_label, _p_dist, _p_drop

_camb = "camb"
_classy = "classy"
_desc = "desc"

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
            ["As", "lambda logA: 1e-10*np.exp(logAs1e10)"],
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

#- r -- params['r'] = '0 0 3 0.03 0.03' --     'r': {'compute_tensors': True},


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
            ["cosmomc_theta", "lambda cosmomc_theta_100: 1.e-2*cosmomc_theta_100"],
            ["theta", {_p_drop: True,
                _prior: {"min": 0.5, "max": 10},
                _p_ref: {_p_dist: "norm", "loc": 1.0411, "scale": 0.0004},
                _p_proposal: 0.0002, _p_label: r"100\theta_\mathrm{MC}"}]])}],
    ["theta_s", {
        _desc: "Angular size of sound horizon (CLASS only)",
        _theory: {_classy: None},
        _params: odict([
            ["100*theta_s", "lambda theta_s_100: theta_s_100"],
            ["theta_s_100", {_p_drop: True,
                _prior: {"min": 0.5, "max": 10},
                _p_ref: {_p_dist: "norm", "loc": 1.0418, "scale": 0.0004},
                _p_proposal: 0.0002, _p_label: r"100\theta_s"}]])}]])
#param[theta] = 1.0411 0.5 10 0.0004 0.0002

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

# Neutrinos and other extra matter
neutrinos = odict([
    ["one_heavy_nu", {
        _theory: {_camb: None,
                  _classy: {"N_ur": 2.0328, "N_ncdm": 1,
                            "m_ncdm": 0.06, "T_ncdm": 0.71611}}}],])

# Reionization
reionization = odict([
    ["std", {
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["tau", {
                _prior: {"min": 0.01, "max": 0.8},
                _p_ref: {_p_dist: "norm", "loc": 0.09, "scale": 0.01},
                _p_proposal: 0.005, _p_label: r"\tau_\mathrm{reio}"}]])}],
    ["gauss_prior", {
        _desc: "Standard reio, lasting delta_z=0.5, gaussian prior around tau=0.07",
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["tau", {
                _prior: {_p_dist: "norm", "loc": 0.07, "scale": 0.02},
                _p_ref: {_p_dist: "norm", "loc": 0.07, "scale": 0.01},
                _p_proposal: 0.005, _p_label: r"\tau_\mathrm{reio}"}]])}],])

# EXPERIMENTS ############################################################################
cmb = odict([
    ["planck_2015_lensing", {
        _desc: "",
        _likelihood: odict([
            ["planck_2015_lowTEB", None],
            ["planck_2015_plikHM", None],
            ["planck_2015_lensing", None]])}],
     ["planck_2015_lensing_bkp", {
        _desc: "",
        _likelihood: odict([
            ["planck_2015_lowTEB", None],
            ["planck_2015_plikHM", None],
            ["planck_2015_lensing", None],
            ["bicep_keck_2015", None]])}],
])
            
# SAMPLERS ###############################################################################
sampler = odict([
    ["MCMC", {
        _desc: "MCMC sampler with covmat learning, and fast dragging.",
        _sampler: {"mcmc": None}}],
    ["PolyChord", {
        _desc: "Nested sampler, affine invariant and multi-modal.",
        _sampler: {"polychord": None}}],
    ])

# PRESETS ################################################################################
preset = odict([
    ["planck_2015_lensing_camb", {
        _desc: "Planck 2015 (Polarised CMB + lensing) with CAMB",
        "theory": "camb",
        "primordial": "SFSR",
        "hubble": "cosmomc_theta",
        "baryons": "omegab_h2",
        "dark_matter": "omegac_h2",
        "neutrinos": "one_heavy_nu",
        "reionization": "std",
        "cmb": "planck_2015_lensing",
        "sampler": "MCMC"}],
    ["planck_2015_lensing_classy", {
        _desc: "Planck 2015 (Polarised CMB + lensing) with CLASS",
        "theory": "classy",
        "primordial": "SFSR",
        "hubble": "theta_s",
        "baryons": "omegab_h2",
        "dark_matter": "omegac_h2",
        "neutrinos": "one_heavy_nu",
        "reionization": "std",
        "cmb": "planck_2015_lensing",
        "sampler": "MCMC"}],
    ["planck_2015_lensing_bicep_camb", {
        _desc: "Planck 2015 + lensing + BKP with CAMB",
        "theory": "camb",
        "primordial": "SFSR_t",
        "hubble": "cosmomc_theta",
        "baryons": "omegab_h2",
        "dark_matter": "omegac_h2",
        "neutrinos": "one_heavy_nu",
        "reionization": "std",
        "cmb": "planck_2015_lensing_bkp",
        "sampler": "MCMC"}],
    ])
