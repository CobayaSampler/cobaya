from __future__ import division, print_function
from collections import OrderedDict as odict

from cobaya.conventions import _theory, _params, _likelihood, _sampler
from cobaya.conventions import _prior, _p_ref, _p_proposal, _p_label, _p_dist

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
            ["logAs1e10", {_prior: {"min": 2, "max": 4},
                           _p_ref: {_p_dist: "norm", "loc": 3.1, "scale": 0.001},
                           _p_proposal: 0.001, _p_label: r"\log(10^{10} A_s",
                           "drop": True}],
            ["As", "lambda logAs1e10: 1e-10*np.exp(logAs1e10)"]])}],
    ["SFSRt", {
        _desc: "Vanilla Single-field Slow-roll Inflation WITH TENSORS",
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["logAs1e10", {
                _prior: {"min": 2, "max": 4},
                _p_ref: {_p_dist: "norm", "loc": 3.1, "scale": 0.001},
                _p_proposal: 0.001, _p_label: r"\log(10^{10} A_s", "drop": True}],
            ["As", "lambda logAs1e10: 1e-10*np.exp(logAs1e10)"],
            ["r", {
                _prior: {"min": 0, "max": 3},
                _p_ref: {_p_dist: "norm", "loc": 0, "scale": 0.03},
                _p_proposal: 0.03, _p_label: r"r_{0.05}"}]])}]])

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
            ["cosmomc_theta_100", {
                _prior: {"min": 0.5, "max": 10},
                _p_ref: {_p_dist: "norm", "loc": 1.0411, "scale": 0.0004},
                _p_proposal: 0.0002, _p_label: r"100\theta_\mathrm{MC}"}]])}],
    ["theta_s", {
        _desc: "Angular size of sound horizon (CLASS only)",
        _theory: {_classy: None},
        _params: odict([
            ["100*theta_s", "lambda theta_s_100: theta_s_100"],
            ["theta_s_100", {
                _prior: {"min": 0.5, "max": 10},
                _p_ref: {_p_dist: "norm", "loc": 1.0418, "scale": 0.0004},
                _p_proposal: 0.0002, _p_label: r"100\theta_s"}]])}]])

# Barions
barions = odict([
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
        _theory: {_camb: None, _classy: None},
        _params: odict([
            ["tau", {
                _prior: {_p_dist: "norm", "loc": 0.09, "scale": 0.01},
                _p_ref: {_p_dist: "norm", "loc": 0.09, "scale": 0.01},
                _p_proposal: 0.005, _p_label: r"\tau_\mathrm{reio}"}]])}],])

# EXPERIMENTS ############################################################################
cmb = odict([
    ["planck 2015", {
        _desc: "",
        _likelihood: odict([
            ["planck_2015_lowTEB", None],
            ["planck_2015_plikHM", None],
            ["planck_2015_lensing", None]])}],
#     ["bkp oct 2015", {
#        _desc: "",
#        _likelihood: odict([
#            ["planck_2015_lowTEB", None],
#            ["planck_2015_plikHM", None],
#            ["planck_2015_lensing", None]])},
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
        
