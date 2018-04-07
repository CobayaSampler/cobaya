from __future__ import division, print_function
from collections import OrderedDict as odict

from cobaya.conventions import _theory, _params
from cobaya.conventions import _prior, _p_ref, _p_proposal, _p_label, _p_dist

_camb = "camb"
_classy = "classy"
_desc = "desc"

# Theory codes
theory = odict([[_camb, None], [_classy, None]])

# Primordial
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
                _p_proposal: 0.03, _p_label: r"r_{0.05}"}]])
    }],
])


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
                _p_proposal: 0.005, _p_label: r"\tau_\mathrm{reio}"}]])}],
    ])
