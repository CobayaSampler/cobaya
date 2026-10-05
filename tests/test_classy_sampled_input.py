"""Sampled CLASS inputs stay assigned when a likelihood also requests them."""

from pathlib import Path

import cobaya.theories.classy as classy_pkg
from cobaya.theories.classy.classy import classy
from cobaya.yaml import yaml_load_file


def test_sampled_density_remains_supported_after_a_likelihood_request():
    theory = classy.__new__(classy)
    info_path = Path(classy_pkg.__file__).with_name("classy.yaml")
    info = yaml_load_file(str(info_path))
    theory.renames = info["renames"]
    supported = set(theory.get_can_support_params())
    for name in (
        "H0",
        "h",
        "omega_b",
        "omega_cdm",
        "tau_reio",
        "A_s",
        "n_s",
        "omegabh2",
        "omegach2",
    ):
        assert name in supported, name
    for derived in ("theta_s_100", "Omega_m", "sigma8"):
        assert derived not in supported, derived
