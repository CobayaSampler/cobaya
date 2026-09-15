"""CLASS sampled inputs stay on classy when a likelihood also requires them.

cobaya #494. Does not import the CLASS C extension.
"""

from cobaya.theories.classy.classy import classy


def test_classy_support_keeps_sampled_densities_and_renames():
    class Dummy:
        renames = {
            "omegach2": "omega_cdm",
            "omegabh2": "omega_b",
            "omegac": "Omega_cdm",
            "omegab": "Omega_b",
        }

    supported = set(classy.get_can_support_params(Dummy()))
    assert {"H0", "omega_b", "omega_cdm", "omegabh2", "omegach2"} <= supported
    assert "theta_s_100" not in supported
    assert "Omega_m" not in supported
