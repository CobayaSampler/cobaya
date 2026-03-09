"""
Test for external non-linear ratio support via use_non_linear_ratio.

Defines a simple Theory class that returns a constant ratio everywhere,
so the non-linear P(k) should be a predictable rescaling of the linear P(k).
"""

import numpy as np
import pytest

from cobaya.likelihood import Likelihood
from cobaya.model import get_model
from cobaya.theory import Theory
from cobaya.typing import InputDict

from .common import process_packages_path
from .conftest import install_test_wrapper

debug = False
ratio_amp = 2.0


class TrivialNonLinearRatio(Theory):
    """
    A simple non-linear ratio Theory that returns a constant ratio for all k and z.
    """

    def get_requirements(self):
        # must make sure we have latest linear transfer functions/parameters
        # if needed by real model for non-linear correction.
        return "CAMB_transfers"

    def get_non_linear_ratio(self, results):
        """
        Called by CAMB's calculate() with the CAMBdata results object.
        Returns k_h, z and ratio arrays.
        """
        assert np.isclose(results.Params.InitPower.ns, 0.9667)
        # Get the k and z arrays from the results transfer data
        # Use the transfer function's k/z grid
        kh = results.Params.Transfer.kmax
        npoints = 200
        k_h = np.logspace(-4, np.log10(kh), npoints)
        z = np.array(
            results.Params.Transfer.PK_redshifts[
                : results.Params.Transfer.PK_num_redshifts
            ]
        )
        z = np.sort(z)
        ratio = ratio_amp * np.ones((len(z), len(k_h)))
        return {"k_h": k_h, "z": z, "ratio": ratio}

    def calculate(self, state, want_derived=True, **params_values_dict):
        pass

    def get_can_support_params(self):
        return []


class NonLinearRatioLike(Likelihood):
    """Likelihood that requests non-linear Pk to trigger the non-linear path."""

    def get_requirements(self):
        return {"Pk_grid": {"z": [0, 0.5, 1.0], "k_max": 10, "nonlinear": [False, True]}}

    def logp(self, **params_values):
        k_lin, z_lin, pk_lin = self.provider.get_Pk_grid(nonlinear=False)
        k_nonlin, z_nonlin, pk_nonlin = self.provider.get_Pk_grid(nonlinear=True)
        np.testing.assert_allclose(k_nonlin, k_lin)
        np.testing.assert_allclose(z_nonlin, z_lin)
        np.testing.assert_allclose(pk_nonlin, ratio_amp**2 * pk_lin, rtol=1e-4, atol=0)
        return 0


info_non_linear_ratio: InputDict = {
    "likelihood": {"like": NonLinearRatioLike},
    "theory": {
        "camb": {"use_non_linear_ratio": True},
        "my_nonlin": TrivialNonLinearRatio,
    },
    "params": {
        "ombh2": 0.022274,
        "omch2": 0.11913,
        "cosmomc_theta": 0.01040867,
        "tau": 0.0639,
        "ns": 0.9667,
        "As": 2.105e-9,
    },
    "stop_at_error": True,
    "debug": debug,
}


def test_trivial_non_linear_ratio(packages_path, skip_not_installed):
    """
    Test that use_non_linear_ratio=True with a constant ratio rescales P(k).
    """
    packages_path = process_packages_path(packages_path)
    info_non_linear_ratio["packages_path"] = packages_path

    try:
        import camb

        if not hasattr(camb.nonlinear, "ExternalNonLinearRatio"):
            pytest.skip("CAMB does not yet have ExternalNonLinearRatio")
    except ImportError:
        pass  # let install_test_wrapper handle missing CAMB

    model = install_test_wrapper(skip_not_installed, get_model, info_non_linear_ratio)
    model.loglikes({})
