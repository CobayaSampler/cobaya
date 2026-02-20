"""
Test for external non-linear ratio support via use_non_linear_ratio.

Defines a trivial Theory class that returns ratio=1 everywhere,
so the non-linear P(k) should equal the linear P(k).
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


class TrivialNonLinearRatio(Theory):
    """
    A trivial non-linear ratio Theory that returns ratio=1 for all k and z,
    meaning no non-linear correction is applied.
    """

    def get_non_linear_ratio(self, results):
        """
        Called by CAMB's calculate() with the CAMBdata results object.
        Returns k_h, z and ratio arrays. ratio=1 means identity (no correction).
        """
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
        ratio = np.ones((len(z), len(k_h)))
        return {"k_h": k_h, "z": z, "ratio": ratio}

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Nothing to pre-compute for the trivial case
        pass

    def get_can_support_params(self):
        return []


class NonLinearRatioLike(Likelihood):
    """Likelihood that requests non-linear Pk to trigger the non-linear path."""

    def get_requirements(self):
        return {"Pk_grid": {"z": [0, 0.5, 1.0], "k_max": 10, "nonlinear": True}}

    def logp(self, **params_values):
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
        "logA": 3.047,
    },
    "stop_at_error": True,
    "debug": debug,
}


def test_trivial_non_linear_ratio(packages_path, skip_not_installed):
    """
    Test that use_non_linear_ratio=True with ratio=1 runs without error.
    Once CAMB implements ExternalNonLinearRatio, this test will also verify
    that ratio=1 produces output equivalent to linear P(k).
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
