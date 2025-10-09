"""
Tests for the Gaussian likelihood.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from cobaya.model import get_model
from cobaya.typing import InputDict


def test_gaussian_1d():
    """Test 1D Gaussian likelihood for both normalized and unnormalized cases."""
    mean = 0.5
    std = 0.2
    cov = std**2

    for normalized in [True, False]:
        info: InputDict = {
            "likelihood": {
                "gaussian": {
                    "mean": mean,
                    "cov": cov,
                    "normalized": normalized,
                    "input_params": ["x"],
                }
            },
            "params": {"x": {"prior": {"min": 0, "max": 1}, "proposal": 0.1}},
        }

        model = get_model(info)

        # Test at various points including mean
        test_points = [mean, 0.1, 0.3, 0.7, 0.9]
        for x in test_points:
            loglike = model.loglike({"x": x})[0]
            if normalized:
                expected = multivariate_normal.logpdf([x], [mean], [[cov]])
            else:
                chi2_val = (x - mean) ** 2 / cov
                expected = -0.5 * chi2_val
            assert np.isclose(loglike, expected, rtol=1e-10)


def test_gaussian_2d():
    """Test 2D Gaussian likelihood for both normalized and unnormalized cases."""
    mean = np.array([0.5, 1.0])
    cov = np.array([[0.1, 0.05], [0.05, 0.2]])

    for normalized in [True, False]:
        # Test with numpy arrays directly (not .tolist())
        info: InputDict = {
            "likelihood": {
                "gaussian": {
                    "mean": mean,  # numpy array directly
                    "cov": cov,  # numpy array directly
                    "normalized": normalized,
                    "input_params": ["x", "y"],
                }
            },
            "params": {
                "x": {"prior": {"min": 0, "max": 1}, "proposal": 0.1},
                "y": {"prior": {"min": 0, "max": 2}, "proposal": 0.1},
            },
        }

        model = get_model(info)

        # Test at various points including mean
        test_points = [(mean[0], mean[1]), (0.2, 0.8), (0.7, 1.3), (0.1, 1.8)]
        for x, y in test_points:
            loglike = model.loglike({"x": x, "y": y})[0]
            if normalized:
                expected = multivariate_normal.logpdf([x, y], mean, cov)
            else:
                inv_cov = np.linalg.inv(cov)
                delta = np.array([x, y]) - mean
                chi2_val = delta.T @ inv_cov @ delta
                expected = -0.5 * chi2_val
            assert np.isclose(loglike, expected, rtol=1e-10)


def test_gaussian_scalar_input():
    """Test that scalar inputs work for 1D case."""
    mean = 0.5
    cov = 0.04  # scalar

    info: InputDict = {
        "likelihood": {
            "gaussian": {
                "mean": mean,  # scalar
                "cov": cov,  # scalar
                "normalized": True,
                "input_params": ["x"],
            }
        },
        "params": {"x": {"prior": {"min": 0, "max": 1}, "proposal": 0.1}},
    }

    model = get_model(info)

    # Should work the same as 1D array case
    loglike = model.loglike({"x": 0.3})[0]
    expected = multivariate_normal.logpdf([0.3], [mean], [[cov]])
    assert np.isclose(loglike, expected, rtol=1e-10)


def test_gaussian_parameter_prefix():
    """Test using parameter prefix instead of explicit input_params."""
    mean = np.array([0.5, 1.0])
    cov = np.array([[0.1, 0.05], [0.05, 0.2]])

    info: InputDict = {
        "likelihood": {
            "gaussian": {
                "mean": mean.tolist(),
                "cov": cov.tolist(),
                "normalized": True,
                "input_params_prefix": "test_",
            }
        },
        "params": {
            "test_0": {"prior": {"min": 0, "max": 1}, "proposal": 0.1},
            "test_1": {"prior": {"min": 0, "max": 2}, "proposal": 0.1},
        },
    }

    model = get_model(info)

    # Test at mean
    loglike_mean = model.loglike({"test_0": mean[0], "test_1": mean[1]})[0]
    expected_mean = multivariate_normal.logpdf(mean, mean, cov)
    assert np.isclose(loglike_mean, expected_mean, rtol=1e-10)


def test_gaussian_errors():
    """Test error handling."""
    # Test mismatched dimensions
    with pytest.raises(Exception):  # Should raise LoggedError
        info: InputDict = {
            "likelihood": {
                "gaussian": {
                    "mean": [0.5, 1.0],  # 2D
                    "cov": [[0.1]],  # 1D
                    "input_params": ["x", "y"],
                }
            },
            "params": {
                "x": {"prior": {"min": 0, "max": 1}},
                "y": {"prior": {"min": 0, "max": 2}},
            },
        }
        get_model(info)

    # Test non-square covariance
    with pytest.raises(Exception):  # Should raise LoggedError
        info: InputDict = {
            "likelihood": {
                "gaussian": {
                    "mean": [0.5, 1.0],
                    "cov": [[0.1, 0.05, 0.02], [0.05, 0.2, 0.01]],  # 2x3 matrix
                    "input_params": ["x", "y"],
                }
            },
            "params": {
                "x": {"prior": {"min": 0, "max": 1}},
                "y": {"prior": {"min": 0, "max": 2}},
            },
        }
        get_model(info)

    # Test singular covariance matrix
    with pytest.raises(Exception):  # Should raise LoggedError
        info: InputDict = {
            "likelihood": {
                "gaussian": {
                    "mean": [0.5, 1.0],
                    "cov": [[0.1, 0.1], [0.1, 0.1]],  # Singular matrix
                    "input_params": ["x", "y"],
                }
            },
            "params": {
                "x": {"prior": {"min": 0, "max": 1}},
                "y": {"prior": {"min": 0, "max": 2}},
            },
        }
        get_model(info)
