"""
Tests some Prior methods.
"""

import numpy as np

from cobaya.parameterization import Parameterization
from cobaya.prior import Prior


def test_prior_confidence():
    info_params = {
        "a": {"prior": {"dist": "uniform", "min": 0, "max": 1}},
        "b": {"prior": {"dist": "norm", "loc": 0, "scale": 1}},
        "c": {"prior": {"dist": "beta", "min": 0, "max": 1, "a": 2, "b": 5}},
        "d": {"prior": {"dist": "truncnorm", "loc": 0, "scale": 1, "a": 2, "b": 5}},
    }
    p = Prior(Parameterization(info_params))
    test_confidence_p1 = np.array(
        [
            [0.45, 0.55],
            [-0.12566135, 0.12566135],
            [0.24325963, 0.28641175],
            [2.24101037, 2.31751979],
        ]
    )
    assert np.allclose(p.bounds(confidence=0.1), test_confidence_p1)
    test_bounds_p68 = np.array([[0.0, 1.0], [-0.99445788, 0.99445788], [0.0, 1.0], [2.0, 5.0]])
    assert np.allclose(p.bounds(confidence_for_unbounded=0.68), test_bounds_p68)


def test_prior_truncnorm():
    info_params = {
        "a": {"prior": {"dist": "truncnorm", "loc": 1, "min": 0, "max": 1}},
        "b": {"prior": {"dist": "truncnorm", "loc": 1, "a": -1, "b": 0}},
    }
    p = Prior(Parameterization(info_params))
    bounds = p.bounds(confidence=0.1)
    assert np.allclose(bounds[0], bounds[1])
