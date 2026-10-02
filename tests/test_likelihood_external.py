"""
Tests the facility for importing external likelihoods.

It tests all possible input methods: callable and string
(direct evaluation and ``import_module``).

In each case, it tests the correctness of the values generated, and of the updated info.

The test likelihood is a gaussian half-ring, combined with a gaussian in one of the tests.

For manual testing, and observing/plotting the density, pass ``manual=True`` to
``body of test``.
"""

import numpy as np

from cobaya.likelihood import Likelihood
from cobaya.model import get_model

from .common_external import (
    body_of_test,
    info_callable,
    info_derived,
    info_import,
    info_method_args,
    info_method_kwargs,
    info_method_unnamed_kwargs,
    info_mixed,
    info_string,
)


def test_likelihood_external_string(tmpdir):
    body_of_test(info_string, "likelihood", tmpdir)


def test_likelihood_external_callable(tmpdir):
    body_of_test(info_callable, "likelihood", tmpdir)


def test_likelihood_external_mixed(tmpdir):
    body_of_test(info_mixed, "likelihood", tmpdir)


def test_likelihood_external_import(tmpdir):
    body_of_test(info_import, "likelihood", tmpdir)


def test_likelihood_external_derived(tmpdir):
    body_of_test(info_derived, "likelihood", tmpdir, derived=True)


def test_likelihood_external_method_args(tmpdir):
    body_of_test(info_method_args, "likelihood", tmpdir)


def test_likelihood_external_method_kwargs(tmpdir):
    body_of_test(info_method_kwargs, "likelihood", tmpdir)


def test_likelihood_external_method_unnamed_kwargs(tmpdir):
    body_of_test(info_method_unnamed_kwargs, "likelihood", tmpdir)


def test_likelihood_zero_dim_array_logp():
    # 0-d arrays (e.g. from numpy or jax) have __len__, but must be treated as scalars
    class ArrayLike(Likelihood):
        params = {"x": None}

        def logp(self, **params_values):
            return np.array(-0.5 * params_values["x"] ** 2)

    prior = {"x": {"prior": {"min": -5, "max": 5}}}
    derived_like = {
        "external": lambda x: (np.array(-0.5 * x**2), {"y": 2 * x}),
        "output_params": ["y"],
    }
    expected = -0.5 * 1.7**2 - np.log(10)
    for like, params in [
        (ArrayLike, prior),
        (lambda x: np.array(-0.5 * x**2), prior),
        (derived_like, dict(prior, y=None)),
    ]:
        model = get_model({"likelihood": {"like": like}, "params": params})
        assert np.isclose(model.logpost({"x": 1.7}), expected)
