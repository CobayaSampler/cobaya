"""
Tests the facility for importing external priors.

It tests all possible input methods: callable and string
(direct evaluation and ``import_module``).

In each case, it tests the correctness of the values generated, and of the updated info.

The test prior is a gaussian half-ring, combined with a gaussian in one of the tests.

For manual testing, and observing/plotting the density, pass `manual=True`
to `body of test`.
"""
# Local
from .common_external import info_string, info_callable, info_mixed, info_import
from .common_external import body_of_test


def test_prior_external_string(tmpdir):
    body_of_test(info_string, "prior", tmpdir)


def test_prior_external_callable(tmpdir):
    body_of_test(info_callable, "prior", tmpdir)


def test_prior_external_mixed(tmpdir):
    body_of_test(info_mixed, "prior", tmpdir)


def test_prior_external_import(tmpdir):
    body_of_test(info_import, "prior", tmpdir)
