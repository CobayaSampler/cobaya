"""
Tests the facility for importing external priors.

It tests all possible input methods: callable and string
(direct evaluation and ``import_module``).

In each case, it tests the correctness of the values generated, and of the updated info.

The test prior is a gaussian half-ring, combined with a gaussian in one of the tests.

For manual testing, and observing/plotting the density, pass `manual=True` to `body of test`.
"""

# Global
from __future__ import division

# Local
from cobaya.conventions import input_prior
from external_common import info_string, info_callable, info_mixed, info_import
from external_common import body_of_test

# Tests

def test_prior_external_string(tmpdir):
    body_of_test(info_string, input_prior, tmpdir)

def test_prior_external_callable(tmpdir):
    body_of_test(info_callable, input_prior, tmpdir) #####, test_yaml=False)   

def test_prior_external_mixed(tmpdir):
    body_of_test(info_mixed, input_prior, tmpdir)

def test_prior_external_import(tmpdir):
    body_of_test(info_import, input_prior, tmpdir)
