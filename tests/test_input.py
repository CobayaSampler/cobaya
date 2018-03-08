# Tests to check correct input parsing and inheritance of defaults

# Global
from __future__ import division
from copy import deepcopy
import pytest

# Local
from cobaya.conventions import _likelihood, _sampler, _prior
from cobaya.conventions import _params, _p_label
from cobaya.run import run
from cobaya.log import HandledException
from cobaya.input import get_default_info

# Aux definitions and functions

test_info_common = {
    _likelihood: {"_test": None},
    _sampler: {"evaluate": None}}


def test_prior_inherit_nonegiven():
    updated_info, products = run(test_info_common)
    likname = test_info_common[_likelihood].keys()[0]
    default_info = get_default_info(likname, _likelihood)
    assert updated_info[_prior] == default_info[_prior]


def test_prior_inherit_differentgiven():
    test_info = deepcopy(test_info_common)
    test_info[_prior] = {"third": "lambda a1: 1"}
    updated_info, products = run(test_info)
    likname = test_info_common[_likelihood].keys()[0]
    default_info = get_default_info(likname, _likelihood)
    default_info[_prior].update(test_info[_prior])
    assert updated_info[_prior] == default_info[_prior]


def test_prior_inherit_samegiven():
    test_info = deepcopy(test_info_common)
    likname = test_info_common[_likelihood].keys()[0]
    default_info = get_default_info(likname, _likelihood)
    name, prior = deepcopy(default_info[_prior]).popitem()
    test_info[_prior] = {name: prior}
    updated_info, products = run(test_info)
    assert updated_info[_prior] == default_info[_prior]


def test_prior_inherit_samegiven_differentdefinition():
    test_info = deepcopy(test_info_common)
    likname = test_info_common[_likelihood].keys()[0]
    default_info = get_default_info(likname, _likelihood)
    name, prior = deepcopy(default_info[_prior]).popitem()
    test_info[_prior] = {name: "this is not a prior"}
    with pytest.raises(HandledException):
        updated_info, products = run(test_info)


def test_inherit_label_and_bounds():
    test_info = deepcopy(test_info_common)
    likname = test_info_common[_likelihood].keys()[0]
    default_info_params = get_default_info(likname, _likelihood)[_params]
    test_info[_params] = deepcopy(default_info_params)
    test_info[_params]["a1"].pop(_p_label, None)
    # First, change one limit (so no inheritance at all)
    test_info[_params]["b1"].pop("min")
    new_max = 2
    test_info[_params]["b1"]["max"] = new_max
    updated_info, products = run(test_info)
    assert updated_info[_params]["a1"] == default_info_params["a1"]
    assert updated_info[_params]["b1"].get("min") is None
    assert updated_info[_params]["b1"]["max"] == new_max
    # Second, remove limits, so they are inherited
    test_info = deepcopy(test_info_common)
    test_info[_params] = deepcopy(default_info_params)
    test_info[_params]["b1"].pop("min")
    test_info[_params]["b1"].pop("max")
    updated_info, products = run(test_info)
    assert updated_info[_params]["b1"]["min"] == default_info_params["b1"]["min"]
    assert updated_info[_params]["b1"]["max"] == default_info_params["b1"]["max"]
