# Tests to check correct input parsing and inheritance of defaults

# Global
from copy import deepcopy
import pytest

# Local
from cobaya.conventions import kinds, partag, _prior, _params
from cobaya.run import run
from cobaya.log import LoggedError
from cobaya.input import get_default_info

# Aux definitions and functions

test_info_common = {
    kinds.likelihood: {"_test": None},
    kinds.sampler: {"evaluate": None}}


def test_prior_inherit_nonegiven():
    updated_info, products = run(test_info_common)
    likname = list(test_info_common[kinds.likelihood])[0]
    default_info = get_default_info(likname, kinds.likelihood)
    assert updated_info[_prior] == default_info[_prior]


def test_prior_inherit_differentgiven():
    test_info = deepcopy(test_info_common)
    test_info[_prior] = {"third": "lambda a1: 1"}
    updated_info, products = run(test_info)
    likname = list(test_info_common[kinds.likelihood])[0]
    default_info = get_default_info(likname, kinds.likelihood)
    default_info[_prior].update(test_info[_prior])
    assert updated_info[_prior] == default_info[_prior]


def test_prior_inherit_samegiven():
    test_info = deepcopy(test_info_common)
    likname = list(test_info_common[kinds.likelihood])[0]
    default_info = get_default_info(likname, kinds.likelihood)
    name, prior = deepcopy(default_info[_prior]).popitem()
    test_info[_prior] = {name: prior}
    updated_info, products = run(test_info)
    assert updated_info[_prior] == default_info[_prior]


def test_prior_inherit_samegiven_differentdefinition():
    test_info = deepcopy(test_info_common)
    likname = list(test_info_common[kinds.likelihood])[0]
    default_info = get_default_info(likname, kinds.likelihood)
    name, prior = deepcopy(default_info[_prior]).popitem()
    test_info[_prior] = {name: "this is not a prior"}
    with pytest.raises(LoggedError):
        updated_info, products = run(test_info)


def test_inherit_label_and_bounds():
    test_info = deepcopy(test_info_common)
    likname = list(test_info_common[kinds.likelihood])[0]
    default_info_params = get_default_info(likname, kinds.likelihood)[_params]
    test_info[_params] = deepcopy(default_info_params)
    test_info[_params]["a1"].pop(partag.latex, None)
    # Remove limits, so they are inherited
    test_info = deepcopy(test_info_common)
    test_info[_params] = deepcopy(default_info_params)
    test_info[_params]["b1"].pop("min")
    test_info[_params]["b1"].pop("max")
    updated_info, products = run(test_info)
    assert updated_info[_params]["b1"]["min"] == default_info_params["b1"]["min"]
    assert updated_info[_params]["b1"]["max"] == default_info_params["b1"]["max"]
