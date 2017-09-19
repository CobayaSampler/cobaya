# Tests to check correct input parsing and inheritance of defaults

# Global
from __future__ import division
import os
from copy import deepcopy
import pytest

# Local
from cobaya.conventions import _defaults_file, _likelihood, _sampler, _prior
from cobaya.tools import get_folder
from cobaya.yaml_custom import yaml_load_file, yaml_dump
from cobaya.run import run
from cobaya.log import HandledException

def test_prior_inherit_nonegiven():
    updated_info, products = run(test_info)
    default_info = _get_default_info(test_info[_likelihood].keys()[0], _likelihood)
    assert updated_info[_prior] == default_info[_prior]

def test_prior_inherit_differentgiven():
    test_info == deepcopy(test_info)
    test_info[_prior] = {"third": "lambda a1: 1"}
    updated_info, products = run(test_info)
    default_info = _get_default_info(test_info[_likelihood].keys()[0], _likelihood)
    default_info[_prior].update(test_info[_prior])
    assert updated_info[_prior] == default_info[_prior]

def test_prior_inherit_samegiven():
    test_info == deepcopy(test_info)
    default_info = _get_default_info(test_info[_likelihood].keys()[0], _likelihood)
    name, prior = deepcopy(default_info[_prior]).popitem()
    test_info[_prior] = {name: prior}
    updated_info, products = run(test_info)
    assert updated_info[_prior] == default_info[_prior]

def test_prior_inherit_samegiven_differentdefinition():
    test_info == deepcopy(test_info)
    default_info = _get_default_info(test_info[_likelihood].keys()[0], _likelihood)
    name, prior = deepcopy(default_info[_prior]).popitem()
    test_info[_prior] = {name: "this is not a prior"}
    with pytest.raises(HandledException):
        updated_info, products = run(test_info)

# Aux definitions and functions

test_info = {
    _likelihood: {"_test": None},
    _sampler: {"evaluate": None}}

def _get_default_info(module, kind):
    path_to_defaults = os.path.join(get_folder(module, kind), _defaults_file)
    return yaml_load_file(path_to_defaults)
