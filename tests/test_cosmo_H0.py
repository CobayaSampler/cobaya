import os
import numpy as np
from scipy.stats import norm

from cobaya.conventions import kinds, _params, _packages_path
from cobaya.run import run
from cobaya.yaml import yaml_load_file
from .common import process_packages_path
from .conftest import install_test_wrapper

fiducial_H0 = 70
fiducial_H0_std = 1

def test_H0_riess2018a(packages_path, skip_not_installed):
    body_of_test(packages_path, like_name="H0.riess2018a",
                 skip_not_installed=skip_not_installed)


def test_H0_riess201903(packages_path, skip_not_installed):
    body_of_test(packages_path, like_name="H0.riess201903",
                 skip_not_installed=skip_not_installed)


def test_H0_docs(packages_path, skip_not_installed):
    like_info = yaml_load_file(os.path.join(
        os.path.dirname(__file__), "../docs/src_examples/H0/custom_likelihood.yaml"))
    like_name = list(like_info[kinds.likelihood])[0]
    like_info[kinds.likelihood][like_name]["external"] = \
        like_info[kinds.likelihood][like_name]["external"].replace(
            "mu_H0", str(fiducial_H0))
    like_info[kinds.likelihood][like_name]["external"] = \
        like_info[kinds.likelihood][like_name]["external"].replace(
            "sigma_H0", str(fiducial_H0_std))
    body_of_test(packages_path, like_info=like_info[kinds.likelihood],
                 skip_not_installed=skip_not_installed)


def body_of_test(packages_path, like_name=None, like_info=None, skip_not_installed=False):
    info = {_packages_path: process_packages_path(packages_path),
            kinds.sampler: {"evaluate": None}}
    if like_name:
        info[kinds.likelihood] = {like_name: None}
    elif like_info:
        info[kinds.likelihood] = like_info
        like_name = list(like_info)[0]
    info[_params] = {"H0": fiducial_H0}
    updated_info, sampler = install_test_wrapper(skip_not_installed, run, info)
    products = sampler.products()
    # The default values for .get are for the _docs_ test
    mean = updated_info[kinds.likelihood][like_name].get("H0_mean", fiducial_H0)
    std = updated_info[kinds.likelihood][like_name].get("H0_std", fiducial_H0_std)
    reference_value = -2 * norm.logpdf(fiducial_H0, loc=mean, scale=std)
    computed_value = (
        products["sample"]["chi2__" + list(info[kinds.likelihood])[0]].values[0])
    assert np.allclose(computed_value, reference_value)
