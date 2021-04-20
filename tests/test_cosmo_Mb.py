import os
import numpy as np

from cobaya.conventions import kinds, _params, _packages_path
from cobaya.run import run
from cobaya.yaml import yaml_load_file
from .common import process_packages_path
from .conftest import install_test_wrapper

fiducial_Mb = -19.2
fiducial_Mb_std = 0.1


def test_Mb_riess2020Mb(packages_path, skip_not_installed):
    body_of_test(packages_path, like_name="H0.riess2020Mb",
                 skip_not_installed=skip_not_installed)


def body_of_test(packages_path, like_name=None, like_info=None, skip_not_installed=False):
    info = {_packages_path: process_packages_path(packages_path),
            kinds.sampler: {"evaluate": None}}
    if like_name:
        info[kinds.likelihood] = {like_name: None}
    elif like_info:
        info[kinds.likelihood] = like_info
        like_name = list(like_info)[0]
    info[_params] = {"Mb": fiducial_Mb}
    updated_info, sampler = install_test_wrapper(skip_not_installed, run, info)
    products = sampler.products()
    # The default values for .get are for the _docs_ test
    mean = updated_info[kinds.likelihood][like_name].get("Mb_mean", fiducial_Mb)
    std = updated_info[kinds.likelihood][like_name].get("Mb_std", fiducial_Mb_std)
    reference_chi2 = (fiducial_Mb - mean) ** 2 / std ** 2
    computed_chi2 = (
        products["sample"]["chi2__" + list(info[kinds.likelihood])[0]].values[0])
    assert np.allclose(computed_chi2, reference_chi2)
