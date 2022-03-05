import os
import numpy as np

from cobaya.run import run
from cobaya.yaml import yaml_load_file
from .common import process_packages_path
from .conftest import install_test_wrapper

fiducial_H0 = 70
fiducial_H0_std = 1

fiducial_Mb = -19.2
fiducial_Mb_std = 0.1


def test_H0_riess2018a(packages_path, skip_not_installed):
    body_of_test(packages_path, like_name="H0.riess2018a",
                 skip_not_installed=skip_not_installed)


def test_H0_riess201903(packages_path, skip_not_installed):
    body_of_test(packages_path, like_name="H0.riess201903",
                 skip_not_installed=skip_not_installed)


def test_H0_docs(packages_path, skip_not_installed):
    like_info = yaml_load_file(os.path.join(
        os.path.dirname(__file__), "../docs/src_examples/H0/custom_likelihood.yaml"))
    like_name = list(like_info["likelihood"])[0]
    like_info["likelihood"][like_name]["external"] = \
        like_info["likelihood"][like_name]["external"].replace(
            "mu_H0", str(fiducial_H0))
    like_info["likelihood"][like_name]["external"] = \
        like_info["likelihood"][like_name]["external"].replace(
            "sigma_H0", str(fiducial_H0_std))
    body_of_test(packages_path, like_info=like_info["likelihood"],
                 skip_not_installed=skip_not_installed)


def test_Mb_riess2020Mb(packages_path, skip_not_installed):
    # Does NOT test use with Pantheon, just trivial chi2 check
    body_of_test(packages_path, like_name="H0.riess2020Mb",
                 skip_not_installed=skip_not_installed, Mb=True)


def body_of_test(packages_path, like_name=None, like_info=None,
                 skip_not_installed=False, Mb=False):
    info = {"packages_path": process_packages_path(packages_path),
            "sampler": {"evaluate": None}}
    if like_name:
        info["likelihood"] = {like_name: None}
    elif like_info:
        info["likelihood"] = like_info
        like_name = list(like_info)[0]
    fiducial, fiducial_std, name = (fiducial_Mb, fiducial_Mb_std, "Mb") if Mb \
        else (fiducial_H0, fiducial_H0_std, "H0")
    info["params"] = {name: fiducial}
    updated_info, sampler = install_test_wrapper(skip_not_installed, run, info)
    products = sampler.products()
    # The default values for .get are for the _docs_ test
    mean = updated_info["likelihood"][like_name].get("%s_mean" % name, fiducial)
    std = updated_info["likelihood"][like_name].get("%s_std" % name, fiducial_std)
    reference_chi2 = (fiducial - mean) ** 2 / std ** 2
    chi2_label = "chi2__" + list(info["likelihood"])[0]
    computed_chi2 = products["sample"][chi2_label].to_numpy(dtype=np.float64)[0]
    assert np.allclose(computed_chi2, reference_chi2)
