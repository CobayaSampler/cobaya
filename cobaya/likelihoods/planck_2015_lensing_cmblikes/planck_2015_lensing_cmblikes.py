"""
.. module:: bicep_keck_2015

:Synopsis: Alternative version of the Planck lensing likelihood
:Author: Antony Lewis

"""

# Global
from __future__ import division, print_function
import os
import logging

# Local
from cobaya.likelihoods._cmblikes_prototype import _cmblikes_prototype
from cobaya.likelihoods._planck_clik_prototype import common_path
from cobaya.install import download_github_release


class planck_2015_lensing_cmblikes(_cmblikes_prototype):

    pass


# name of the supplementary data and covmats repo/folder
supp_data_name = "planck_supp_data_and_covmats"
supp_data_version = "v1.0"


def is_installed(**kwargs):
    return os.path.exists(os.path.realpath(
        os.path.join(kwargs["path"], "data", common_path, supp_data_name)))


def install(path=None, force=False, code=False, data=True, no_progress_bars=False):
    if not data:
        return True
    log = logging.getLogger(__name__.split(".")[-1])
    log.info("Downloading supplementary likelihood data and covmats...")
    # Create common planck folder
    planck_data_path = os.path.realpath(os.path.join(path, "data", common_path))
    return download_github_release(planck_data_path, supp_data_name, supp_data_version,
                                   no_progress_bars=no_progress_bars)
