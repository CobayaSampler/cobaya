"""
.. module:: planck_2015_lensing_cmblikes

:Synopsis: Alternative version of the Planck lensing likelihood
:Author: Antony Lewis

"""

# Global
from __future__ import division, print_function
import os
import logging

# Local
from cobaya.likelihoods._cmblikes_prototype import _cmblikes_prototype
from cobaya.install import download_github_release


class planck_2015_lensing_cmblikes(_cmblikes_prototype):
    # Installation routines ##################################################################

    # name of the supplementary data and covmats repo/folder
    supp_data_name = "planck_supp_data_and_covmats"
    supp_data_version = "v2.0"

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(os.path.join(path, "data", cls.supp_data_name))

    @classmethod
    def is_installed(cls, **kwargs):
        return os.path.exists(os.path.realpath(
            os.path.join(kwargs["path"], "data", cls.supp_data_name)))

    @classmethod
    def install(cls, path=None, force=False, code=False, data=True, no_progress_bars=False):
        if not data:
            return True
        log = logging.getLogger(__name__.split(".")[-1])
        log.info("Downloading supplementary likelihood data and covmats...")
        return download_github_release(os.path.join(path, "data"), cls.supp_data_name,
                                       cls.supp_data_version, no_progress_bars=no_progress_bars)
