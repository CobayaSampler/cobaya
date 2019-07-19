"""
.. module:: planck_2018_cmblikes_lensing

:Synopsis: Alternative version of the Planck lensing likelihood
:Author: Antony Lewis

"""

# Local
from cobaya.likelihoods._cmblikes_prototype import _cmblikes_prototype
from cobaya.likelihoods._planck_calibration_base import _planck_calibration_base


class planck_2018_cmblikes_lensing(_cmblikes_prototype, _planck_calibration_base):
    install_options = {"github_repository": "CobayaSampler/planck_supp_data_and_covmats", "github_release": "master"}
