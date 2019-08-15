"""
.. module:: planck_2018_cmblikes_lensing

:Synopsis: Native python version of the Planck lensing likelihood
:Author: Antony Lewis

"""

from cobaya.likelihoods._base_classes import _cmblikes_prototype


class native(_cmblikes_prototype):
    install_options = {"github_repository": "CobayaSampler/planck_supp_data_and_covmats", "github_release": "master"}
