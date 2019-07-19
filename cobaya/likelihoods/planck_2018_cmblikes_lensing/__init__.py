"""
.. module:: planck_2018_cmblikes_lensing

:Synopsis: Alternative version of the Planck lensing likelihood
:Author: Antony Lewis

"""

# Local
from cobaya.likelihoods._cmblikes_prototype import _cmblikes_prototype


class planck_2018_cmblikes_lensing(_cmblikes_prototype):
    install_options = {"github_repository": "CobayaSampler/planck_supp_data_and_covmats", "github_release": "master"}
