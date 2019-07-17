"""
.. module:: planck_2015_lensing_cmblikes

:Synopsis: Alternative version of the Planck lensing likelihood,
           marginalized over the CMB power spectrum
:Author: Antony Lewis

"""

from cobaya.likelihoods.planck_2018_cmblikes_lensing import planck_2018_cmblikes_lensing


class planck_2018_cmblikes_lensing_cmbmarged(planck_2018_cmblikes_lensing):
    supp_data_version = "v2.0"
