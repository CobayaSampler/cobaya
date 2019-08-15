"""
.. module:: planck_2018_cmblikes_lensing_cmbmarged

:Synopsis: Alternative version of the Planck lensing likelihood,
           marginalized over the CMB power spectra. Native python .dataset-based implementation
:Author: Antony Lewis

"""

from cobaya.likelihoods.planck_2018_cmblikes_lensing import planck_2018_cmblikes_lensing


class cmbmarged(planck_2018_cmblikes_lensing):
    pass
