"""
.. module:: likelihoods._hst_prototype

:Synopsis: Hubble measurement constraint,
           based on corresponding angular diamter distance value and error
:Author: Antony Lewis (translated to Python by Jesus Torrado)

.. |br| raw:: html

   <br />

This is a simple gaussian likelihood using the latest :math:`H_0` measurement from the
Hubble Space telescope.

.. note::

   If you use this likelihood, please cite it as:
   |br|
   A. Riess et al, *"New Parallaxes of Galactic Cepheids from Spatially Scanning the
   Hubble Space Telescope: Implications for the Hubble Constant"*
   `(arXiv:1801.01120) <https://arxiv.org/abs/1801.01120>`_

"""

# Python 2/3 compatibility
from __future__ import division

# Global
from scipy.stats import norm

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import HandledException


class _hst_prototype(Likelihood):

    def initialise(self):
        if getattr(self, "zeff", 0) != 0:
            if not hasattr(self, "angconversion"):
                self.log.error(
                    "'angconversion' must be given of effective z is non zero.")
                raise HandledException
            self.theory.needs({"angular_diameter_distance": {"redshifts": self.zeff}})
        else:
            self.zeff = 0
        self.norm = norm(loc=self.H0, scale=self.H0_err)

    def logp(self, **params_values):
        if self.zeff != 0:
            theory = (self.angconversion /
                      self.theory.get_angular_diameter_distance(self.zeff))
        else:
            theory = self.theory.get_param("H0")
        return self.norm.logpdf(theory)
