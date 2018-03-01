"""
.. module:: likelihoods._hst_prototype

:Synopsis: Hubble measurement constraint, based on corresponding angular diamter distance value and error
:Author: Antony Lewis

"""

# Python 2/3 compatibility
from __future__ import division

# Global
from scipy.stats import norm
import logging

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import HandledException

# Logging
log = logging.getLogger(__name__)


class _hst_prototype(Likelihood):

    def initialise(self):
        if getattr(self, "zeff", 0) != 0:
            if not hasattr(self, "angconversion"):
                log.error("'angconversion' must be given of effective z is non zero.")
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
