r"""
.. module:: one

:Synopsis: Unit likelihood: always outputs :math:`0 = \log(1)`.
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import division

# Global
from random import random

# Local
from cobaya.likelihood import AbsorbUnusedParamsLikelihood


class one(AbsorbUnusedParamsLikelihood):
    """
    Likelihood that evaluates to 1.
    """

    def initialize(self):
        if self.noise:
            self.logp = self.logp_noise
        else:
            self.logp = self.logp_one

    def logp_one(self, **params_values):
        self.wait()
        return 0.

    def logp_noise(self, **params_values):
        self.wait()
        return self.noise * random()
