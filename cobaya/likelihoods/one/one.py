"""
.. module:: one

:Synopsis: Mock constant value (1) likelihood
:Author: Jesus Torrado

Likelihoods that evaluates to 1. Useful to explore priors.

Options and defaults
--------------------

Simply copy this block in your input ``yaml`` file and modify whatever options you want
(you can delete the rest).

.. literalinclude:: ../cobaya/likelihoods/one/defaults.yaml
   :language: yaml

"""

# Python 2/3 compatibility
from __future__ import division

# Global
import numpy as np

# Local
from cobaya.likelihood import Likelihood

class one(Likelihood):
    """
    Likelihood that evaluates to 1.
    """

    def logp(self, **params_values):
        self.wait()
        return 0.
