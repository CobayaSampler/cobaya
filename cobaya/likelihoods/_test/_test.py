# Python 2/3 compatibility
from __future__ import division

# Global
import numpy as np

# Local
from cobaya.likelihood import Likelihood


class _test(Likelihood):
    """
    Likelihood that evaluates to 1.
    """

    def logp(self, **params_values):
        self.wait()
        params_values["_derived"]["b1"] = 0
        return 0.
