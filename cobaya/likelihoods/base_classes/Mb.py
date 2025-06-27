r"""
.. module:: Mb

:Synopsis: Prototype class for local Hubble parameter measurements quantified in terms
    of the magnitude measurement (closer to what is measured than H0)
:Author: Pablo Lemos
"""

from cobaya.likelihood import Likelihood


class Mb(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "Mb"

    # variables from yaml
    Mb_mean: float
    Mb_std: float

    def initialize(self):
        self.minus_half_invvar = -0.5 / self.Mb_std**2

    def get_requirements(self):
        return {}

    def logp(self, **params_values):
        Mb_theory = params_values["Mb"]
        return self.minus_half_invvar * (Mb_theory - self.Mb_mean) ** 2
