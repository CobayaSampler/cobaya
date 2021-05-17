r"""
.. module:: Mb

:Synopsis: Prototype class for local type Ia SNe absolute magnitude 
measurements, which combined with a SNe data set yield Hubble parameter constraints
:Author: Pablo Lemos

This is a simple gaussian likelihood for the latest local :math:`M_b` measurements
using a combination of different data.

It defines the following likelihoods

- ``H0.riess2020Mb``: Updated local measurement of :math:`H_0`.
  `(arXiv:2012.08534) <https://arxiv.org/abs/2012.08534>`_

Combining with SNe data sets
----------------------------
This likelihood can only be combined with the Pantheon SNe data set. 
The JLA data set cannot be used in combination with Mb, as it uses a different
callibration!


"""

# Local
from cobaya.likelihood import Likelihood


class Mb(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "Mb"

    # variables from yaml
    Mb_mean: float
    Mb_std: float

    def initialize(self):
        self.minus_half_invvar = - 0.5 / self.Mb_std ** 2

    def get_requirements(self):
        return {}

    def logp(self, **params_values):
        Mb_theory = params_values.get("Mb",None)
        return self.minus_half_invvar * (Mb_theory - self.Mb_mean) ** 2
