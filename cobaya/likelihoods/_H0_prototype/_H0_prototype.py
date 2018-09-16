"""
.. module:: likelihoods._H0_prototype

:Synopsis: Prototype class for local Hubble parameter measurements
:Author: Jesus Torrado

This is a simple gaussian likelihood for the latest local :math:`H_0` measurements
using a combination of different data.

It defines the following likelihoods

- ``H0_riess2018a``: A legacy local measurement of :math:`H_0`, used in the analysis of
  Planck data for the Planck 2018 release.
- ``H0_riess2018b``: The latest local measurement of :math:`H_0`.

.. note::

   If you use this likelihood, please cite it as:

   - ``H0_riess2018a``: A. Riess et al, *"New Parallaxes of Galactic Cepheids from
     Spatially Scanning the Hubble Space Telescope: Implications for the Hubble Constant"*
     `(arXiv:1801.01120) <https://arxiv.org/abs/1801.01120>`_

   - ``H0_riess2018b``: A. Riess et al, *"Milky Way Cepheid Standards for Measuring Cosmic
     Distances and Application to Gaia DR2: Implications for the Hubble Constant"*
     `(arXiv:1804.10655) <https://arxiv.org/abs/1804.10655>`_


Using a different measurement
-----------------------------

If you would like to use different values for the :math:`H_0` constraint, as a mean and a
standard deviation, simply add the following likelihood, substituting ``mu_H0`` and
``sigma_H0`` for the respective values:

.. code:: yaml

   likelihood:
     my_H0: 'lambda _theory={"H0": None}: stats.norm.logpdf(_theory.get_param("H0"), loc=mu_H0, scale=sigma_H0)'

"""

# Python 2/3 compatibility
from __future__ import division

# Global
from scipy.stats import norm

# Local
from cobaya.likelihood import Likelihood


class _H0_prototype(Likelihood):

    def initialize(self):
        self.norm = norm(loc=self.H0, scale=self.H0_std)

    def add_theory(self):
        self.theory.needs(H0=None)

    def logp(self, **params_values):
        H0_theory = self.theory.get_param("H0")
        return self.norm.logpdf(H0_theory)
