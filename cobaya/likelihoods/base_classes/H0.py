r"""
.. module:: H0

:Synopsis: Prototype class for local Hubble parameter measurements
:Author: Jesus Torrado, Pablo Lemos

This is a simple gaussian likelihood for the latest local :math:`H_0` measurements
using a combination of different data.

It defines the following likelihoods

- ``H0.riess2018a``: A legacy local measurement of :math:`H_0`, used in the analysis of
  Planck data for the Planck 2018 release.
  `(arXiv:1801.01120) <https://arxiv.org/abs/1801.01120>`_
- ``H0.riess2018b``: Updated local measurement of :math:`H_0`.
  `(arXiv:1804.10655) <https://arxiv.org/abs/1804.10655>`_
- ``H0.riess201903``:  Riess et al. 2019 constraint
  `(arXiv:1903.07603) <https://arxiv.org/abs/1903.07603>`_
- ``H0.riess2020``:  Riess et al. 2020 constraint
  `(arXiv:2012.08534) <https://arxiv.org/abs/2012.08534>`_
- ``H0.freedman2020``:  Freedman et al. 2020 constraint
  `(arXiv:2002.01550) <https://arxiv.org/abs/2002.01550>`_

The above are all based on simple Gaussian posteriors for the value of H0 today, which
is not what is directly measured. It can be more accurate to put a constraint on the
intrinsic magnitude, which can then be combined with Supernovae constraints to relate to
the expansion. An example is provided in

- ``H0.riess2020Mb``:  Riess et al. 2020 constraint on M_b
  `(arXiv:2012.08534) <https://arxiv.org/abs/2012.08534>`_

which should be run in combination with sn.pantheon with use_abs_mag: True
(contributed by Pablo Lemos).

Using a different measurement
-----------------------------

If you would like to use different values for the :math:`H_0` constraint, as a mean and a
standard deviation, simply add the following likelihood, substituting ``mu_H0`` and
``sigma_H0`` for the respective values:

.. literalinclude:: ./src_examples/H0/custom_likelihood.yaml
   :language: yaml

"""

# Local
from cobaya.likelihood import Likelihood


class H0(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "H0"

    # variables from yaml
    H0_mean: float
    H0_std: float

    def initialize(self):
        self.minus_half_invvar = - 0.5 / self.H0_std ** 2

    def get_requirements(self):
        return {'H0': None}

    def logp(self, **params_values):
        H0_theory = self.provider.get_param("H0")
        return self.minus_half_invvar * (H0_theory - self.H0_mean) ** 2
