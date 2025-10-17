``gaussian`` likelihood
=======================

A simple single-mode Gaussian likelihood with optional normalization.

This is a simpler alternative to :doc:`gaussian_mixture <likelihood_gaussian_mixture>` for cases where you only need a single Gaussian mode.

Usage
-----

The mean and covariance matrix must be specified with the options ``mean`` and ``cov`` respectively. The dimensionality of the likelihood is determined from these options.

The ``normalized`` parameter (default: ``True``) controls whether to include the full normalization constant. When ``False``, only the chi-squared term is computed, which is useful when only relative likelihoods matter.

The following example defines a 2D Gaussian likelihood:

.. code-block:: yaml

   likelihood:
     gaussian:
       mean: [0.5, 1.0]
       cov: [[0.1, 0.05],
             [0.05, 0.2]]
       normalized: True
       input_params: ['x', 'y']

   params:
     x:
       prior:
         min: 0
         max: 1
     y:
       prior:
         min: 0
         max: 2

The option ``input_params_prefix`` can be used instead of explicit ``input_params``, similar to :doc:`gaussian_mixture <likelihood_gaussian_mixture>`. The number of parameters must match the dimensionality defined by the mean and covariance.

For 1D cases, scalar values can be used:

.. code-block:: yaml

   likelihood:
     gaussian:
       mean: 0.5
       cov: 0.04
       input_params: ['x']

The default option values for this likelihood are:

.. literalinclude:: ../cobaya/likelihoods/gaussian/gaussian.yaml
   :language: yaml


Gaussian likelihood class
-------------------------

.. automodule:: likelihoods.gaussian.gaussian
   :noindex:

.. autoclass:: likelihoods.gaussian.Gaussian
   :members:
