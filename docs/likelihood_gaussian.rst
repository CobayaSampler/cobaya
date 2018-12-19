``gaussian`` likelihood
=======================

A simple (multi-modal if required) Gaussian mixture likelihood, aimed at testing. The pdf is normalized to 1 when integrated over an infinite domain, regardless of the number of modes.

.. warning::

   This likelihood will soon be deprecated in favour of :doc:`likelihood_gaussian_mixture`

Usage
-----

The mean and covariance matrix for one or more modes must be specified with the options ``mean`` and ``covmat`` respectively. The dimensionality of the likelihood and the number of modes are guessed from this options (if they are consistent to each other).

The following example defines 3 modes in 2 dimensions, and expects two parameters whose
names start with ``test_`` and must be defined in the ``params`` block:

.. code-block:: yaml

   likelihood:
     gaussian:
       prefix: test_
       mean: [ [0.1,0.1],
               [0.3,0.3],
               [0.4,0.5] ]
       cov:  [ [[0.01, 0],
                [0,    0.05]],
               [[0.02,  0.003],
                [0.003, 0.01]],
               [[0.01, 0],
                [0,    0.01]] ]


The option ``prefix`` fixes the parameters that will be understood by this likelihood (see :ref:`likehood_mock_params`).

Derived parameters can be tracked, as many as sampled parameters times the number of modes,
and they represent the standardized parameters of each of the modes, i.e. those distributed
as :math:`\mathcal{N}(0,I)` around each mode (notice that if a mode is close to the
boundary of the prior, you should not expect to recover a unit covariance matrix from the
sample).

A delay (in seconds) in the likelihood evaluation can be specified with the keyword
``delay``.

.. note::

   This module also provides functions to generate random means and covariances – see automatic documentation below.

The default option values for this module are

.. literalinclude:: ../cobaya/likelihoods/gaussian/gaussian.yaml
   :language: yaml


Gaussian likelihood class
--------------------------

.. automodule:: likelihoods.gaussian.gaussian
   :noindex:

.. autoclass:: likelihoods.gaussian.gaussian
   :members:

Generating random means and covariance matrices
-----------------------------------------------

.. autofunction:: likelihoods.gaussian.random_mean
.. autofunction:: likelihoods.gaussian.random_cov
.. autofunction:: likelihoods.gaussian.info_random_gaussian
