``gaussian_mixture`` likelihood
===============================

A simple (multi-modal if required) Gaussian mixture likelihood. The pdf is normalized to 1 when integrated over an infinite domain, regardless of the number of modes.

Usage
-----

The mean and covariance matrix for one or more modes must be specified with the options ``means`` and ``covmats`` respectively. The dimensionality of the likelihood and the number of modes are guessed from this options (if they are consistent to each other).

The following example defines 3 modes in 2 dimensions, and expects two parameters whose
names start with ``test_`` and must be defined in the ``params`` block:

.. code-block:: yaml

   likelihood:
     gaussian_mixture:
       input_params_prefix: test_
       means: [ [0.1,0.1],
                [0.3,0.3],
                [0.4,0.5] ]
       covs:  [ [[0.01, 0],
                 [0,    0.05]],
                [[0.02,  0.003],
                 [0.003, 0.01]],
                [[0.01, 0],
                 [0,    0.01]] ]


The option ``input_params_prefix`` fixes the parameters that will be understood by this likelihood: it is a special kind of likelihood that can have any number of non-predefined parameters, as long as they start with this prefix. If this prefix is not defined (or defined to an empty string), the likelihood will understand all parameter as theirs. The number of parameters taken as input must match the dimensionality defined by the means and covariance matrices.

Derived parameters can be tracked, as many as sampled parameters times the number of modes,
and they represent the standardized parameters of each of the modes, i.e. those distributed
as :math:`\mathcal{N}(0,I)` around each mode (notice that if a mode is close to the
boundary of the prior, you should not expect to recover a unit covariance matrix from the
sample). To track them, add the option `derived: True`, and they will be identified by a prefix defined by ``output_params_prefix``.

A delay (in seconds) in the likelihood evaluation can be specified with the keyword
``delay``.

.. note::

   This module also provides functions to generate random means and covariances – see automatic documentation below.

The default option values for this likelihood are

.. literalinclude:: ../cobaya/likelihoods/gaussian_mixture/gaussian_mixture.yaml
   :language: yaml


Gaussian mixture likelihood class
---------------------------------

.. automodule:: likelihoods.gaussian_mixture.gaussian_mixture
   :noindex:

.. autoclass:: likelihoods.gaussian_mixture.gaussian_mixture
   :members:

Generating random means and covariance matrices
-----------------------------------------------

.. autofunction:: likelihoods.gaussian_mixture.random_mean
.. autofunction:: likelihoods.gaussian_mixture.random_cov
.. autofunction:: likelihoods.gaussian_mixture.info_random_gaussian_mixture
