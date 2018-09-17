``mcmc`` sampler
================

.. |br| raw:: html

   <br />

.. note::
   **If you use this sampler, please cite it as:**
   |br|
   `A. Lewis and S. Bridle, "Cosmological parameters from CMB and other data: A Monte Carlo approach"
   (arXiv:astro-ph/0205436) <https://arxiv.org/abs/astro-ph/0205436>`_
   |br|
   `A. Lewis, "Efficient sampling of fast and slow cosmological parameters"
   (arXiv:1304.4473) <https://arxiv.org/abs/1304.4473>`_
   |br|
   If you use *fast-dragging*, you should also cite
   |br|
   `R.M. Neal, "Taking Bigger Metropolis Steps by Dragging Fast Variables"
   (arXiv:math/0502099) <https://arxiv.org/abs/math/0502099>`_


This is the Markov Chain Monte Carlo Metropolis sampler used by CosmoMC, and described in
`Lewis, "Efficient sampling of fast and slow cosmological parameters" (arXiv:1304.4473)
<https://arxiv.org/abs/1304.4473>`_.

The proposal pdf is a gaussian mixed with an exponential pdf in random directions,
which is more robust to misestimation of the width of the proposal than a pure gaussian.
The scale width of the proposal can be specified per parameter with the property
``proposal`` (it defaults to the standard deviation of the reference pdf, if defined,
or the prior's one, if not). However, initial performance will be much better if you
provide a covariance matrix, which overrides the default proposal scale width set for
each parameter.

.. note::

   The ``proposal`` size for a certain parameter should be close to its **conditional**
   posterior, not it marginalised one, since for strong degeneracies, the latter being
   wider than the former, it could cause the chain to get stuck.

A callback function can be specified through the ``callback_function`` option. In it, the
sampler instance is accessible as ``sampler_instance``, which has ``prior``, ``likelihood``
and (sample) ``collection`` as attributes.

Initial point and covariance of the proposal pdf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The initial points for the chains are sampled from the *reference* pdf
(see :doc:`params_prior`). The reference pdf can be a fixed point, and in that case the
chain starts always from that same point. If there is no reference pdf defined for a
parameter, the initial sample is drawn from the prior instead.

Example *parameters* block:

.. code-block:: yaml
   :emphasize-lines: 10,17

   params:
     a:
      ref:
        min: -1
        max:  1
      prior:
        min: -2
        max:  2
      latex: \\alpha
      proposal: 0.5
     b:
      ref: 2
      prior:
        min: -1
        max:  4
      latex: \\beta
      proposal: 0.25
     c:
      ref:
        dist: norm
        loc: 0
        scale: 0.2
      prior:
        min: -1
        max:  1
      latex: \\gamma

+ ``a`` -- the initial point of the chain is drawn from an uniform pdf between -1 and 1,
  and its proposal width is 0.5.
+ ``b`` -- the initial point of the chain is always 2,
  and its proposal width is 0.25.
+ ``c`` -- the initial point of the chain is drawn from a gaussian centred at 0
  with standard deviation 0.2; its proposal width is not specified, so it is taken to be
  that of the reference pdf, 0.2.

A good initial covariance matrix for the proposal is critical for convergence.
It can be specified either with the property ``proposal`` of each parameter, as shown
above, or through ``mcmc``'s property ``covmat``, as a file name (including path,
if not located at the invocation folder).
The first line of the ``covmat`` file must start with ``#``, followed by a list of parameter
names, separated by a space. The rest of the file must contain the covariance matrix,
one row per line. It does not need to contain the same parameters as the sampled ones:
it overrides the ``proposal``'s (and adds covariances) for the sampled parameters,
and ignores the non-sampled ones.

An example for the case above::

   # a     b
     0.1   0.01
     0.01  0.2

In this case, internally, the final covariance matrix of the proposal would be::

   # a     b     c
     0.1   0.01  0
     0.01  0.2   0
     0     0     0.04

If the option `learn_proposal` is set to ``True``, the covariance matrix will be updated
once in a while to accelerate convergence.

If you are not sure that your posterior has one single mode, or if its shape is very
irregular, you should probably set ``learn_proposal: False``.

If you don't know how good your initial guess for starting point and covariance of the
proposal are, it is a good idea to allow for a number of initial *burn in* samples,
e.g. 10 per dimension. This can be specified with the parameter ``burn_in``.
These samples will be ignored for all purposes (output, convergence, proposal learning...)


.. _mcmc_speed_hierarchy:

Taking advantage of a speed hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The proposal pdf is *blocked* by speeds, i.e. it allows for efficient sampling of a
mixture of *fast* and *slow* parameters, such that we can avoid recomputing the slowest
parts of the likelihood when sampling along the fast directions only.

Two different sampling schemes are available to take additional advantage from a speed
hierarchy:

- **Dragging the fast parameters:** implies a number of intermediate steps when jumping between fast+slow combinations, such that the jump in the fast parameters is optimised with respect to the jump in the slow parameters to explore any possible degeneracy between them. If enabled (``drag: True``), tries to spend the same amount of time doing dragging steps as it takes to compute a jump in the slow direction (make sure your likelihoods ``speed``'s are accurate; see below).

- **Oversampling the fast parameters:** consists simply of taking a larger proportion of steps in the faster directions, useful when exploring their conditional distributions is cheap. If enabled (``oversample: True``), it tries to spend the same amount of time in each block.

In general, the *dragging* method is the recommended one, since oversampling can potentially produce too many samples. For a thorough description of both methods, see
`A. Lewis, "Efficient sampling of fast and slow cosmological parameters" (arXiv:1304.4473) <https://arxiv.org/abs/1304.4473>`_.

The relative speeds can be specified per likelihood/theory, with the option ``speed``,
preferably in evaluations per second (approximately).

To measure the speed of your likelihood, set ``timing: True`` at the highest level of your input (i.e. not inside any of the blocks), set the ``mcmc`` options ``burn_in: 0`` and ``max_samples`` to a reasonably large number (so that it will be done in a few minutes), and check the output: it should have printed, towards the end, computation times for the likelihoods and the theory code in seconds, the *inverse* of which are the speeds.

If the speed has not been specified for a likelihood, it is assigned the slowest one in
the set. If two or more likelihoods with different speeds share a parameter,
said parameter is assigned to a separate block with a speed that takes into account the
computation time of all the likelihoods it belongs to.

For example:

.. code-block:: yaml

   theory:
     theory_code:
       speed: 2

   likelihood:
     lik_a:
     lik_b:
       speed: 4

Here, evaluating the theory code is the slowest step, while the ``lik_b`` is faster.
Likelihood ``lik_a`` is assumed to be as slow as the theory code.


Options and defaults
--------------------

Simply copy this block in your input ``yaml`` file and modify whatever options you want (you can delete the rest).

.. literalinclude:: ../cobaya/samplers/mcmc/mcmc.yaml
   :language: yaml


Module documentation
--------------------

.. automodule:: samplers.mcmc.mcmc
   :noindex:

Sampler class
^^^^^^^^^^^^^

.. autoclass:: samplers.mcmc.mcmc
   :members:

Proposal
^^^^^^^^

.. automodule:: samplers.mcmc.proposal
   :noindex:

.. autoclass:: samplers.mcmc.proposal.CyclicIndexRandomizer
   :members:
.. autoclass:: samplers.mcmc.proposal.RandDirectionProposer
   :members:
.. autoclass:: samplers.mcmc.proposal.BlockedProposer
   :members:

