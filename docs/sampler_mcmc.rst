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
<https://arxiv.org/abs/1304.4473>`_. It works well on simple uni-modal (or only weakly multi-modal) distributions.

The proposal pdf is a gaussian mixed with an exponential pdf in random directions,
which is more robust to misestimation of the width of the proposal than a pure gaussian.
The scale width of the proposal can be specified per parameter with the property
``proposal`` (it defaults to the standard deviation of the reference pdf, if defined,
or the prior's one, if not). However, initial performance will be much better if you
provide a covariance matrix, which overrides the default proposal scale width set for
each parameter.

.. note::

   The ``proposal`` size for a certain parameter should be close to its **conditional**
   posterior, not it marginalized one, since for strong degeneracies, the latter being
   wider than the former, it could cause the chain to get stuck.

If the distribution being sampled is known have tight strongly non-linear parameter degeneracies, re-define the sampled
parameters to remove the degeneracy before sampling (linear degeneracies are not a problem, esp. if you provide an
approximate initial covariance matrix).


.. _mcmc_progress:

Progress monitoring – *new in 2.1*
----------------------------------

When writing to the hard drive, the MCMC sampler produces an additional ``[output_prefix].progress`` file containing the acceptance rate and the Gelman :math:`R-1` diagnostics (for means and confidence level contours) per checkpoint, so that the user can monitor the convergence of the chain. In interactive mode (when running inside a Python script of in the Jupyter notebook), an equivalent ``progress`` table in a ``pandas.DataFrame`` is returned among the ``products``.

The ``mcmc`` modules provides a plotting tool to produce a graphical representation of convergence, see :func:`~samplers.mcmc.plot_progress`. An example plot can be seen below:

.. code:: Python

   from cobaya.samplers.mcmc import plot_progress
   # Assuming chain saved at `chains/gaussian`
   plot_progress("chains/gaussian", fig_args={"figsize": (6,4)})
   import matplotlib.pyplot as plt
   plt.tight_layout()
   plt.show()

.. image:: img/mcmc_progress.png
   :align: center

When writing to the hard drive (i.e. when an ``[output_prefix].progress`` file exists), one can produce these plots even if the sampler is still running.


.. _mcmc_callback:

Callback functions
------------------

A callback function can be specified through the ``callback_function`` option. It must be a function of a single argument, which at runtime is the current instance of the ``mcmc`` sampler. You can access its attributes and methods inside your function, including the ``collection`` of chain points and the ``model`` (of which ``prior`` and ``likelihood`` are attributes). For example, the following callback function would print the points added to the chain since the last callback:

.. code:: python

    def my_callback(sampler):
        print(sampler.collection[sampler.last_point_callback:])

The callback function is called every ``callback_every`` points have been added to the chain, or at every checkpoint if that option has not been defined.


Initial point and covariance of the proposal pdf
------------------------------------------------

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
      latex: \alpha
      proposal: 0.5
     b:
      ref: 2
      prior:
        min: -1
        max:  4
      latex: \beta
      proposal: 0.25
     c:
      ref:
        dist: norm
        loc: 0
        scale: 0.2
      prior:
        min: -1
        max:  1
      latex: \gamma

+ ``a`` -- the initial point of the chain is drawn from an uniform pdf between -1 and 1,
  and its proposal width is 0.5.
+ ``b`` -- the initial point of the chain is always 2,
  and its proposal width is 0.25.
+ ``c`` -- the initial point of the chain is drawn from a gaussian centred at 0
  with standard deviation 0.2; its proposal width is not specified, so it is taken to be
  that of the reference pdf, 0.2.

Fixing the initial point is not usually recommended, since to assess convergence it is useful to run multiple chains
(which you can do using MPI), and use the difference between the chains to assess convergence: if the chains all start
in exactly the same point, the chains could appear to have converged just because they started at the same place. On the
other hand if your initial points are spread much more widely than the posterior it could take longer for chains to
converge.

A good initial covariance matrix for the proposal is useful for faster and more reliable convergence.
It can be specified either with the property ``proposal`` of each parameter, as shown
above, or through ``mcmc``'s property ``covmat``, as a file name (including path,
if not located at the invocation folder).
The first line of the ``covmat`` file must start with ``#``, followed by a list of parameter
names, separated by a space. The rest of the file must contain the covariance matrix,
one row per line. It does not need to contain the same parameters as the sampled ones:
where sampled parameters exist in the file the they override the ``proposal`` (and add covariance information),
non-sampled ones are ignored, and for missing parameters the specified input ``proposal`` is used, assuming no correlations.

An example for the case above::

   # a     b
     0.1   0.01
     0.01  0.2

In this case, internally, the final covariance matrix of the proposal would be::

   # a     b     c
     0.1   0.01  0
     0.01  0.2   0
     0     0     0.04


If the option ``learn_proposal`` is set to ``True``, the covariance matrix will be updated
regularly. This means that accuracy of the initial covariance is not critical, and even if you do not initially know
the covariance, it will be adaptively learnt (just make sure your ``proposal`` widths are sufficiently small that
chains can move and hence explore the local shape; if your widths are too wide the parameter may just remain stuck).

If you are not sure that your posterior has one single mode, or if its shape is very
irregular, you should probably set ``learn_proposal: False``; however the MCMC sampler is not likely to work
well in this case and other samplers designed for multi-modal distributions may be much more efficient.

If you don't know how good your initial guess for the starting point and covariance is, a number of initial *burn in* samples
can be ignored from the start of the chains (e.g. 10 per dimension). This can be specified with the parameter ``burn_in``.
These samples will be ignored for all purposes (output, convergence, proposal learning...). Of course there may well
also be more burn in after these points are discarded, as the chain points converge (and, using ``learn_proposal``, the proposal estimates
also converge). Often removing the first 30% the entire final chains gives good results (using ``ignore_rows=0.3`` when analysing with `getdist <http://getdist.readthedocs.org/en/latest/>`_).


.. _mcmc_speed_hierarchy:

Taking advantage of a speed hierarchy
-------------------------------------

The proposal pdf is *blocked* by speeds, i.e. it allows for efficient sampling of a
mixture of *fast* and *slow* parameters, such that we can avoid recomputing the slowest
parts of the likelihood when sampling along the fast directions only. This is often very useful when the likelihoods
have large numbers of nuisance parameters, but recomputing the likelihood for different sets of nuisance parameters is fast.

Two different sampling schemes are available to take additional advantage from a speed
hierarchy:

- **Dragging the fast parameters:** implies a number of intermediate steps when jumping between fast+slow combinations, such that the jump in the fast parameters is optimized with respect to the jump in the slow parameters to explore any possible degeneracy between them. If enabled (``drag: True``), tries to spend the same amount of time doing dragging steps as it takes to compute a jump in the slow direction (make sure your likelihoods ``speed``'s are accurate; see below).

- **Oversampling the fast parameters:** consists simply of taking a larger proportion of steps in the faster directions, useful when exploring their conditional distributions is cheap. If enabled (``oversample: True``), it tries to spend the same amount of time in each block.

In general, the *dragging* method is the recommended one if there are non-trivial degeneracies between fast and slow parameters.
Oversampling can potentially produce very large output files; dragging outputs
smaller chain files since fast parameters are effectively partially marginalized over internally. For a thorough description of both methods and references, see
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

.. _mcmc_speed_hierarchy_manual:

Manual specification of speed-blocking – *new in 1.1*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Automatic* speed-blocking takes advantage of differences in speed *per likelihood* (or theory). If the parameters of your likelihood or theory have some internal speed hierarchy that you would like to exploit (e.g. if your likelihood internally caches the result of a computation depending only on a subset of the likelihood parameters), you can specify a fine-grained list of parameter blocks and their speeds under the ``mcmc`` option ``blocking``.

E.g. if a likelihood depends of parameters ``a``, ``b`` and ``c`` and the cost of varying ``a`` is *twice* as big as the other two, your ``mcmc`` block should look like

.. code-block:: yaml

   mcmc:
     blocking:
       - [1, [a]]
       - [2, [b,c]]
     oversampling: True  # if desired
     # or `drag: True`, if 2-blocks only (put fastest last)
     # [other options...]

.. warning::

   The cost of a parameter block should be the **total** cost of varying one parameter in the block, i.e. it needs to take into account the time needed to re-compute every part of the code that depends (directly or indirectly) on it.

   For example, if varying parameter ``a`` in the example above would also force a re-computation of the part of the code associated to parameters ``b`` and ``c``, then the relative cost of varying the parameters in each block would not be 2-to-1, but (2+1)-to-1, meaning relative speeds would be 1 and 3.

.. note::

   If ``blocking`` is specified, it must contain **all** the sampled parameters.

.. note::

   If automatic learning of the proposal covariance is enabled, after some checkpoint the proposed steps will mix parameters from different blocks, but *always towards faster ones*. Thus, it is important to specify your blocking in **ascending order of speed**, when not prevented by the architecture of your likelihood (e.g. due to internal caching of intermediate results that require some particular order of parameter variation).

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

Progress monitoring
^^^^^^^^^^^^^^^^^^^

.. autofunction:: samplers.mcmc.plot_progress

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

