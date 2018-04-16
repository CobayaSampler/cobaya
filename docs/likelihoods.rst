Likelihoods
===========

Input: specifying likelihoods to explore
----------------------------------------

Likelihoods are specified under the `likelihood` block, together with their options:

.. code-block:: yaml

   likelihood:
     [likelihood 1]:
        [option 1]: [value 1]
        [...]
     [likelihood 2]:
        [option 1]: [value 1]
        [...]

Likelihood parameters are specified within the ``params`` block, as explained in :doc:`params_prior`.

**cobaya** comes with a number of *internal* mock and cosmological likelihoods. You can define your *external* ones too with simple Python functions, as explained below.


*Internal* likelihoods: code conventions and defaults
-----------------------------------------------------

*Internal* likelihoods are defined inside the ``likelihoods`` directory of the source tree. Each has its own directory, named as itself, containing at least *three* files:

- A trivial ``__init__.py`` file containing a single line: ``from [name] import [name]``, where ``name`` is the name of the likelihood, and it's folder.
- A ``[name].py`` file, containing the particular class definition of the likelihood, inheriting from the :class:`Likelihood` class (see below).
- A ``[name].yaml`` file containing allowed options for the likelihood and the default *experimental model*:

  .. code-block:: yaml

     # Default options
     likelihood:
       [name]:
         [option 1]: [value 1]
         [...]

     # Experimental model
     params:
       [param 1]:
         prior:
           [prior info]
         [label, ref, etc.]
     prior:
       [prior 1]: [definition]

  The options and parameters defined in this file are the only ones recognized by the likelihood, and they are loaded automatically with their default values (options) and priors (parameters) by simply mentioning the likelihood in the input file, where one can re-define any of those options with a different prior, value, etc (see :ref:`prior_inheritance`). The same parameter may be defined by different likelihoods; in those cases, it needs to have the same default information (prior, label, etc.) in the defaults file of all the likelihoods using it.

- An *optional* ``[name].bibtex`` file containing *bibtex* references associated to the likelihood, if relevant.

.. note::

   Some *mock* likelihoods can have any number of non-predefined parameters, as long as they start with a certain prefix specified by the user with the option ``prefix`` of said likelihood.

.. note::

   Actually, there are some user-defined options that are common to all likelihoods and do not need to be specified in the defaults ``[name].yaml`` file, such as the computational ``speed`` of the likelihood (see :ref:`mcmc_speed_hierarchy`).


.. _likelihood_external:

*External* likelihoods: how to quickly define your own
------------------------------------------------------

*External* likelihoods are defined as:

.. code:: yaml

   likelihood:
     # Simple way (does not admit additional options)
     my_lik_1: [definition]
     # Alternative way (can also take speeds, etc)
     my_lik_1:
       external: [definition]
       speed: [...]
       [more options]

The ``[definition]`` follows the exact same rules as :ref:`external priors <prior_external>`, so check out that section for the details.

The only difference with the custom priors is that external likelihoods can take **derived** parameters. This can only be achieved using a ``def``'ed function (as opposed to a ``lambda`` one. To do that:

1. In your function definition, define a *keyword* argument ``derived`` with a list of derived parameter names as the default value.
2. Inside the function, assume that you have been passed a dictionary through the keyword ``derived`` and **update** it with the derived parameter values corresponding to the input files. 

For an application, check out the :ref:`advanced example <example_advanced_likderived>`.


Implementing your own *internal* likelihood
-------------------------------------------

Even if defining likelihoods with simple Python functions is easy, you may want to create a new *internal*-like likelihood to incorporate to your fork of **cobaya**, or to suggest us to include it in the main source.

Since cobaya was created to be flexible, creating your own likelihood is very easy: simply create a folder with its name under ``likelihoods`` in the source tree and follow the conventions explained above for *internal* likelihoods. Options defined in the ``[name].yaml`` are automatically accesible as attributes of your likelihood class at runtime.

You only need to specify one, or at most four, functions (see the :class:`Likelihood` class documentation below):

- A ``logp`` method taking a dictionary of (sampled) parameter values and returning a log-likelihood.
- An (optional) ``initialise`` method preparing any computation, importing any necessary code, etc.
- An (optional) ``add_theory`` method specifying the requests from the theory code, if it applies.
- An (optional) ``close`` method doing whatever needs to be done at the end of the sampling (e.g. releasing memory).

You can use the :doc:`Gaussian likelihood <likelihood_gaussian>` as a guide. If your likelihood needs a cosmological code, just define one in the input file and you can automatically access it as an attribute of your class: ``[your_likelihood].theory``. Use the :doc:`Planck likelihood <likelihood_planck>` as a guide to create your own cosmological likelihood.

.. note:: ``theory`` and ``derived`` are reserved parameter names: you cannot use them as options in your defaults file!


Likelihood module
=================

.. automodule:: likelihood
   :noindex:

Likelihood class
----------------
   
.. autoclass:: likelihood.Likelihood
   :members:

LikelihoodCollection class
---------------------------
   
.. autoclass:: likelihood.LikelihoodCollection
   :members:
