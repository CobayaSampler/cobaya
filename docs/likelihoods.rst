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

**cobaya** comes with a number of *internal* general and cosmological likelihoods.
You can also define your *external* likelihoods with simple Python functions, or by implementing a Python class
defined in an external module.


.. _likelihood_external:

*External* likelihoods: how to quickly define your own functions
----------------------------------------------------------------

*External* likelihood functions are defined as:

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

The only difference with external priors is that external likelihoods can provide **derived** parameters. This can only be achieved using a ``def``'ed function (as opposed to a ``lambda`` one. To do that:

1. In your function definition, define a *keyword* argument ``_derived`` with a list of derived parameter names as the default value.
2. Inside the function, assume that you have been passed a dictionary through the keyword ``_derived`` and **update** it with the derived parameter values corresponding to the input files.

For an application, check out the :ref:`advanced example <example_advanced_likderived>`.

If your external likelihood needs the products of a **theory code**:

1. In your function definition, define a *keyword* argument ``_theory`` with a default value stating the *needs* of your theory code, i.e. the argument that will be passed to the ``needs`` method of the theory code, to let it know what needs to be computed at every iteration.
2. At runtime, the current theory code instance will be passed through that keyword, so you can use it to invoke the methods that return the necessary producs.

For an application, check out :doc:`cosmo_external_likelihood`.

.. note:: Obviously, ``_derived`` and ``_theory`` are reserved parameter names that you cannot use as arguments in your likelihood definition, except for the purposes explained above.


*Internal* likelihoods: code conventions and defaults
-----------------------------------------------------

*Internal* likelihoods are defined inside the ``likelihoods`` directory of the source tree. Each has its own directory, named as itself, containing at least *three* files:

- A trivial ``__init__.py`` file containing a single line: ``from [name] import [name]``, where ``name`` is the name of the likelihood, and it's folder.
- A ``[name].py`` file, containing the particular class definition of the likelihood, inheriting from the :class:`.likelihood.Likelihood` class (see below).
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

   Actually, there are some user-defined options that are common to all likelihoods and do not need to be specified in the defaults ``[name].yaml`` file, such as the computational ``speed`` of the likelihood (see :ref:`mcmc_speed_hierarchy`).


Implementing your own *internal* likelihood
-------------------------------------------

Even if defining likelihoods with simple Python functions is easy, you may want to create a new *internal*-like likelihood to incorporate to your fork of **cobaya**, or to suggest us to include it in the main source.

Since cobaya was created to be flexible, creating your own likelihood is very easy: simply create a folder with its name under ``likelihoods`` in the source tree and follow the conventions explained above for *internal* likelihoods. Options defined in the ``[name].yaml`` are automatically accessible as attributes of your likelihood class at runtime.

You only need to specify one, or at most four, functions (see the :class:`.likelihood.Likelihood` class documentation below):

- A ``logp`` method taking a dictionary of (sampled) parameter values and returning a log-likelihood.
- An (optional) ``initialize`` method preparing any computation, importing any necessary code, etc.
- An (optional) ``add_theory`` method specifying the requests from the theory code, if it applies.
- An (optional) ``close`` method doing whatever needs to be done at the end of the sampling (e.g. releasing memory).

You can use the :doc:`Gaussian mixtures likelihood <likelihood_gaussian_mixture>` as a guide. If your likelihood needs a cosmological code, just define one in the input file and you can automatically access it as an attribute of your class: ``[your_likelihood].theory``. Use the :doc:`Planck likelihood <likelihood_planck>` as a guide to create your own cosmological likelihood.

.. note:: ``_theory`` and ``_derived`` are reserved parameter names: you cannot use them as options in your defaults file!

For an application, check out :doc:`cosmo_external_likelihood_class`.

Implementing your own *external* likelihood class
--------------------------------------------------

Instead of including the likelihood within the standard Cobaya likelihood modules, you may wish to make an external
package that can be redistributed easily. To do this you make a module containing a class defined exactly the same way
as for internal likelihoods above (inheriting from :class:`Likelihood` as documentated below). By default the class is
assumed to have the same name as the containing file, e.g. if you have a package called *mycodes*, containing
a likelihood in *mycodes.mylike* you can use

  .. code-block:: yaml

     likelihood:
       mycodes.mylike:
         [option 1]: [value 1]
         [...]

This is assuming that *mycodes.mylike* contains a likelihood class called *mylike* and that the *mycodes* is on your
Python path. You can also specify an explicit class name and path for the module, e.g.

  .. code-block:: yaml

     likelihood:
       mycodes.mylike:
         class_name: MyLikelihood
         python_path: /path/to/mycodes_dir
         [option 1]: [value 1]
         [...]

For an example class implementation, check out :doc:`cosmo_external_likelihood_class`.

Likelihood module
-----------------

.. automodule:: likelihood
   :noindex:

Likelihood class
^^^^^^^^^^^^^^^^

.. autoclass:: likelihood.Likelihood
   :members:

LikelihoodCollection class
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: likelihood.LikelihoodCollection
   :members:
