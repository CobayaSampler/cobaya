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
2. At runtime, the current theory code instance will be passed through that keyword, so you can use it to invoke the methods that return the necessary products.

For an application, check out :doc:`cosmo_external_likelihood`.

.. note::

   Obviously, ``_derived`` and ``_theory`` are reserved parameter names that you cannot use as arguments in your likelihood definition, except for the purposes explained above.

.. note::

   Input and output (derived) parameters of an external function are guessed from its definition. But in some cases you may prefer not to define your likelihood function with explicit arguments (e.g. if the number of them may vary). In that case, you can manually specify the input and output parameters via the ``input_params`` and ``output_params`` options of the likelihood definition in your input dictionary.

   E.g. the following two snippets are equivalent, but in the second one, one can alter the i/o parameters programmatically:

   .. code:: Python

      # Default: guessed from function signature

      from typing import Mapping

      def my_like1(a0, a1, _derived=["sum_a"]):
          if isinstance(_derived, Mapping):
              _derived["sum_a"] = a0 + a1
          return # some function of `(a0, a1)`

      info_like = {"likelihood": my_like}

   .. code:: Python

      # Manual: no explicit function signature required

      from typing import Mapping

      # Define the lists of i/o params
      my_input_params = ["a0", "a1"]
      my_output_params = ["sum_a"]

      def my_like1(**kwargs):
          current_input_values = [kwargs[p] for p in my_input_params]
          if isinstance(kwargs.get("_derived"), Mapping):
              kwargs["_derived"][my_output_params[0]] = sum(current_input_values)
          return # some function of the input params

      info_like = {"likelihood": {
          "external": my_like,
          "input_params": my_input_params, "output_params": my_output_params}}


*Internal* likelihoods: code conventions and defaults
-----------------------------------------------------

*Internal* likelihoods are defined inside the ``likelihoods`` directory of the source tree, where each subdirectory defines
a subpackage containing one or more likelihoods. Each likelihood inherits from the :class:`.likelihood.Likelihood` class (see below).
The subpackage contains at least two files:

- the standard Python subpackage ``__init__.py`` file
- a ``[ClassName].yaml`` file containing allowed options for each likelihood and the default *experimental model*:

  .. code-block:: yaml

     # Default options
     [option 1]: [value 1]
     [...]

     # Experimental model parameters
     params:
       [param 1]:
         prior:
           [prior info]
         [label, ref, etc.]
     prior:
       [prior 1]: [definition]

  The options and parameters defined in this file are the only ones recognized by the likelihood, and they are loaded automatically with their default values (options) and priors (parameters) by simply mentioning the likelihood in the input file, where one can re-define any of those options with a different prior, value, etc (see :ref:`prior_inheritance`). The same parameter may be defined by different likelihoods; in those cases, it needs to have the same default information (prior, label, etc.) in the defaults file of all the likelihoods using it.

- An *optional* ``[ClassName].bibtex`` file containing *bibtex* references associated with each likelihood, if relevant.
  Inherited likelihoods can be used to share a common .bibtex file, since the bibtex file is use by all descendants unless overridden.

.. note::

   Actually, there are some user-defined options that are common to all likelihoods and do not need to be specified in the defaults ``[ClassName].yaml`` file, such as the computational ``speed`` of the likelihood (see :ref:`mcmc_speed_hierarchy`).


It is up to you where to define your likelihood class(es): the ``__init__`` file can define a class [ClassName] directly, or you can define a class in a ``module.py`` file inside the likelihood directory (subpackage).

Assuming your ``__init__`` file defines the class, or imports it (``from .module_name import ClassName``),
when running Cobaya you can reference the internal likelihood using:

  .. code-block:: yaml

     likelihood:
       directory_name.ClassName:
         [option 1]: [value 1]
         [...]

If you defined the class in *module_name.py* then you would reference it as

  .. code-block:: yaml

     likelihood:
       directory_name.module_name.ClassName:
         [option 1]: [value 1]
         [...]

If the class name is the same as the module name it can be omitted.



Implementing your own *internal* likelihood
-------------------------------------------

Even if defining likelihoods with simple Python functions is easy, you may want to create a new *internal*-like likelihood to incorporate to your fork of **cobaya**, or to suggest us to include it in the main source.

Since cobaya was created to be flexible, creating your own likelihood is very easy: simply create a folder with its name under ``likelihoods`` in the source tree and follow the conventions explained above for *internal* likelihoods. Options defined in the ``[ClassName].yaml`` are automatically accessible as attributes of your likelihood class at runtime.

You only need to specify one, or at most four, functions (see the :class:`.likelihood.Likelihood` class documentation below):

- A ``logp`` method taking a dictionary of (sampled) parameter values and returning a log-likelihood.
- An (optional) ``initialize`` method preparing any computation, importing any necessary code, etc.
- An (optional) ``get_requirements`` method returning dictionary of requests from the theory code, if needed.
- An (optional) ``close`` method doing whatever needs to be done at the end of the sampling (e.g. releasing memory).

You can use the :doc:`Gaussian mixtures likelihood <likelihood_gaussian_mixture>` as a guide. If your likelihood needs a cosmological code, just define one in the input file and you can automatically access it as an attribute of your class: ``[your_likelihood].theory``. Use the :doc:`Planck likelihood <likelihood_planck>` as a guide to create your own cosmological likelihood.

.. note:: ``_theory`` and ``_derived`` are reserved parameter names: you cannot use them as options in your defaults file!

For an application, check out :doc:`cosmo_external_likelihood_class`.

Implementing your own *external* likelihood class
--------------------------------------------------

Instead of including the likelihood within the standard Cobaya likelihoods, you may wish to make an external
package that can be redistributed easily. To do this you make a package containing a class defined exactly the same way
as for internal likelihoods above (inheriting from :class:`Likelihood` as documented below). For example if you have a
package called *mycodes*, containing a likelihood class called MyLike in *mycodes.mylikes*, when running Cobaya you can
use the input

  .. code-block:: yaml

     likelihood:
       mycodes.mylikes.MyLike:
         [option 1]: [value 1]
         [...]

This is assuming that *mycodes* is on your Python path (e.g. it is an installed package).
You can also specify an explicit path for the module, e.g.

  .. code-block:: yaml

     likelihood:
       mycodes.mylikes.MyLike:
         python_path: /path/to/mycodes_dir
         [option 1]: [value 1]
         [...]

For an example class implementation and how to support data file auto-installation, check out :doc:`cosmo_external_likelihood_class`.
There is also a simple `external likelihood package <https://github.com/CobayaSampler/example_external_likelihood>`_
and a real-word cosmology example in the `sample external CMB likelihood <https://github.com/CobayaSampler/planck_lensing_external>`_.

Likelihood module
-----------------

.. automodule:: likelihood
   :noindex:

Likelihood class
^^^^^^^^^^^^^^^^

.. autoclass:: likelihood.Likelihood
   :show-inheritance:
   :members:

Likelihood interface
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: likelihood.LikelihoodInterface
   :members:

LikelihoodCollection class
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: likelihood.LikelihoodCollection
   :show-inheritance:
   :members:
