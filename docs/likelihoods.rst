Likelihoods
===========

Input: specifying likelihoods to explore
----------------------------------------

Likelihoods are specified under the `likelihood` block of your input .yaml file, together with their options:

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
defined in an external module. If your likelihood is just a simple Python functions, using an external function can be convenient.
Any likelihood using data files or with more complex dependencies is best implemented as a new likelihood class.


.. _likelihood_external:

External likelihood functions
---------------------------------

External likelihood functions can be used by using input of the form:

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

The only difference with external priors is that external likelihoods can provide **derived** parameters. To do that:

1. In your function, return a tuple of the log-likelihood and a dictionary of derived parameter values ``{derived_1: value_1, [...]}``.
2. When preparing Cobaya's input, add to your external likelihood info an option ``output_params`` listing the names of the available derived parameters.

For an application, check out the :ref:`advanced example <example_advanced_likderived>`.

If your external likelihood needs the products of a **theory code**:

1. In your function definition, define a *keyword* argument ``_self`` through which at runtime you will get accesst to an instance of the Cobaya likelihood wrapper of your function.
2. When preparing Cobaya's input, add to your external likelihood info an option ``requires`` stating the requirements of your likelihood.
3. At run-time, you can call ``get_[...]`` methods of ``_self.provider`` to get the requested quantities.

For an application, check out :doc:`cosmo_external_likelihood`.

.. note::

   Obviously, ``_theory`` is a reserved parameter name that you cannot use as an argument in your likelihood definition, except for the purposes explained above.

.. note::

   The input parameters of an external function are guessed from its definition. But in some cases you may prefer not to define your likelihood function with explicit arguments (e.g. if the number of them may vary). In that case, you can manually specify the input parameters via the ``input_params`` option of the likelihood definition in your input dictionary.

   E.g. the following two snippets are equivalent, but in the second one, one can alter the input parameters programmatically:

   .. code:: Python

      # Default: guessed from function signature

      def my_like(a0, a1):
          logp =  # some function of `(a0, a1)`
          devived = {"sum_a": a0 + a1}
          return logp, derived

      info_like = {"my_likelihood": {
          "external": my_like, "output_params": ["sum_a"]}}

   .. code:: Python

      # Manual: no explicit function signature required

      # Define the lists of input params
      my_input_params = ["a0", "a1"]

      def my_like(**kwargs):
          current_input_values = [kwargs[p] for p in my_input_params]
          logp =  # some function of the input params
          derived = {"sum_a": sum(current_input_values)}
          return logp, derived

      info_like = {"my_likelihood": {
          "external": my_like,
          "input_params": my_input_params, "output_params": ["sum_a"]}}


.. _likelihood_classes:

Likelihood classes: code conventions and defaults
-------------------------------------------------

Each likelihood inherits from the :class:`.likelihood.Likelihood` class (see below).

*Internal* likelihoods are defined inside the ``likelihoods`` directory of the Cobaya source tree, where each subdirectory defines
a subpackage containing one or more likelihoods. *External* likelihood classes can be defined in an external python package.

In addition to the likelihood class itself, each likelihood can have additional files:

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

   Likelihood class options are inherited from any ancestor class .yaml files or class attributes.
   So there are some user-defined options that are common to all likelihoods and do not need to be specified in the defaults ``[ClassName].yaml`` file, such as the computational ``speed`` of the likelihood (see :ref:`mcmc_speed_hierarchy`).


It is up to you where to define your likelihood class(es): the ``__init__`` file can define a class [ClassName] directly, or you can define a class in a ``module.py`` file inside the likelihood directory (subpackage).

Using an *internal* likelihood class
------------------------------------

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

Using an *external* likelihood class
------------------------------------

If you have a package called *mycodes*, containing a likelihood class called MyLike in *mycodes.mylikes*, when running Cobaya you can
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

If MyLike is imported by your package ``__init__`` you can also simply reference it as ``mycodes.MyLike``.

Implementing your own likelihood class
--------------------------------------

For your likelihood class you just need to define a few standard class methods:

- A :meth:`~likelihood.Likelihood.logp` method taking a dictionary of (sampled) parameter values and returning a log-likelihood.
- An (optional) :meth:`~theory.Theory.initialize` method preparing any computation, importing any necessary code, etc.
- An (optional) :meth:`~theory.Theory.get_requirements` method returning dictionary of requests from a theory code component, if needed.

The latter two methods are standard for all Likelihood and Theory components, and are inherited from the base :class:`~theory.Theory` class.
There are also a number of other methods that you can also implement for more advanced usage.

Options defined in the ``[ClassName].yaml`` are automatically accessible as attributes of your likelihood class at runtime, with values that can be overridden by the
input .yaml file used for the run. If you prefer you can also define class attributes directly, rather than using a .yaml file
(private class attributes that cannot be changed by input parameters should start with an underscore). If you define parameters in the .yaml you may also want to define
their type in the Python source by adding an annotation, which will make it easier to perform automated checks on your code.

For an example class implementation and how to support data file auto-installation, check out :doc:`cosmo_external_likelihood_class`.
There is also a simple `external likelihood package <https://github.com/CobayaSampler/example_external_likelihood>`_
and a real-word cosmology example in the `sample external CMB likelihood <https://github.com/CobayaSampler/planck_lensing_external>`_.
You could also use the :doc:`Gaussian mixtures likelihood <likelihood_gaussian_mixture>` as a guide.

A likelihood package becomes an *internal* likelihood if you move it into Cobaya's ``likelihoods`` directory, but usually this is not necessary.
Keeping your likelihood in a separate packages makes it easier to separately update the codes , and you can their easily publicly distribute your likelihood package
(just having Cobaya as a package requirement).

If your likelihood depends on the calculation of some specific observable (typically depending on input parameters), you may want to split the calculation of the observable
into a separate :class:`~theory.Theory` class. This would allow different likelihoods to share the same theory calculation result, and also allow speed optimization if computing the likelihood
is much faster than calculating the observables that it depends on.
To understand how to make your own Theory and Likelihood classes, and handle dependencies between then, see :doc:`theories_and_dependencies`.


Likelihood module
-----------------

.. automodule:: likelihood
   :noindex:

Likelihood class
^^^^^^^^^^^^^^^^

.. autoclass:: likelihood.Likelihood
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
