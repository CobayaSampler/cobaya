Development notes
==================

This document gathers some notes about the development flow, release checklist, and general design decisions and their motivation. It is recommended to read it thoroughly before attempting to make modifications to this code.


Development flow
-----------------

1. Clone the repo from github.
2. From its folder, install in editable mode: ``pip install -e . --user``
3. Modify stuff.
4. Test (py3, nosetests, etc)
5. Pull requests, cloning, etc.
6. etc.


Release checklist
-----------------

+ Version number
+ Pypi upload
+ ...


New features
------------

1. Document
2. Add test


Design choices
--------------

Generality and modularity
^^^^^^^^^^^^^^^^^^^^^^^^^

This code is being developed as a general-purpose sampler, with a Bayesian focus. The different aspects of the sampling, namely prior, likelihood and sampler, are kept as isolated as possible: the prior and sampler know which parameters are sampled and fixed, but not the likelihood, which does not need to know; the sampler does now know which likelihood understands which parameters, since it does not care (just cares about their respective priors, speeds, etc). This designs choices take some compromises, but are a fair price for making the code more easily extendable and maintainable: e.g. adding a new likelihood on the fly, or modify their parameters, needs no modification of the main code.

The cosmology only enters through particular likelihood and theory modules, and the main source does not contain significant cosmological code or experimental data, just wrappers, so that the general user does not need to download gigabytes of data.

Ideally, in the near future, the source will be split in two: a general sampler package on one side, and the cosmological modules on the other.


Parameters
^^^^^^^^^^

Likelihoods (and theory codes) can take arguments modifying their behaviour in two different ways:

+ those that are not expected to ever be sampled (paths to data, on-off options, precision parameters, etc.) , which we will call *options*, and pass within the likelihood specification.
+ those that may be sampled or fixed, which we call *parameters* and pass through the parameters section

.. code-block:: yaml
   :emphasize-lines: 3,6,7,10

   likelihood:
     gaussian:
       option: value

   params:
     fixed_parameter: value
     sampled_parameter:
       prior:
         [...]
     derived_parameter: # nothing here!


Different likelihoods may share part of the same experimental model, and so they may have parameters in common! To deal with this, we have taken the following design choices:

+ The parameters could have been defined inside the ``likelihood`` block of each likelihood, either as *options* or within a ``params`` block. We have chosen to define them in a *common* external block to make the sharing of parameters look more natural.
+ Two likelihood parameters with the same name are considered by default to be the same parameter. Thus, when defining custom likelihoods or creating new interfaces for external likelihoods, use preferably non-trivial names, e.g. instead of ``A``, use ``amplitude``, or even better, ``amplitude_of_something``.
+ Parameters priors, labels, etc. are inherited from the definitions in the ``defaults.yaml`` of each likelihood, which are gathered at execution time. If there is a conflict between the priors (or fixed value, or derived state) defined in each of the defaults file, an error will be produced, unless the user settles the conflict by specifying the desired behaviour for said parameter in the input file.


Mock likelihoods
""""""""""""""""

Mock likelihoods, which exist mainly for test purposes, have often dynamically defined parameters (e.g. its dimensionality is defined through its options, see :doc:`likelihood_gaussian`).

These kind of likelihoods implement an option, ``mock_prefix``, indicating that said likelihood will understand any parameter starting by said prefix.


How parameters are passed around
""""""""""""""""""""""""""""""""

The sampler (MCMC, PolyChord, etc.) does not need to know about to which likelihood understands and make use of each particular parameter. It is the :class:`likelihood.LikelihoodCollection` class who is responsible for passing the parameters around to the likelihoods (and to the theory code). Since, as we said above, more than one likelihood may share a parameter, the parameters cannot be separated in blocks trivially. Thus, we have to choose between

a) blocking the parameters per likelihood (blocks could have a non-trivial intersection),
b) pass all the parameters to each likelihoods, and let each likelihood recognize their own parameters.

.. note::
   Whatever the choice, we need each likelihood to be able to recognize its own parameters. But they do, since they have been defined in the respective ``defaults.yaml`` file (or for external likelihoods, they can be extracted using Python's *introspection* capabilities).

The complicated bits that could make us decide between one approach or the other are:

*Pythonicity* and information compartmentalization
  Method (a) makes likelihood calls more natural and *pythonic*: the arguments of the method to get the log-likelihood are simply the parameters. The price is more overhead on the side of the main code, in particular the :class:`likelihood.LikelihoodCollection` class, that has to block the parameters by likelihood, and manage possible overlaps between blocks. In contrast, method (b) is simpler to code and more manageable, since we don't need to know beforehand which parameters to pass to which likelihood; but it is less pythonic since the arguments of the log-likelihood calls are not explicit, but a dictionary.

How to deal with derived parameters?
  Derived parameters have their value given back by the likelihood, opposite to sampled parameters. Since in Python parameter values (``float``'s) are *immutable*, they are passed by value, not by reference, so their value cannot be *modified back*. Thus, method (a) needs to get the derived parameters in a way different than passing them as arguments, e.g. as a dictionary passes through a ``derived`` keyword argument. This is not ideal, since there are reasons for them to be on the same grounds as sampled parameters (e.g. we may want to allow the user to sample the parameter ``x`` and get the *derived* value of the parameter ``log(x)``, or the other way around). In method (b), since dictionaries are *mutable* objects, when their contents are modified the modifications are permanent, which makes a natural way of dealing with derived parameters on the same ground as sampled parameters. **Method (b) is simpler here.**

What if two likelihoods had the same name for a *different* parameter?
  In method (a), we would re-specify both parameters in the input info, each prefixed by the name of its respective likelihood followed by an agreed separator, e.g. ``__``; that way, it's easy to assign them to their respective blocks. Method (b) would have it hard to deal with that without having to modify the one of the conflicting likelihoods; it could e.g. implement the possibility of *renaming* parameters dynamically: in the likelihood block we indicate that its version of the shared parameter ``a`` is dealt with as ``a_something``. **Method (a) proves more natural here.**

We have chosen method **(a)**. From here one, any implementation details described depend on this choice.

Dealing with derived parameters
"""""""""""""""""""""""""""""""

Computing derived parameters may be expensive, and we won't need them for samples that are not going to be stored (e.g. they are rejected, only used just to perform *fast-dragging*, or just to train a model). Thus, their computation must be **optional**.

But in general, one needs the current *state* of the sampled parameters to compute the derived ones. Thus, if the sample is potentially an interesting one, we will have to get the derived parameters immediately after the likelihood computation (otherwise, if we have jumped somewhere else and then decided to get them, we may have to re-compute the likelihood at the point of interest, which is probably more costly than having computed derived parameters that we are likely to throw away). It is up the each sampler to decide whether the derived parameters at one particular sample are worth computing.

We could implement the passing of derived parameters in two ways:

a) A keyword option in the log-likelihood function to request the computation of derived parameters (passed back as a mutable argument of that same function).
b) An optional method of the :class:`Likelihood` class, say ``get_derived``, that is called whenever the derived parameters are needed (e.g. just by the :class:`Collection` class).

In option (b) the ``get_derived`` method, when called, would always have to be called immediately after computing the likelihood; otherwise, we risk doing something that changes the *state* of the likelihood (and/or the theory code) potentially returning the wrong set of values. For this reason, we adopt option (a).

Both for the ``log-likelihood`` method of a :class:`Likelihood` and for external likelihood functions, we will create a keyword argument ``derived``. If that keyword valued ``None``, the derived parameters will not be computed, and if valued as an empty dictionary, it will be used to return the derived parameters (thanks to Python's passing mutable objects by reference, not value).

From the sampler point of view, the dictionary above becomes a list in the call to obtain the log-pdf of the :class:`LikelihoodCollection`: an empty list is passed which is populated with a list of the derived parameters values, in the order in which :class:`LikelihoodCollection` stores them. This way, the sampler does not need to keep track of the names of the derived parameters.

Unfortunately, for many samplers, such as basic MH-MCMC, we do not know a priori if we are going to save a particular point, so we are forced to compute derived parameters even when they are not necessary. In those case, if their computation is prohibitively expensive, it may be faster to run the sample without derived parameters, and add them after the sampling process is finished.


Reparametrization layer
"""""""""""""""""""""""

**Statistical parameters** are specified according to their rôles for the **sampler**: as *fixed*, *sampled* and *derived*. On the other hand, the **likelihood** (and the **theory code**, if present) cares only about input and output arguments. In a trivial case, those would correspond respectively to *fixed+sampled* and *derived* parameters.

Actually, this needs not be the case in general, e.g. one may want to fix one or more likelihood arguments to a function of the value of a sampled parameter, or sample from some function or scaling of a likelihood argument, instead of from the likelihood argument directly. The **reparametrization layers** allow us to specify this non-trivial behaviour at run-time (i.e. in the *input*), instead of  having to change the likelihood code to make it understand different parametrizations or impose certain conditions as fixed input arguments.

In general, we would distinguish between two different reparametrization blocks:

* The **in** block: :math:`f(\text{fixed and sampled params})\,\Longrightarrow \text{input args}`.
* The **out** block: :math:`f(\text{output [and maybe input] args})\,\Longrightarrow \text{derived params}`.

.. note::
   In the **out** block, we can specify the derived parameters as a function of the output parameters and *either* the fixed+sampled parameters (pre-**in** block) or the input arguments (post-**in** block). We choose the **post** case, because it looks more consistent, since it does not mix likelihood arguments with sampler parameters.

Let us look first at the **in** case, in particular at its specification in the input. As an example, let us assume that we want to sample the log of a likelihood argument :math:`x`.

In principle, we would have to specify in one block our statistical parameters, and, in a completely separate block, the input arguments as a series of functions of the fixed and sampled parameters. In our example:

.. code:: yaml

   params:
     logx:
       prior: ...  # whatever prior, over logx, not x!
       ref: ...    # whatever reference pdf, over logx, not x!

   arguments:
     x: lambda logx: numpy.exp(logx)

This is a little redundant, specially if we want to store :math:`x` also as a derived parameter: it would appear once in the ``params`` block, and again in the ``arguments`` block. Let us *assume* that in almost all cases we communicate trivially with the likelihood using parameter names that it understands, such that the default functions are identities and we only have to specify the non-trivial ones. In that case, it makes sense to specify those functions as **substitutions**, which in out example would look like:

.. code:: yaml

  params:
    logx:
      prior: ...  # whatever prior, over logx, not x!
      ref: ...    # whatever reference pdf, over logx, not x!
      subs:
        x: lambda logx: numpy.exp(logx)

If the correspondences are not one-to-one, because some number of statistical parameters specify a *larger* number of input arguments, we can create additional **fixed** parameters to account for the extra input arguments. E.g. if a statistical parameter :math:`y` (not understood by the likelihood) defines two arguments (understood by the likelihood), :math:`u=2y` and :math:`v=3y`, we could do:

.. code:: yaml

  params:
    y:
      prior: ...  # whatever prior, over y
      subs:
        u: lambda y: 2*y
    v: lambda y: 3*y

or even better (clearer input), change the prior so that only arguments known by the likelihood are explicit:

.. code:: yaml

   params:
     u:
       prior: ...  # on u, *transformed* from prior of y
     v: lambda u: 3/2*u

.. note::

  The arguments of the functions defining the *understood* arguments should be statistical parameters for now. At the point of writing this notes, we have not implemented multi-level dependencies.


Now, for the **out** reparametrization.

First, notice that if derived parameters which are given by a function were just specified by assigning them that function, they would look exactly like the fixed, function-valued parameters above, e.g. :math:`v` in the last example. We need to distinguish them from input parameters. Notice that an assignment looks more like how a fixed parameter would be specified, so we will reserve that notation for those (also, derived parameters may contain other sub-fields, such as a *range*, which are incompatible with a pure assignment). Thus, we will specify function-valued derived parameters with the key ``derived``, to which said function is assigned. E.g. if we want to sampling :math:`x` and store :math:`x^2` along the way, we would input

.. code:: yaml

   params:
     x:
       prior: ...  # whatever prior for x
     x2:
       derived: lambda x: x**2
       min: ...  # optional


As in the **in** case, for now we avoid multilevel dependencies, by making derived parameters functions of input and output arguments only, not of other derived parameters.

Notice that if a non trivial reparametrization layer is present, we need to change the way we check at initialization that the likelihoods undestand the parameters specified in the input: now, the list of parameters to check will include the fixed and sampled parameters, but applying the **substitutions** given by the ``subs`` fields. Also, since derived parameters may depend on output arguments that are not explicitly requested (i.e. only appear as arguments of the function defining the derived parameters), one needs to check that the likelihood understands both the derived parameters which are **not** specified by a function, and the **arguments** of the functions specifying derived parameters, whenever those arguments are not input arguments.

.. note::

   In the current implementation, if we want to store as a derived parameter a fixed parameter that is specified through a function, the only way to do it is to defined an additional derived parameter which is trivially equal to the fixed one. In the :math:`u,\,v` example above, if we would want to store the value of :math:`v` (fixed) we would create a copy of it, :math:`V`:

   .. code:: yaml

      params:
        u:
          prior: ...  # *transformed* from prior of y
        v: lambda u: 3/2*u
        V:
          derived: lambda v: v


About the ``theory`` module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many physical applications, the Bayesian model can be separated in to a theoretical part :math:`\mathcal{T}` with parameters :math:`\tau`, and an experimental part :math:`\mathcal{E}` with parameters :math:`\epsilon`. In turn, the likelihood :math:`\mathcal{L}[\mathcal{D}|\mathcal{E}(\epsilon),\mathcal{T}(\tau)]` can be written in terms of an intermediate quantity, the *observable* :math:`\mathcal{O}` that contains all the dependence on the theoretical model, and which is the input of an *experimental likelihood* :math:`\mathcal{L}_\mathrm{exp}[\mathcal{D}|\mathcal{E}(\epsilon),\mathcal{O}]`, which does not care about which model was used to compute the observable:

.. math::

   \mathcal{L}[\mathcal{D}|\mathcal{E}(\epsilon),\mathcal{T}(\tau)] =
   \mathcal{L}_\mathrm{exp}[\mathcal{D}|\mathcal{E}(\epsilon),\mathcal{O}]
   \quad\text{with}\quad
   \mathcal{O}[\mathcal{T}(\tau)]

It is also common that more than one experimental likelihood make use of the same observables, or of elements of a set of observables, all of them computed with the same *theory code*. This is the code that we wrap in the ``theory`` module.

Since the sampling process should not necessarily care about this particular physical aspect of the problem, the ``theory`` module, which is optional, belongs into the collection of likelihoods :class:`likelihood.LikelihoodCollection`.

Since we normally expect to heavily modify the theoretical models, introducing new parameters and priors, we are not defining default parameter sets for theory codes in the respective ``defaults.yaml`` file, to avoid having to modify the theory wrapper if we introduce a new theory parameter. This means that the parameters of the theory must be **identifiable** in some other way:

a) Listed at the same level as the likelihood parameters, and identified as the parameters not understood by any likelihood.
b) Defined in a separate block or with a separate prefix, such as ``theory__[name]``.

We choose option **(b)**. The separation is natural within the context of the distinction between theoretical and experimental model, as explained above.


------------------------------------------------------------------------------------


TODO from here on!!!
--------------------

Towards `models` instead of `codes`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ideally, one would define models, not codes: in models, it is natural to define default parameter priors (and actually desirable!), and one could use *inheritance* for extending the model.

Models would have their own ``defaults.yaml`` file. All codes considered compatible with that model must understand all of the parameters defined there, and have methods to set them and get them (ideally their names are not mentioned in the code wrapper, but are passed to or retrieved from the code transparently, so if we extend/inherit the model and modify the theory code to add a new parameter, we don't need to edit the wrapper of the theory code accordingly).

The distinction between theory and likelihood also has a consequence here. As, in practise, we may use different codes to compute the observables given by the same theoretical model, models and codes are separated in the source.

* **Models** are defined by a yaml file containing parameters (in the sense described above, as opposed to options.
* **Codes** are defined as a python object with methods for initializing, computing and closing, and also `get`-methods for the observables that the likelihoods may request.

The model to be inherited is mentioned in the `theory` block. Its parameters, fixed, sampled and derived, are automatically added to the `params` block internally.

If one wants to sample a modification of a code, simply state the differences (a fixed parameter with a different value or sampled by default, a new derived parameter, a different prior...) inside the `params` block, and everything indicated there takes precedence over the inherited model.

.. todo::

   NB: we may want to include code-specific options in a model, e.g. to ensure precision. They would be something like `camb__accuracy: 2`, where the double underscore would serve as a separator between the name of the code to be passes to, and the name of the parameter.

.. code-block:: yaml

   theory:
     model: lcdm_planck
       code: camb
       path: /path/to/camb
       option_1: value_1
       option_2: value_2
       ...

   likelihood:
     experiment_1:
       option_1: value_1
     experiment_2:
       ...

   params:
     # Directly passed to the theory module (RESERVED PARAMETER NAME!)
     theory:
       # fixed: (args)
       f_1: 1
       ...
       # sampled: (params)
       p_1:
         prior:
           ...
       ...
       # derived:
       d_1:
    # Parameters passed to the likelihoods
    likelihood:
      # fixed: (args)
      f_1: 1
      ...
      # sampled: (params)
      p_1:
        prior:
          ...
      ...
      # derived:
      d_1:
      ...

