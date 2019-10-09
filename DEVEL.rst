Development notes
==================

This document gathers some notes about the development flow, release checklist, and general design decisions and their motivation. If you intend to heavily modify Cobaya, we recommend you to read it first.


``git`` development model
-------------------------

We (*tend to*) use the model described `here <https://barro.github.io/2016/02/a-succesful-git-branching-model-considered-harmful/>`_:

* Development happens in the master branch (only exceptionally are features developed in their own branches).
* Releases are branched out, and only critical bug fixes are pushed onto them.


Development flow for contributors
---------------------------------

.. note::

   WIP!

1. Fork and clone the repo from github.
2. From its folder, install in editable mode: ``pip install -e . --user``
3. Modify stuff.
4. Test with pytest.
5. Pull requests, etc.

Contributors must agree to the license (see ``LICENCE.txt`` in the root folder).


Release checklist
-----------------

+ Make sure all test pass in Travis (or the package won't be pushed to PyPI).
+ Make sure everything relevant has been added to the Changelog.
+ Delete old deprecation notices (>=2 versions before)
+ Bump version number in ``__init__.py`` and ``CHANGELOG.md`` (also date)
+ If archived version:
  - make ``__obsolete__ = True`` in ``__init__.py``
  - Fix CAMB's version to latest relase (right now, it installs ``master`` by default)
+ Update year of copyright in ``__init__.py``.
+ Update year of copyright in ``LICENCE.txt``.
+ Commit + tag with new version + ``git push`` + ``git push --tags``
+ If needed, re-build the documentation.
+ If applicable, delete branches merged for this version.
+ Notify via the e-mail list.


Notes on some design choices
----------------------------

Generality and modularity
^^^^^^^^^^^^^^^^^^^^^^^^^

This code is being developed as a general-purpose sampler, with a Bayesian focus. The different aspects of the sampling, namely prior, likelihood and sampler, are kept as isolated as possible: the prior and sampler know which parameters are sampled and fixed, but not the likelihood, which does not need to know; the sampler does now know which likelihood understands which parameters, since it does not care (just cares about their respective priors, speeds, etc). This designs choices take some compromises, but are a fair price for making the code more easily extendable and maintainable: e.g. adding a new likelihood on the fly, or modify their parameters, needs no modification of the main code.

The cosmology only enters through particular likelihood and theory modules, and the main source does not contain significant cosmological code or experimental data, just wrappers, so that the general user does not need to download gigabytes of data.

Ideally, in the near future, the source will be split in two: a general sampler package on one side, and the cosmological modules on the other.

.. image:: ./img/diagram.svg
   :align: center
   :width: 60%


Dealing with parameters
^^^^^^^^^^^^^^^^^^^^^^^

Parameter roles
"""""""""""""""

Parameters have different roles with respect to different parts of the code:

- The :class:`sampler.Sampler` cares about whether parameters are **fixed** (thus irrelevant), **sampled** over, or **derived** from sampled and fixed parameters. The :class:`prior.Prior` cares about **sampled** parameters only.
- The :class:`likelihood.Likelihood` and the :class:`theory.Theory` care about whether parameters are to be taken as **input**, or are expected to be part of their **output**.

The :class:`parameterization.Parameterization` class (see diagram) takes care of interfacing between these two sets of roles, which, as it can be seen below, is sometimes not as simple as ``sampled + fixed = input``, and ``derived = output``.

.. warning::

   Despite generating some ambiguity, we call output parameters sometimes also *derived*, when it is clear that we are in the likelihood context, not the sampler context.


How likelihoods and theory decide which input/output parameters go where
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Once the :class:`parameterization.Parameterization` has decided which are the **input** and **output** parameters, the :class:`likelihood.LikelihoodCollection` needs to decide how to distribute them between the likelihoods.

The simplest way to do that would be tagging each parameter with its corresponding likelihood(s) or theory, but this would make the input much more verbose and does not add much. Alternatively we could hard-code parameter routes for known parameters (e.g. for cosmological models), but hard-coding parameter names impose having to edit Cobaya's source if we want to modify a theory code or likelihood to add a new parameter, and we definitely want to avoid people having to edit Cobaya's source (maintainability, easier support, etc).

So, in order not to have tag parameters or hard-code their routes, the only option left is that each likelihood and theory can tell us which parameters it understands. There are a number of possible ways a likelihood or theory could do that:

- If it is defined as a Python function (an *external* likelihood, in our terminology), we can use *introspection* to get the possible arguments. Introspection for output parameters is a little more complicated (see note below).
- For *internal* likelihoods and theories (i.e. more complex classes that allow more flexibility and that have no function to inspect), we need either:

  + to keep a *list* of possible input/output parameters
  + to define a *rule* (e.g. a prefix) that allows us to pick the right ones from a larger set

- Finally, if there is a likelihood or theory that cannot be asked and does not keep a list of parameters, that would not necessarily be a problem, but we would have to choose between passing it either all of the parameters, **or** just those that have not been claimed by anyone else (in this last case, there could obviously be *only one* likelihood or theory in the collection with this property).

.. note::

   For callable (*external*) likelihoods, output parameters cannot be simple **keyword** arguments, since in Python parameter values (``float``'s) are *immutable*: they are passed by value, not by reference, so their value cannot be *modified back*. Thus, we interface them via a dictionary passed through a ``_output`` keyword argument. Since dictionaries are *mutable* objects, when their contents are modified the modifications are permanent, which makes a natural way of dealing with derived parameters on the same ground as sampled parameters. At function definition, we assign this keyword argument a list of possible keys, which we can get, via *introspection*, as the list of output parameters understood by that likelihood. 

We should also take into account the following:

- Different likelihoods may share part of the same model, so they may have input parameters in common (but not output parameters; or if they do, we still only need to compute them once).
- Some likelihoods may not take any input parameter at all, but simply get an observable through their interface with a theory code.
- Some parameters may be both input and output, e.g. when only a subset of them can determine the value of the rest of them; e.g. a likelihood may depend on ``a`` and ``b``, but we may want to expose ``a+b`` too, so that the user can choose any two of the three as input, and the other one as output.
- External functions may have a variable number of input parameters, since some may be represented by keyword arguments with a default value, and would thus be optional.

To implement these behaviours, we have taken the following design choices:

- Two parameters with the same name are considered by default to be the same parameter. Thus, when defining custom likelihoods or creating new interfaces for external likelihoods, use preferably non-trivial names, e.g. instead of ``A``, use ``amplitude``, or even better, ``amplitude_of_something``. (The case of two likelihoods naming two *different* parameter the same is still an open problem: we could defined two parameters prefixed with the name of the likelihood, and have the :class:`likelihood.LikelihoodCollection` deal with those cases; or we could define some dynamical renaming.)
- If a likelihood or theory does not specify a parameter set/criterion and it is not the only element in the collection, we pass it only the parameters which have *not been claimed* by any other element.
- Theory codes are special in the sense that they may understand a very large number of input/output parameters, so they will often be the "no knowledge" kind. On the other hand, they should **not** share parameters with the likelihoods: if the likelihoods do depend on any theoretical model parameter, they should request it via the same interface the theory-computed observables are, so that the parameterization of the theoretical model can be changed without changing the parameterization of the likelihoods (e.g. a SN likelihood may require the Hubble constant today, but if it where an input parameter of the likelihood, it would be more complicated to choose an alternative parameterization for the theoretical model e.g. some standard ruler plus some matter content).
- Since theories an likelihoods do not share parameters, we choose that when theories mark parameters for themselves, they **absorb** them, so that they are ignored by all other parts of the code.
- Given the ambiguity between input and output roles for particular parameters, *internal* likelihoods that would keep a list known parameters can do so in two ways:

  + The preferred one: a common list of all possible parameters in a ``params`` block in the defaults file; but **outside** the defaults block, so that it can be copied literally into an input file. There, parameters would appear with their **default** role. This has the advantage that priors, labels, etc can be inherited at initialisation from these definitions (though the definitions in the user-provided input file would take precedence). If there is a conflict between the priors (or fixed value, or derived state) for *the same parameter* defined in different defaults files of likelihoods that share it, an error will be produced (unless the user settles the conflict by specifying the desired behaviour for said parameter in the input file).
  + Alternatively (and preferred when there is a conflict), they could keep two lists: one of input and one of output parameters.

- It may be that the likelihood does not depend on (i.e. has constraining power over) a particular parameter(s). In that case, we still throw an error if some input parameter has not been recognised by any likelihood, since parameter names may have been misspelled somewhere, and it is easier to define a mock likelihood to absorb the unused ones than maybe finding a warning about unused parameters (or use the unit likelihood described below).
- Some times we are not interested in the likelihood, because we want to explore just the prior, or the distribution the prior induces on a derived parameter. In those cases, we would need a mock unit likelihood. This unit likelihood would automatically recognise all input parameters (except those absorbed by the theory, if a theory is needed to compute derived parameters).
- For external likelihoods, where we can get input and output parameters via introspection, we may not want to use all of the input ones, as stated above, since they may have a fixed default value as keyword arguments. This would be treated as a special case of having a list of input parameters.

Given these principles, we implement the following algorithm to resolve input/output parameter dependencies: (in the following, "likelihoods" include the theory code)

0. Start with a dictionary of input parameters as keys, and another one for output parameters. The values will be a list of the likelihoods that depend on each parameter.
1. Iterate over likelihoods that have knowledge of their own parameters, either because they are *callable*, or because they have input/output parameters lists, a prefix, or a mixed ``params`` list, *in that order of priority*. Add them to the lists in the initial parameters dictionaries if applicable.
2. Deal with the case (check that it is only one) of a likelihood with no parameters declared, and assign it all unclaimed parameters.
3. Check that, if a theory code is present, it does not share parameters with anyone else.
4. If the unit likelihood is present, assign it all input parameters (except those absorbed by the theory code).
5. Check that there are no unclaimed input/output parameters, and no output parameters with more than one claim.

Whether this algorithm runs before of after the initialisation of the likelihoods depends on whether likelihoods do have knowledge about their parameters before being initialised or not. They do not in the case of callable (*external*) likelihoods, so we would prefer to initialise the likelihoods before assigning them parameters. But, on the other hand, some likelihood may perform checks on the parameters assigned at initialisation. This leaves us with two options:

a) Initialise callable likelihoods first, then assign parameters, and finally initialise the rest.
b) Initialise all likelihoods but without performing any test on parameters, then assign parameters, and finally call a "post-initialisation" method that does the parameter checks.

Option (a) is a little more complex from the coding point of view, whereas option (b) makes the interfaces to the likelihoods a little more complicated. Since this implementation does not affect the API much (just when creating new likelihoods), it does not matter too much. We pick option (b) and define an optional post-``__init__`` :meth:`likelihood.Likelihood.initialize` method for *internal* likelihoods, to be called after parameter assignment. If, in the future, there are *internal* likelihoods that need some initialisation to determine its known parameters in some dynamical way (other than having a list of input/output/mixed parameters), we can redefine their ``__init__``, making sure that we call that of the parent class first of all with a ``super``.

After parameters have been assigned, we save the assignments in the updated (*full*) info using the unambiguous "input/output lists" option, for future use by e.g. post-processing: during post-processing, unused likelihoods are not initialised, in case they do not exist any more (e.g. an external function), but we still need to know on which parameters it depended.


Special considerations for output parameters
""""""""""""""""""""""""""""""""""""""""""""

Computing output parameters may be expensive, and we won't need them for samples that are not going to be stored (e.g. they are rejected, only used just to perform *fast-dragging*, or just to train a machine-learning model). Thus, their computation must be **optional**.

But in general, one needs the current *state* (value of the input parameters) to compute the output ones. Thus, if the state is potentially an interesting one for the sampler, we will have to get the output parameters immediately after the likelihood computation (otherwise, if we have jumped somewhere else and then decided to get them, we may have to re-compute the likelihood at the point of interest, which is probably more costly than having computed output parameters that we are likely to throw away). It is up the each sampler to decide whether the output parameters (of, for the sampler, the derived parameters) at one particular sample are worth computing.

.. note::

   Unfortunately, for many samplers, such as basic MH-MCMC, we do not know a priori if we are going to save a particular point, so we are forced to compute derived parameters even when they are not necessary. In those case, if their computation is prohibitively expensive, it may be faster to run the sample without derived parameters, and add them after the sampling process is finished.

We could implement the way likelihoods and theory communicate output parameters to the model in two ways:

a) A keyword option in the log-likelihood (or theory's ``compute``) method to request the computation of output parameters (passed back as a mutable argument of that same function).
b) An optional method of the :class:`Likelihood` class, say ``get_output_parameters``, that is called only if needed.

Since the method in option (b) would *always* have to be called *immediately* after computing the likelihood (or otherwise risk inadvertently changing the state and getting the wrong set of set of output parameters), we adopt option (a).

So we create, for the ``log-likelihood`` method of a :class:`Likelihood` and the ``compute`` method of a theory, a keyword argument ``_output``. If that keyword is valued ``None``, the output parameters will not be computed, and if valued as an empty dictionary, it will be used to return the output parameters (thanks to Python's passing mutable objects by reference, not value).

From the sampler point of view (now of *derived*, not *output*, parameters), we use a list, not a dictionary, as an argument of the log-likelihood of the :class:`LikelihoodCollection`; we do so (dropping parameter names by using a list) so that the sampler does not need to keep track of the names of the derived parameters.


Dynamical reparameterization layer (a bit old!)
"""""""""""""""""""""""""""""""""""""""""""""""

As stated above, parameters are specified according to their roles for the **sampler**: as *fixed*, *sampled* and *derived*. On the other hand, the **likelihood** (and the **theory code**, if present) cares only about input and output arguments. In a trivial case, those would correspond respectively to *fixed+sampled* and *derived* parameters.

Actually, this needs not be the case in general, e.g. one may want to fix one or more likelihood arguments to a function of the value of a sampled parameter, or sample from some function or scaling of a likelihood argument, instead of from the likelihood argument directly. The **reparameterization layer** allow us to specify this non-trivial behaviour at run-time (i.e. in the *input*), instead of  having to change the likelihood code to make it understand different parameterizations or impose certain conditions as fixed input arguments.

In general, we would distinguish between two different reparameterization blocks:

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


Now, for the **out** reparameterization.

First, notice that if derived parameters which are given by a function were just specified by assigning them that function, they would look exactly like the fixed, function-valued parameters above, e.g. :math:`v` in the last example. We need to distinguish them from input parameters. Notice that an assignment looks more like how a fixed parameter would be specified, so we will reserve that notation for those (also, derived parameters may contain other sub-fields, such as a *range*, which are incompatible with a pure assignment). Thus, we will specify function-valued derived parameters with the key ``derived``, to which said function is assigned. E.g. if we want to sampling :math:`x` and store :math:`x^2` along the way, we would input

.. code:: yaml

   params:
     x:
       prior: ...  # whatever prior for x
     x2:
       derived: lambda x: x**2
       min: ...  # optional


As in the **in** case, for now we avoid multilevel dependencies, by making derived parameters functions of input and output arguments only, not of other derived parameters.

Notice that if a non trivial reparameterization layer is present, we need to change the way we check at initialisation that the likelihoods understand the parameters specified in the input: now, the list of parameters to check will include the fixed and sampled parameters, but applying the **substitutions** given by the ``subs`` fields. Also, since derived parameters may depend on output arguments that are not explicitly requested (i.e. only appear as arguments of the function defining the derived parameters), one needs to check that the likelihood understands both the derived parameters which are **not** specified by a function, and the **arguments** of the functions specifying derived parameters, whenever those arguments are not input arguments.

.. note::

   In the current implementation, if we want to store as a derived parameter a fixed parameter that is specified through a function, the only way to do it is to defined an additional derived parameter which is trivially equal to the fixed one. In the :math:`u,\,v` example above, if we would want to store the value of :math:`v` (fixed) we would create a copy of it, :math:`V`:

   .. code:: yaml

      params:
        u:
          prior: ...  # *transformed* from prior of y
        v: lambda u: 3/2*u
        V:
          derived: lambda v: v

