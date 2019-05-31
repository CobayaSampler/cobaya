Importance reweighting and general ``post``-processing
======================================================

The `post` module provides a way to post-process an existing sample in different ways:

- Add/remove/recompute a prior, e.g. to impose a parameter cut.
- Add/remove/recompute a likelihood.
- Force the recomputation of the theory code and (optionally) things that depend on it, e.g. with more precision.
- Add/remove/recompute a derived parameter.

The requested operations are detailed in a ``post`` block, which may contain one ``add`` and one ``remove`` sub-block. Under ``add``, *using standard input syntaxis*, specify what you want to add duting preprocessing, and under ``remove`` what you would like to remove (no options necessary for removed stuff: just a mention). To force an update of a prior/likelihood/derived-parameter, include it both under ``remove`` and ``add``, with the new options, if needed insinde ``add``.

The input sample is specified via the ``output`` option with the same value as the original sample. Cobaya will look for it and check that it is compatible with the requested operations. If multiple samples are found (e.g. from an MPI run), all of them are loaded and concatenated. The resulting sample will have a suffix ``_post_[your_suffix]``, where ``[your_suffix]`` is specified with the ``suffix`` option of ``post``.

.. note::

   In a scripted call, instead, specify the initial sample as ....

EXAMPLE
- add/remove likelihood or multidim prior
- Change 1d prior (limits)

NOTE:
- New pdf must be absolutely continuous over the old one (it's the user's responsibility to ensure that!)


Interaction with theory codes
-----------------------------

OJO: si cambio theory, solo se recomputan likes que estan del+add.
Ventaja: usa el sistema implícito de las likes para decidir que computar.
Desventaja: depende del usuario asegurarse de qué hay que recomputar:
si cambio theory y recomputo low-ell pero no high-ell, estoy metiendo la pata!

NO NEED TO RE-SPECIFY THEORY IF THERE WAS ONE!!!

EXAMPLE!!!!

- [ ] Current limitation: the user is responsible for tracking dependencies: theory recomputation only updates removed+added likes and derived params; also, dyamic derived params that may depend on recomputed ones will not changed unless explicitly removed+added (notice for partial chi2 sums in cosmo cases) -- could add a warning in the code, but detecting it is the hardest part: could as well fix it!


Ignoring burn-in and thinning the sample
----------------------------------------

You can **skip** any number of initial samples using the option ``skip``, with an integer value for a precise number of rows, and and a value :math:`<1` for an initial fraction of the chain.

To **thin** the sample, give the ``thin`` option any value :math:`>1`, and only one every ``[thin]`` samples will be used.


Sequential application of post-processing
-----------------------------------------

.. warning::

   This is still WIP, sorry!
