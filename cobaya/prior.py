"""
.. module:: prior

:Synopsis: Class containing the prior and reference pdf, and other parameter information.
:Author: Jesus Torrado


Basic parameter specification
-----------------------------

The ``params`` block contains all the information intrinsic
to the parameters of the model: their prior pdf and reference pdf, their latex labels,
and useful properties for particular samplers (e.g. width of a proposal pdf in and MCMC
-- see the documentation for each sampler).
The prior and reference pdf are managed by the :class:`Prior` class.

You can specify three different kinds of parameters:

+ **Fixed** parameters are specified by assigning them a value, and are passed directly to
  the likelihood or theory code.
+ **Sampled** parameters need a ``prior`` pdf definition and, optionally,
  a reference pdf (``ref``),
  a LaTeX label to be used when producing plots,
  and additional properties to aid particular samplers.
+ **Derived** parameters do **not** have a prior definition
  (neither a reference value or pdf),
  but can have a LaTeX label and a ``min`` and/or ``max``
  to be used in the sample analysis
  (e.g. to guess the pdf tails correctly if the derived quantity needs to be positive or
  negative -- defaulted to ``-inf``, ``inf`` resp.).

The (optional) **reference** pdf (``ref``) for **sampled** parameters
defines the region of the prior which is of most
interest (e.g. where most of the prior mass is expected);
samplers needing an initial point
for a chain will attempt to draw it from the ``ref`` if it has been defined (otherwise
from the prior). A good reference pdf will avoid a long *burn-in* stage during the
sampling. If you assign a single value to ``ref``, samplers will always start from
that value.


The syntax for priors and ref's has the following fields:

+ ``dist`` (default: ``uniform``): any 1-dimensional continuous distribution from
  `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions>`_;
  e.g. ``uniform``, ``[log]norm`` for a [log-]gaussian,
  or ``halfnorm`` for a half-gaussian.
+ ``loc`` and ``scale`` (default: 0 and 1, resp.): the *location* and *scale* of the pdf,
  as they are understood for each particular pdf in :class:`scipy.stats`; e.g. for a
  ``uniform`` pdf, ``loc`` is the lower bound and ``scale`` is the length of the domain,
  whereas in a gaussian (``norm``) ``loc`` is the mean and ``scale`` is the standard
  deviation).
+ Additional specific parameters of the distribution, e.g. ``a`` and ``b`` as the powers
  of a Beta pdf.

.. note::

   For bound distributions (e.g. ``uniform``, ``beta``...), you can use the more
   intuitive arguments ``min`` and ``max`` (default: 0 and 1 resp.) instead of ``loc`` and
   ``scale`` (NB: unexpected behaviour for an unbounded pdf).


The order of the parameters is conserved in the table of samples, except the fact that
derived parameters are always moved to the last places.

An example ``params`` block:

.. code-block :: yaml

     params:

       A: 1 # fixed!

       B1:  # sampled! with uniform prior on [-1,1] and ref pdf N(mu=0,sigma=0.25)
         prior:
           min: -1
           max:  1
         ref:
           dist: norm
           loc:   0
           scale: 0.25
         latex: \mathcal{B}_2
         proposal: 0.25

       B2:  # sampled! with prior N(mu=0.5,sigma=3) and fixed reference value 1
         prior:
           dist: norm
           loc: 0.5
           scale: 3
         ref: 1  # fixed reference value: all chains start here!
         latex: \mathcal{B}_2
         proposal: 0.5

       C:  # derived!
         min: 0
         latex: \mathcal{C}

You can find another basic example :ref:`here <example_quickstart_shell>`.


.. _prior_external:

Multidimensional priors and custom functions
--------------------------------------------

The priors described above are obviously 1-dimensional.
You can also define more complicated, multidimensional priors
using a ``prior`` block at the base level (i.e. not indented), as in the
:ref:`the advanced example <example_advanced_shell>`. All the details below also apply to
the definition of :ref:`custom likelihoods <likelihood_external>`.

Inside the ``prior`` block, list a pair of priors as ``[name]: [function]``, where the
functions must return **log**-priors. This priors will be multiplied by the
one-dimensional ones defined above. Even if you define a custom prior for a parameter
in the ``prior`` block, you still have to specify its bounds in the ``params`` block.

A prior function can be specified in two different ways:

a) **As a function**, either assigning a function defined with ``def``, or a ``lambda``.
   Its arguments must be known parameter names. This option is not compatible with shell
   invocation, since the ``yaml`` output cannot store python objects.

b) **As a string,** which will be passed to ``eval()``. The string can be a
   ``lambda`` function definition with known parameters as its arguments (you can
   use ``scipy.stats`` and  ``numpy`` as ``stats`` and ``np`` resp.), or an
   ``import_module('filename').function`` statement if your function is defined in a
   separate file ``filename`` `in the current folder` and is named ``function``
   (see e.g. :ref:`the advanced example <example_advanced_shell>`).

.. note::

   If you don't want to write your function definition in a separate file and don't care
   about reproducibility of the run (you should!), there is a workaround: add
   ``force_reproducible: False`` at the top of your input.

.. note::

   At this moment, custom priors can only be functions of sampled and fixed parameters.
   Extending it to derived parameters is WIP.

.. note::

   In your function definition, you can use keyword arguments that aren't known parameter
   names. They will simply be left to their default.


Prior normalization for evidence computation
--------------------------------------------

The one-dimensional priors defined within the ``params`` block are automatically
normalized, so any sampler that computes the evidence will produce the right results as
long as no custom priors have been defined, whose normalisation is unknown.

To get the prior normalisation if using custom functions as priors, you can substitute
your likelihood by the :doc:`dummy unit likelihood <likelihood_one>`, and make an initial
run with :doc:`PolyChord <sampler_polychord>` to get the prior volume.


In general, avoid improper priors, since they will produce unexpected errors, e.g.

.. code-block:: yaml

    params:
      a:
        prior:
          min: -inf
          max:  inf

.. _repar:

Defining parameters dynamically
-------------------------------

We may want to sample in a paramter space different than the one understood by the
likelihood, e.g. because we expect the posterior to be simpler on the alternative
parameters.

For instance, in the :doc:`advanced example <example_advanced>`, the posterior on the
radius and the angle is a gaussian times a uniform, instead of a more complicated
gaussian ring. This is done in a simple way at
:ref:`the end of the example <example_advanced_rtheta>`.
Let us discuss the general case here.

To enble this, **cobaya** creates a `re-parametrisation` layer between the `sampled`
parameters, and the `input` paramters of the likelihood. E.g. if we want to **sample**
from the logarithm of an **input** parameter of the likelihood, we would do:

.. code:: yaml

    params:
      logx:
        prior: [...]
        drop: True
      x: "lambda logx: np.exp(x)"
      y:
        derived: "lambda x: x**2"

When that information is loaded **cobaya** will create an interface between two sets of
parameters:

+ **Sampled** parameters, i.e. those that have a prior, including ``logx`` but not ``x``
  (it does not have a prior: it's fixed by a function). Those are the ones with which the
  **sampler** interacts. Since the likelihood does not understand them, they must be
  **dropped** with ``drop: True``.

+ Likelihood **input** parameters: those that will be passed to the likelihood
  (or theory), indentified because they are **not derived** (see below) and have not been
  **dropped**. Here, that would be ``x``, but not ``logx``.

We can use a similar approach to define dynamical **derived** paramters. To distinguish
them from the input paramters, we insert the functions defining them under a ``derived``
property (see the paramater ``y`` in the example above).


.. note::

   For now, **dynamical derived** parameters must be functions of **input** but not
   **dynamical sampled** parameters. They can also be functions of yet-undefined params.
   In that case, those parameters will be automatically requested to the likelihood and
   theory code.

.. note::

   For now, the values for the samples of input parameters which are not directly sampled
   (e.g. ``x`` above) are not saved into the output products. If you want to track their
   value, create new dummy derived parameter for each one, e.g.
   ``xprime: {derived: "lamdba x: x"}``

.. note::

   There is only **one level** of re-parametrisation, i.e. a dynamically-defined parameter
   cannot be a function of another dynamically-defined parameter: ``y=f(x)`` and
   ``z=g(y)`` is not allowed. This is so for the sake
   of keeping **cobaya**'s overhead as small as possible, and because it would complicate
   the task of finding the `owner` of each parameter. But this may change in the future.

   For now, a workaround is repeating the definition of one of the dynamical parameters
   inside the definition of the other, so that both depend on the same original parameter,
   i.e. instead of ``y=f(x)`` and ``z=g(y)``, define ``y=f(x)`` and
   ``z=g[f(xx)]``.

.. note::

   It is not possible to **fix** the value of a dynamically defined parameter
   (there would be no way to indicate that we want to drop it). Fix the corresponding
   likelihood parameters instead.
   [But let us know if this is a feature that you'd really, really like.]


.. _prior_inheritance:

Changing and redefining parameters; inheritance
-----------------------------------------------

As we will see in the :doc:`likelihoods documentation <likelihoods>`, just by mentioning a
likelihood, the parameters of its default experimental model are
inherited without needing to re-state them.

But you may want to fix their value, re-define their prior or some other property in
your input file. To do that, simply state you new parameter definition as you would
normally do. For your convenience, if you do that, you don't need to re-state all of the
properties of the parameters (e.g. the latex label if just changing the prior).

As a general rule, when trying to redefine anything in a parameter, everything not
re-defined is inherited, except for the prior, which must not be inheritable in order to
allow re-defining sampled parameters into something else.

Just give it a try and it should work fine, but, in case you need the details:

* Re-defining **fixed** into:

 - **fixed**: simply assign it a new value
 - **sampled**: define a prior for it (and optionally reference, label, etc.)
 - **derived**: mention the parameter and assign nothing to it (you may define a
   min, max or latex label, but *not a prior*)

* Re-defining **sampled** into:

 - **fixed**: simply assign it a value
 - **sampled**: change any of prior/ref/latex, etc, to your taste; the rest are inherited
   from the defaults
 - **derived**: mention the parameter and assign nothing to it (you may define a
   min, max or latex label, but *not a prior*; the label is inherited from the defaults)

* Re-defining **derived** into:

 - **fixed**: simply assign it a value
 - **sampled**: define a prior for it (and optionally reference, label, etc.)
 - **derived**: change any of prior/ref/latex, etc, to your taste; the rest are inherited
   from the defaults

"""


# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
from collections import OrderedDict as odict
import numpy as np
import numbers
import inspect
from copy import deepcopy

# Local
from cobaya.conventions import _prior, _p_ref
from cobaya.tools import get_external_function, get_scipy_1d_pdf
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__.split(".")[-1])


class Prior(object):
    """
    Class managing the prior and reference pdf's.
    """
    def __init__(self, parametrization, info_prior=None):
        """
        Initialises the prior and reference pdf's from the input information.
        """
        fixed_params_info = parametrization.fixed_params()
        sampled_params_info = parametrization.sampled_params()
        if not sampled_params_info:
            log.warning("No sampled parameters requested! "
                        "This will fail for non-mock samplers.")
        # pdf: a list of independent components
        # in principle, separable: one per parameter
        self.name = []
        self.pdf = []
        self.ref_pdf = []
        self._bounds = np.zeros((len(sampled_params_info), 2))
        for i, p in enumerate(sampled_params_info):
            self.name += [p]
            prior = sampled_params_info[p].get(_prior)
            self.pdf += [get_scipy_1d_pdf({p: prior})]
            # Get the reference (1d) pdf
            ref = sampled_params_info[p].get(_p_ref)
            # Cases: number, pdf (something, but not a number), nothing
            if isinstance(ref, numbers.Number):
                self.ref_pdf += [float(ref)]
            elif ref is not None:
                self.ref_pdf += [get_scipy_1d_pdf({p: ref})]
            else:
                self.ref_pdf += [np.nan]
            self._bounds[i] = [-np.inf, np.inf]
            try:
                self._bounds[i] = self.pdf[-1].interval(1)
            except AttributeError:
                log.error("No bounds defined for parameter '%s' "
                          "(maybe not a scipy 1d pdf).", p)
                raise HandledException
        # Process the external prior(s):
        self.external = odict()
        for name in (info_prior if info_prior else {}):
            log.debug("Loading external prior '%s' from: '%s'", name, info_prior[name])
            self.external[name] = (
                {"logp": get_external_function(info_prior[name], name=name)})
            self.external[name]["argspec"] = (
                inspect.getargspec(self.external[name]["logp"]))
            self.external[name]["params"] = {
                p:list(sampled_params_info).index(p)
                for p in self.external[name]["argspec"].args if p in sampled_params_info}
            self.external[name]["fixed_params"] = {
                p:fixed_params_info[p]
                for p in self.external[name]["argspec"].args if p in fixed_params_info}
            if (not (len(self.external[name]["params"]) +
                     len(self.external[name]["fixed_params"]))):
                log.error("None of the arguments of the external prior '%s' "
                          "are known *fixed* or *sampled* parameters. "
                          "This prior recognises: %r",
                          name, self.external[name]["argspec"].args)
                raise HandledException
            params_without_default = self.external[name]["argspec"].args[
                :(len(self.external[name]["argspec"].args) -
                  len(self.external[name]["argspec"].defaults or []))]
            if not all([(p in self.external[name]["params"] or
                         p in self.external[name]["fixed_params"])
                        for p in params_without_default]):
                log.error("Some of the arguments of the external prior '%s' "
                          "cannot be found and don't have a default value either: %s",
                          name, list(set(params_without_default)
                                .difference(self.external[name]["params"])
                                .difference(self.external[name]["fixed_params"])))
                raise HandledException
            log.warning("External prior '%s' loaded. "
                        "Mind that it might not be normalized!", name)

    def d(self):
        """
        Returns:
           Dimensionality of the parameter space.
        """
        return len(self.pdf)

    def bounds(self, confidence_for_unbounded=1):
        """
        For unbounded parameters, if ``confidence_for_unbounded < 1`` given, the
        returned interval contains the requested confidence level interval with equal
        areas around the median.

        Returns:
           An array of bounds ``[min,max]`` for the parameters, in the order given by the
           input.

        NB: If an external prior has been defined, the bounds given in the 'prior'
        sub-block of that particular parameter's info may not be faithful to the
        externally defined prior.
        """
        if confidence_for_unbounded >= 1:
            return self._bounds
        try:
            bounds = deepcopy(self._bounds)
            infs = list(set(np.argwhere(np.isinf(bounds)).T[0]))
            if infs:
                log.warn("There are unbounded parameters. Prior bounds are given at %s "
                         "confidence level. Beware of likelihood modes at the edge of "
                         "the prior", confidence_for_unbounded)
                bounds[infs] = [
                    self.pdf[i].interval(confidence_for_unbounded) for i in infs]
            return bounds
        except AttributeError:
            log.error("Some parameter names (positions %r) have no bounds defined.", infs)
            raise HandledException

    def sample(self, n=1, external_error=True):
        """
        Generates samples from the prior pdf.

        If an external prior has been defined, it is not possible to sample from the prior
        directly. In that case, if you want to sample from the "default" pdf (i.e.
        ignoring the external prior), set `external_error` to `False`.

        Returns:
          Array of ``n`` samples from the prior, as vectors ``[value of param 1, ...]``.
        """
        if external_error and self.external:
            log.error("It is not possible to sample from an external prior.")
            raise HandledException
        return np.array([pdf.rvs(n) for pdf in self.pdf]).T

    def p(self, x):
        """
        Returns:
           The probability density of the given point or array of points.
        """
        return (np.prod([pdf.pdf(xi) for pdf,xi in zip(self.pdf,x)]) *
                np.exp(self.logp_external(x)))

    def logp(self, x):
        """
        Returns:
           The log-probability density of the given point or array of points.
        """
        log.debug("Evaluating prior at %r", x)
        logp = (sum([pdf.logpdf(xi) for pdf,xi in zip(self.pdf,x)]) +
                self.logp_external(x))
        log.debug("Got logprior = %g", logp)
        return logp

    def logp_external(self, x):
        """Evaluates the logprior using the external prior only."""
        return sum([ext["logp"](**dict({p:x[i] for p,i in ext["params"].items()},
                                       **ext["fixed_params"]))
                    for ext in self.external.values()])

    def covmat(self, external_error=True):
        """
        Returns:
           The covariance matrix of the prior.
        """
        if external_error and self.external:
            log.error("It is not possible to get the covariance matrix "
                      "from an external prior.")
            raise HandledException
        return np.diag([pdf.var() for pdf in self.pdf]).T

    def reference(self, max_tries=np.inf):
        """
        Returns:
          One sample from the ref pdf. For those parameters that do not have a ref pdf
          defined, their value is sampled from the prior.

        NB: The way this function works may be a little dangerous:
        if two parameters have an (external)
        joint prior defined and only one of them has a reference pdf, one should
        sample the other from the joint prior *conditioned* to the first one being
        drawn from the reference. Otherwise, one may end up with a point with null
        prior pdf. In any case, it should work for getting initial reference points,
        e.g. for an MCMC chain.
        """
        if np.nan in self.ref_pdf:
            log.info("Reference values or pdf's for some parameters were not provided. "
                     "Sampling from the prior instead for those parameters.")
        while max_tries:
            max_tries -= 1
            ref_sample = np.array([getattr(ref_pdf, "rvs", lambda: ref_pdf.real)()
                                   for i, ref_pdf in enumerate(self.ref_pdf)])
            where_no_ref = np.isnan(ref_sample)
            if np.any(where_no_ref):
                prior_sample = self.sample(external_error=False)[0]
                ref_sample[where_no_ref] = prior_sample[where_no_ref]
            if self.logp(ref_sample) > -np.inf:
                return ref_sample
        log.error("Couldn't sample from the reference pdf a point with non-"
                  "null prior density after '%d' tries. "
                  "Maybe your prior is improper of your reference pdf is "
                  "null-defined in the domain of the prior.", max_tries)
        raise HandledException

    def reference_covmat(self):
        """
        Returns:
          The standard deviation of the 1d ref pdf's. For those parameters that do not
          have a ref pdf defined, the standard deviation of the prior is taken instead.
        """
        covmat = np.diag([getattr(ref_pdf, "var", lambda: np.nan)()
                          for i, ref_pdf in enumerate(self.ref_pdf)])
        where_no_ref = np.isnan(covmat)
        if np.any(where_no_ref):
            log.warn("Reference pdf not defined or improper for some parameters. "
                     "Using prior's sigma instead for them.")
            covmat[where_no_ref] = self.covmat(external_error=False)[where_no_ref]
        return covmat

    # Python magic for the "with" statement

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
