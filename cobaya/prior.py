r"""
.. module:: prior

:Synopsis: Class containing the prior and reference pdf, and other parameter information.
:Author: Jesus Torrado


Basic parameter specification
-----------------------------

The ``params`` block contains all the information intrinsic
to the parameters of the model: their prior pdf and reference pdf, their latex labels,
and useful properties for particular samplers (e.g. width of a proposal pdf in an MCMC
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
  negative -- defaulted to ``-.inf``, ``.inf`` resp.).

The (optional) **reference** pdf (``ref``) for **sampled** parameters
defines the region of the prior which is of most
interest (e.g. where most of the prior mass is expected);
samplers needing an initial point
for a chain will attempt to draw it from the ``ref`` if it has been defined (otherwise
from the prior). A good reference pdf will avoid a long *burn-in* stage during the
sampling. If you assign a single value to ``ref``, samplers will always start from
that value; however this makes convergence tests less reliable as each chain will
start from the same point (so all chains could be stuck near the same point).


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


The order of the parameters is conserved in the table of samples, except that
derived parameters are always moved to the end.

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

Multidimensional priors
-----------------------

The priors described above are obviously 1-dimensional.
You can also define more complicated, multidimensional priors
using a ``prior`` block at the base level (i.e. not indented), as in the
:ref:`the advanced example <example_advanced_shell>`. All the details below also apply to
the definition of :ref:`external likelihoods <likelihood_external>`. We will call these
custom priors "`external` priors".

Inside the ``prior`` block, list a pair of priors as ``[name]: [function]``, where the
functions must return **log**-priors. This priors will be multiplied by the
one-dimensional ones defined above. Even if you define an prior for some parameters
in the ``prior`` block, you still have to specify their bounds in the ``params`` block.

A prior function can be specified in two different ways:

a) **As a function**, either assigning a function defined with ``def``, or a ``lambda``.
   Its arguments must be known parameter names.

b) **As a string,** which will be passed to ``eval()``. The string can be a
   ``lambda`` function definition with known parameters as its arguments (you can
   use ``scipy.stats`` and  ``numpy`` as ``stats`` and ``np`` resp.), or an
   ``import_module('filename').function`` statement if your function is defined in a
   separate file ``filename`` `in the current folder` and is named ``function``
   (see e.g. :ref:`the advanced example <example_advanced_shell>`).

.. note::

   Your function definition may contain keyword arguments that aren't known parameter
   names. They will simply be ignored and left to their default.

.. warning::

   When **resuming** a run using an *external* python object as input (e.g. a prior or a
   likelihood), there is no way for Cobaya to know whether the new object is the same as
   the old one: it is left to the user to ensure reproducibility of those objects between
   runs.

.. warning::

   At this moment, external priors can only be functions of **sampled** and **fixed**
   parameters. This may be extended in the future to dependence on **derived** parameters,
   probably just for dynamically defined ones, but not for those computed by the theory
   code, since otherwise the full prior could not be computed **before** the likelihood,
   preventing us from avoiding computating the likelihood when the prior is null, or
   forcing a *post-call* to the prior.

   **Workaround #1:** If the derived parameter(s) can be computed easily from sampled and
   fixed parameters, do the computation inside the call to the external prior.

   **Workaround #2:** Define your function as a
   :ref:`external likelihood <likelihood_external>` instead, since likelihoods do have
   access to derived parameters (for an example, see `this comment
   <https://github.com/CobayaSampler/cobaya/issues/18#issuecomment-447818811>`_).


Prior normalization for evidence computation
--------------------------------------------

The one-dimensional priors defined within the ``params`` block are automatically
normalized, so any sampler that computes the evidence will produce the right results as
long as no external priors have been defined, whose normalization is unknown.

To get the prior normalization if using external functions as priors, you can substitute
your likelihood by the :doc:`dummy unit likelihood <likelihood_one>`, and make an initial
run with :doc:`PolyChord <sampler_polychord>` to get the prior volume
(see section :ref:`polychord_bayes_ratios`).

In general, avoid improper priors, since they will produce unexpected errors, e.g.

.. code-block:: yaml

    params:
      a:
        prior:
          min: -.inf
          max:  .inf


.. _repar:

Defining parameters dynamically
-------------------------------

We may want to sample in a parameter space different than the one understood by the
likelihood, e.g. because we expect the posterior to be simpler on the alternative
parameters.

For instance, in the :doc:`advanced example <example_advanced>`, the posterior on the
radius and the angle is a gaussian times a uniform, instead of a more complicated
gaussian ring. This is done in a simple way at
:ref:`the end of the example <example_advanced_rtheta>`.
Let us discuss the general case here.

To enble this, **cobaya** creates a `re-parameterization` layer between the `sampled`
parameters, and the `input` parameters of the likelihood. E.g. if we want to **sample**
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
  (or theory). They are identified by either having a prior and not being **dropped**, or
  having being assigned a fixed value or a function. Here, ``x`` would be an input
  parameter, but not ``logx``.

.. image:: ./img/diagram.svg
   :align: center
   :width: 60%

We can use a similar approach to define dynamical **derived** parameters, which can depend
on *input* and *sampled* parameters. To distinguish their notation from that of input
parameters, we insert the functions defining them under a ``derived`` property
(see the parameter ``y`` in the example above).

.. note::

   **Dynamical derived** parameters can also be functions of yet-undefined parameters.
   In that case, those parameters will be automatically requested from the likelihood (or
   theory code) that understands them.

.. note::

   Also available are the :math:`\chi^2 = -2 \log \mathcal{L}` of the used likelihoods,
   as `chi2__[name]`, e.g.

   .. code:: yaml

     likelihood:
       my_like: [...]
     params:
       my_derived:
         derived: "lambda chi2__my_like: [...]"

.. note::

   By default, the values of **dynamical input** parameters (e.g. ``x`` above) are saved
   as if they were derived parameters. If you would like to ignore them, define them using
   the following *extended notation*:

   .. code:: yaml

      params:
        x:
          value: "lambda logx: np.exp(logx)"
          derived: False

.. note::

   If you want to fix the value of a parameter whose only role is being an argument of a
   dynamically defined one and is *not supposed to be passed to the likelihood*, you need
   to explicilty *drop* it. E.g. suppose that you want to sample from a likelihood that
   depends on ``x``, but want to use ``log(x)`` as the sampled parameter; you would do it
   like this:

   .. code:: yaml

      params:
        logx:
          prior: [...]
          drop: True
        x:
          value: "lambda logx: np.exp(x)"

   Now, if you want to fix the value of ``logx`` without changing the structure of the
   input, do

   .. code:: yaml

      params:
        logx:
          value: [fixed_value]
          drop: True
        x:
          value: "lambda logx: np.exp(x)"


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
from copy import deepcopy
from types import MethodType

# Local
from cobaya.conventions import _prior, _p_ref, _prior_1d_name
from cobaya.tools import get_external_function, get_scipy_1d_pdf, read_dnumber
from cobaya.tools import _fast_uniform_logpdf, _fast_norm_logpdf, getargspec
from cobaya.log import HandledException

# Logger
import logging

log = logging.getLogger(__name__.split(".")[-1])

# Fast logpdf for uniforms and norms (do not understand nan masks!)
fast_logpdfs = {"uniform": _fast_uniform_logpdf, "norm": _fast_norm_logpdf}


class Prior(object):
    """
    Class managing the prior and reference pdf's.
    """

    def __init__(self, parameterization, info_prior=None):
        """
        Initializes the prior and reference pdf's from the input information.
        """
        constant_params_info = parameterization.constant_params()
        sampled_params_info = parameterization.sampled_params_info()
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
            fast_logpdf = fast_logpdfs.get(self.pdf[-1].dist.name)
            if fast_logpdf:
                self.pdf[-1].logpdf = MethodType(fast_logpdf, self.pdf[-1])
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
            if name == _prior_1d_name:
                log.error("The name '%s' is a reserved prior name. "
                          "Please use a different one.", _prior_1d_name)
                raise HandledException
            log.debug("Loading external prior '%s' from: '%s'", name, info_prior[name])
            self.external[name] = (
                {"logp": get_external_function(info_prior[name], name=name)})
            self.external[name]["argspec"] = (
                getargspec(self.external[name]["logp"]))
            self.external[name]["params"] = {
                p: list(sampled_params_info).index(p)
                for p in self.external[name]["argspec"].args if p in sampled_params_info}
            self.external[name]["constant_params"] = {
                p: constant_params_info[p]
                for p in self.external[name]["argspec"].args if p in constant_params_info}
            if (not (len(self.external[name]["params"]) +
                     len(self.external[name]["constant_params"]))):
                log.error("None of the arguments of the external prior '%s' "
                          "are known *fixed* or *sampled* parameters. "
                          "This prior recognizes: %r",
                          name, self.external[name]["argspec"].args)
                raise HandledException
            params_without_default = self.external[name]["argspec"].args[
                                     :(len(self.external[name]["argspec"].args) -
                                       len(self.external[name]["argspec"].defaults or []))]
            if not all([(p in self.external[name]["params"] or
                         p in self.external[name]["constant_params"])
                        for p in params_without_default]):
                log.error("Some of the arguments of the external prior '%s' "
                          "cannot be found and don't have a default value either: %s",
                          name, list(set(params_without_default)
                                     .difference(self.external[name]["params"])
                                     .difference(self.external[name]["constant_params"])))
                raise HandledException
            log.warning("External prior '%s' loaded. "
                        "Mind that it might not be normalized!", name)

    def d(self):
        """
        Returns:
           Dimensionality of the parameter space.
        """
        return len(self.pdf)

    # Not very useful, except for getting the prior names as list(self)
    # Created for consistency with likelihoods
    def __iter__(self):
        return (p for p in [_prior_1d_name] + list(self.external))

    def __len__(self):
        return 1 + len(self.external)

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
                log.warning("There are unbounded parameters. Prior bounds are given at %s "
                            "confidence level. Beware of likelihood modes at the edge of "
                            "the prior", confidence_for_unbounded)
                bounds[infs] = [
                    self.pdf[i].interval(confidence_for_unbounded) for i in infs]
            return bounds
        except AttributeError:
            log.error("Some parameter names (positions %r) have no bounds defined.", infs)
            raise HandledException

    def sample(self, n=1, ignore_external=False):
        """
        Generates samples from the prior pdf.

        If an external prior has been defined, it is not possible to sample from the prior
        directly. In that case, if you want to sample from the "default" pdf (i.e.
        ignoring the external prior), set ``ignore_external=True``.

        Returns:
          Array of ``n`` samples from the prior, as vectors ``[value of param 1, ...]``.
        """
        if not ignore_external and self.external:
            log.error("It is not possible to sample from an external prior "
                      "(see help of this function on how to fix this).")
            raise HandledException
        return np.array([pdf.rvs(n) for pdf in self.pdf]).T

    def logps(self, x):
        """
        Takes a point (sampled parameter values, in the correct order).

        Returns:
           An array of the prior log-probability densities of the given point
           or array of points. The first element on the list is the products
           of 1d priors specified in the ``params`` block, and the following
           ones (if present) are the priors specified in the ``prior`` block
           in the same order.
        """
        log.debug("Evaluating prior at %r", x)
        logps = [
                    sum([pdf.logpdf(xi) for pdf, xi in zip(self.pdf, x)])] + self.logps_external(x)
        log.debug("Got logpriors = %r", logps)
        return logps

    def logp(self, x):
        """
        Takes a point (sampled parameter values, in the correct order).

        Returns:
           The prior log-probability density of the given point or array of points.
        """
        return np.sum(self.logps(x), axis=0)

    def logps_external(self, x):
        """Evaluates the logprior using the external prior only."""
        return [ext["logp"](**dict({p: x[i] for p, i in ext["params"].items()},
                                   **ext["constant_params"]))
                for ext in self.external.values()]

    def covmat(self, ignore_external=False):
        """
        Returns:
           The covariance matrix of the prior.
        """
        if not ignore_external and self.external:
            log.error("It is not possible to get the covariance matrix "
                      "from an external prior.")
            raise HandledException
        return np.diag([pdf.var() for pdf in self.pdf]).T

    def reference(self, max_tries=np.inf, max_tries_warning="10d"):
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
        tries = 0
        max_tries_warning = read_dnumber(max_tries_warning, self.d())
        while tries < max_tries:
            tries += 1
            ref_sample = np.array([getattr(ref_pdf, "rvs", lambda: ref_pdf.real)()
                                   for i, ref_pdf in enumerate(self.ref_pdf)])
            where_no_ref = np.isnan(ref_sample)
            if np.any(where_no_ref):
                prior_sample = self.sample(ignore_external=True)[0]
                ref_sample[where_no_ref] = prior_sample[where_no_ref]
            if self.logp(ref_sample) > -np.inf:
                return ref_sample
            if tries == max_tries_warning:
                log.warning("If stuck here, maybe it is not possible to sample from the "
                            "reference pdf a point with non-null prior. Check that they "
                            "are consistent.")
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
            log.warning("Reference pdf not defined or improper for some parameters. "
                        "Using prior's sigma instead for them.")
            covmat[where_no_ref] = self.covmat(ignore_external=True)[where_no_ref]
        return covmat

    # Python magic for the "with" statement

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
