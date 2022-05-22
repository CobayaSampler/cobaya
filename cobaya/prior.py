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

The (optional) **reference** pdf (``ref``) for **sampled** parameters defines the region
of the prior which is of most interest (e.g. where most of the prior mass is expected);
samplers needing an initial point for a chain will attempt to draw it from the ``ref``
if it has been defined (otherwise from the prior). A good reference pdf will avoid a long
*burn-in* stage during the sampling. If you assign a single value to ``ref``, samplers
will always start from that value; however this makes convergence tests less reliable as
each chain will start from the same point (so all chains could be stuck near the same
point).

.. note::

   When running in parallel with MPI you can give ``ref`` different values for each of
   the different parallel processes (either at initialisation or by calling
   :func:`Prior.set_reference`). They will be taken into account e.g. for starting
   parallel MCMC chains at different points/distributions of your choice.

The syntax for priors and ref's has the following fields:

+ ``dist`` (default: ``uniform``): any 1-dimensional continuous distribution from
  `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions>`_;
  e.g. ``uniform``, ``[log]norm`` for a [log-]gaussian,
  or ``halfnorm`` for a half-gaussian.
+ ``loc`` and ``scale`` (default: 0 and 1, resp.): the *location* and *scale* of the pdf,
  as they are understood for each particular pdf in :class:`scipy.stats`; e.g. for a
  ``uniform`` pdf, ``loc`` is the lower bound and ``scale`` is the length of the domain,
  whereas in a gaussian (``norm``) ``loc`` is the mean and ``scale`` is the standard
  deviation.
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
functions must return **log**-priors. These priors will be multiplied by the
one-dimensional ones defined above. Even if you define a prior for some parameters
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

   When **resuming** a run using an **external** python object as input (e.g. a prior or a
   likelihood), there is no way for Cobaya to know whether the new object is the same as
   the old one: it is left to the user to ensure reproducibility of those objects between
   runs.

.. warning::

   External priors can only be functions **sampled** and **fixed**
   and **derived** parameters that are dynamically defined in terms of other inputs.
   Derived parameters computed by the theory code cannot be used in a prior, since
   otherwise the full prior could not be computed **before** the likelihood,
   preventing us from avoiding computing the likelihood when the prior is null, or
   forcing a *post-call* to the prior.

   **Workaround:** Define your function as a
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

We may want to sample in a parameter space different from the one understood by the
likelihood, e.g. because we expect the posterior to be simpler on the alternative
parameters.

For instance, in the :doc:`advanced example <example_advanced>`, the posterior on the
radius and the angle is a gaussian times a uniform, instead of a more complicated
gaussian ring. This is done in a simple way at
:ref:`the end of the example <example_advanced_rtheta>`.
Let us discuss the general case here.

To enable this, **cobaya** creates a `re-parameterization` layer between the `sampled`
parameters, and the `input` parameters of the likelihood. E.g. if we want to **sample**
from the logarithm of an **input** parameter of the likelihood, we would do:

.. code:: yaml

    params:
      logx:
        prior: [...]
        drop: True
      x: "lambda logx: np.exp(logx)"
      y:
        derived: "lambda x: x**2"

When that information is loaded **cobaya** will create an interface between two sets of
parameters:

+ **Sampled** parameters, i.e. those that have a prior, including ``logx`` but not ``x``
  (it does not have a prior: it's fixed by a function). Those are the ones with which the
  **sampler** interacts. Since the likelihood does not understand them, they must be
  **dropped** with ``drop: True`` if you are using any parameter-agnostic components.

+ Likelihood **input** parameters: those that will be passed to the likelihood
  (or theory). They are identified by either having a prior and not being **dropped**, or
  having being assigned a fixed value or a function. Here, ``x`` would be an input
  parameter, but not ``logx``.

.. _cobaya_diagram:
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
   to explicitly *drop* it. E.g. suppose that you want to sample from a likelihood that
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

# Global
import numpy as np
import numbers
from types import MethodType
from typing import Sequence, NamedTuple, Callable, Optional, Mapping, List

# Local
from cobaya.conventions import prior_1d_name
from cobaya.typing import PriorsDict
from cobaya.tools import get_external_function, get_scipy_1d_pdf, read_dnumber
from cobaya.tools import _fast_norm_logpdf, getfullargspec
from cobaya.log import LoggedError, HasLogger
from cobaya.parameterization import Parameterization

# Fast logpdf for uniforms and norms (do not understand nan masks!)
fast_logpdfs = {"norm": _fast_norm_logpdf}


class ExternalPrior(NamedTuple):
    logp: Callable
    params: Sequence[str]


class Prior(HasLogger):
    """
    Class managing the prior and reference pdf's.
    """

    def __init__(self, parameterization: Parameterization,
                 info_prior: Optional[PriorsDict] = None):
        """
        Initializes the prior and reference pdf's from the input information.
        """
        self.set_logger()
        self._parameterization = parameterization
        sampled_params_info = parameterization.sampled_params_info()
        # pdf: a list of independent components
        # in principle, separable: one per parameter
        self.params = []
        self.pdf = []
        self._bounds = np.zeros((len(sampled_params_info), 2))
        for i, p in enumerate(sampled_params_info):
            self.params += [p]
            prior = sampled_params_info[p].get("prior")
            self.pdf += [get_scipy_1d_pdf({p: prior})]
            fast_logpdf = fast_logpdfs.get(self.pdf[-1].dist.name)
            if fast_logpdf:
                self.pdf[-1].logpdf = MethodType(fast_logpdf, self.pdf[-1])
            self._bounds[i] = [-np.inf, np.inf]
            try:
                self._bounds[i] = self.pdf[-1].interval(1)
            except AttributeError:
                raise LoggedError(self.log, "No bounds defined for parameter '%s' "
                                            "(maybe not a scipy 1d pdf).", p)
        self._uniform_indices = np.array(
            [i for i, pdf in enumerate(self.pdf) if pdf.dist.name == 'uniform'],
            dtype=int)
        self._non_uniform_indices = np.array(
            [i for i in range(len(self.pdf)) if i not in self._uniform_indices],
            dtype=int)
        self._non_uniform_logpdf = [self.pdf[i].logpdf for i in self._non_uniform_indices]
        self._upper_limits = self._bounds[:, 1].copy()
        self._lower_limits = self._bounds[:, 0].copy()
        self._uniform_logp = -np.sum(np.log(self._upper_limits[self._uniform_indices] -
                                            self._lower_limits[self._uniform_indices]))
        # Set the reference pdf's
        self.set_reference({p: v.get("ref") for p, v in sampled_params_info.items()})
        # Process the external prior(s):
        self.external = {}
        self.external_dependence = set()
        info_prior = info_prior or {}
        for name in info_prior:
            if name == prior_1d_name:
                raise LoggedError(self.log, "The name '%s' is a reserved prior name. "
                                            "Please use a different one.", prior_1d_name)
            self.log.debug(
                "Loading external prior '%s' from: '%s'", name, info_prior[name])
            logp = get_external_function(info_prior[name], name=name)
            argspec = getfullargspec(logp)
            known = set(parameterization.input_params())
            params = [p for p in argspec.args if p in known]
            params_without_default = set(
                argspec.args[:(len(argspec.args) - len(argspec.defaults or []))])
            unknown = params_without_default - known
            if unknown:
                if unknown.intersection(parameterization.derived_params()):
                    err = ("External prior '%s' has arguments %s that are output derived "
                           "parameters, Priors must be functions of input parameters. "
                           "Use a separate 'likelihood' for the prior if needed.")
                else:
                    err = ("Some of the arguments of the external prior '%s' cannot be "
                           "found and don't have a default value either: %s")
                raise LoggedError(self.log, err, name, list(unknown))
            self.external_dependence.update(params)
            self.external[name] = ExternalPrior(logp=logp, params=params)
            self.mpi_warning("External prior '%s' loaded. "
                             "Mind that it might not be normalized!", name)

        parameterization.check_dropped(self.external_dependence)

    def d(self):
        """
        Returns:
           Dimensionality of the parameter space.
        """
        return len(self.pdf)

    # Not very useful, except for getting the prior names as list(self)
    # Created for consistency with likelihoods
    def __iter__(self):
        return (p for p in [prior_1d_name] + list(self.external))

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
        bounds = self._bounds.copy()
        infs = list(set(np.argwhere(np.isinf(bounds)).T[0]))
        try:
            if infs:
                self.mpi_warning("There are unbounded parameters (%r). Prior bounds "
                                 "are given at %s confidence level. Beware of "
                                 "likelihood modes at the edge of the prior",
                                 [self.params[ix] for ix in infs],
                                 confidence_for_unbounded)
                bounds[infs] = [
                    self.pdf[i].interval(confidence_for_unbounded) for i in infs]
            return bounds
        except AttributeError:
            raise LoggedError(
                self.log,
                "Some parameter names (positions %r) have no bounds defined.", infs)

    def sample(self, n=1, ignore_external=False, random_state=None):
        """
        Generates samples from the prior pdf.

        If an external prior has been defined, it is not possible to sample from the prior
        directly. In that case, if you want to sample from the "default" pdf (i.e.
        ignoring the external prior), set ``ignore_external=True``.

        Returns:
          Array of ``n`` samples from the prior, as vectors ``[value of param 1, ...]``.
        """
        if not ignore_external and self.external:
            raise LoggedError(
                self.log, "It is not possible to sample from an external prior "
                          "(see help of this function on how to fix this).")
        return np.array([pdf.rvs(n, random_state=random_state) for pdf in self.pdf]).T

    def logps(self, x: np.ndarray) -> List[float]:
        """
        Takes a point (sampled parameter values, in the correct order).

        Returns:
           An array of the prior log-probability densities of the given point
           or array of points. The first element on the list is the products
           of 1d priors specified in the ``params`` block, and the following
           ones (if present) are the priors specified in the ``prior`` block
           in the same order.
        """
        logps = self.logps_internal(x)
        if logps != -np.inf:
            if self.external:
                input_params = self._parameterization.to_input(x)
                return [logps] + self.logps_external(input_params)
            else:
                return [logps]
        else:
            return [-np.inf] * (1 + len(self.external))

    def logp(self, x: np.ndarray):
        """
        Takes a point (sampled parameter values, in the correct order).

        Returns:
           The prior log-probability density of the given point or array of points.
        """
        return np.sum(self.logps(x), axis=0)

    def logps_internal(self, x: np.ndarray) -> float:
        """
        Takes a point (sampled parameter values, in the correct order).

        Returns:
           The prior log-probability density of the given point
           or array of points, only including the products
           of 1d priors specified in the ``params`` block, no external priors
        """
        self.log.debug("Evaluating prior at %r", x)
        if all(x <= self._upper_limits) and all(x >= self._lower_limits):
            logps = self._uniform_logp + (sum([logpdf(xi) for logpdf, xi in
                                               zip(self._non_uniform_logpdf,
                                                   x[self._non_uniform_indices])])
                                          if len(self._non_uniform_indices) else 0)
        else:
            logps = -np.inf

        self.log.debug("Got logpriors = %r", logps)
        return logps

    def logps_external(self, input_params) -> List[float]:
        """Evaluates the logprior using the external prior only."""
        return [ext.logp(**{p: input_params[p] for p in ext.params})
                for ext in self.external.values()]

    def covmat(self, ignore_external=False):
        """
        Returns:
           The covariance matrix of the prior.
        """
        if not ignore_external and self.external:
            raise LoggedError(
                self.log,
                "It is not possible to get the covariance matrix from an external prior.")
        return np.diag([pdf.var() for pdf in self.pdf]).T

    def set_reference(self, ref_info):
        """
        Sets or updates the reference pdf with the given parameter input info.

        ``ref_info`` should be a dict ``{parameter_name: [ref definition]}``, not
        ``{parameter_name: {"ref": [ref definition]}}``.

        When called after prior initialisation, not mentioning a parameter leaves
        its reference pdf unchanged, whereas explicitly setting ``ref: None`` sets
        the prior as the reference pdf.

        You can set different reference pdf's/values for different MPI processes,
        e.g. for fixing different starting points for parallel MCMC chains.
        """
        if not hasattr(self, "ref_pdf"):
            # Initialised with nan's in case ref==None: no ref -> uses prior
            self.ref_pdf = [np.nan] * self.d()
        unknown = set(ref_info).difference(self.params)
        if unknown:
            raise LoggedError(self.log,
                              f"Cannot set reference pdf for parameter(s) {unknown}: "
                              "not sampled parameters.")
        for i, p in enumerate(self.params):
            # The next if ensures correct behavious in "update call",
            # where not mentioning a parameter and making its ref None are different
            # (not changing vs setting to prior)
            if p not in ref_info:
                continue  # init: use prior; update: don't change
            ref = ref_info[p]
            # [number, number] interpreted as Gaussian
            if isinstance(ref, Sequence) and len(ref) == 2 and all(
                    isinstance(n, numbers.Number) for n in ref):
                ref = {"dist": "norm", "loc": ref[0], "scale": ref[1]}
            if isinstance(ref, numbers.Real):
                self.ref_pdf[i] = float(ref)
            elif isinstance(ref, Mapping):
                self.ref_pdf[i] = get_scipy_1d_pdf({p: ref})
            elif ref is None:
                # We only get here if explicit `param: None` mention!
                self.ref_pdf[i] = np.nan
            else:
                raise LoggedError(self.log,
                                  "'ref' for starting position should be None or a number"
                                  ", a list of two numbers for normal mean and deviation,"
                                  "or a dict with parameters for a scipy distribution.")
        # Re-set the pointlike-ref property
        if hasattr(self, "_ref_is_pointlike"):
            delattr(self, "_ref_is_pointlike")
        self.reference_is_pointlike

    @property
    def reference_is_pointlike(self) -> bool:
        """
        Whether there is a fixed reference point for all parameters, such that calls to
        :func:`Prior.reference` would always return the same.
        """
        if not hasattr(self, "_ref_is_pointlike"):
            self._ref_is_pointlike = all(
                # np.nan is a numbers.Number instance, but not a fixed ref (uses prior)
                (isinstance(ref, numbers.Number) and ref is not np.nan)
                for ref in self.ref_pdf)
        return self._ref_is_pointlike

    def reference(self, max_tries=np.inf, warn_if_tries="10d", ignore_fixed=False,
                  warn_if_no_ref=True, random_state=None) -> np.ndarray:
        """
        Returns:
          One sample from the ref pdf. For those parameters that do not have a ref pdf
          defined, their value is sampled from the prior.

        If `ignored_fixed=True` (default: `False`), fixed reference values will be ignored
        in favor of the full prior, ensuring some randomness for all parameters (useful
        e.g. to prevent caching when measuring speeds).

        NB: The way this function works may be a little dangerous:
        if two parameters have an (external)
        joint prior defined and only one of them has a reference pdf, one should
        sample the other from the joint prior *conditioned* to the first one being
        drawn from the reference. Otherwise, one may end up with a point with null
        prior pdf. In any case, it should work for getting initial reference points,
        e.g. for an MCMC chain.
        """
        if np.nan in self.ref_pdf and warn_if_no_ref:
            self.log.info(
                "Reference values or pdfs for some parameters were not provided. "
                "Sampling from the prior instead for those parameters.")

        where_ignore_ref = [r is np.nan or ignore_fixed and isinstance(r, numbers.Real)
                            for r in self.ref_pdf]
        tries = 0
        warn_if_tries = read_dnumber(warn_if_tries, self.d())
        ref_sample = np.empty(len(self.ref_pdf))
        while tries < max_tries:
            tries += 1
            if any(where_ignore_ref):
                prior_sample = self.sample(ignore_external=True,
                                           random_state=random_state)[0]
                ref_sample[where_ignore_ref] = prior_sample[where_ignore_ref]
            for i, ref_pdf in enumerate(self.ref_pdf):
                if not where_ignore_ref[i]:
                    if hasattr(ref_pdf, "rvs"):
                        ref_sample[i] = ref_pdf.rvs(
                            random_state=random_state)  # type: ignore
                    else:
                        ref_sample[i] = ref_pdf.real

            if self.logp(ref_sample) > -np.inf:
                return ref_sample
            if tries == warn_if_tries:
                self.log.warning(
                    "If stuck here, maybe it is not possible to sample from the "
                    "reference pdf a point with non-null prior. Check that the reference "
                    "pdf and the prior are consistent.")
        if self.reference_is_pointlike:
            raise LoggedError(self.log, "The reference point provided has null prior. "
                                        "Set 'ref' to a different point or a pdf.")
        raise LoggedError(
            self.log, "Could not sample from the reference pdf a point with non-"
                      "null prior density after %d tries. "
                      "Maybe your prior is improper of your reference pdf is "
                      "null-defined in the domain of the prior.", max_tries)

    def reference_variances(self):
        """
        Returns:
          The standard variances of the 1d ref pdf's. For those parameters that do not
          have a proposal defined, the standard deviation of the prior can be taken
          as a starting point to estimate one.
        """
        variances = np.array([getattr(ref_pdf, "var", lambda: np.nan)()
                              for i, ref_pdf in enumerate(self.ref_pdf)])
        where_no_ref = np.isnan(variances)
        if np.any(where_no_ref):
            self.mpi_warning("Reference pdf not defined or improper for some parameters. "
                             "Using prior's sigma instead for them.")
            variances[where_no_ref] = np.diag(self.covmat(ignore_external=True
                                                          ))[where_no_ref]
        return variances
