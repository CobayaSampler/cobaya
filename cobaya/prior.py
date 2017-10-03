"""
.. module:: prior

:Synopsis: Class containing the prior and reference pdf, and other parameter information.
:Author: Jesus Torrado

The ``params`` block contains all the information intrinsic
to the parameters of the model: their prior pdf and reference pdf, their latex labels,
or useful properties for particular samplers (e.g. with of a proposal pdf in and MCMC
-- see the docs section for each sampler). All this information is stored and managed by
the :class:`Prior` class.

You can specify three different kinds of parameters:

+ **Fixed** parameters are specified by assigning them a value, and are passed directly to
  the likelihood or theory code.
+ **Sampled** parameters need a ``prior`` pdf definition and, optionally,
  a reference pdf or value (``ref``), a LaTeX label to be used when producing plots,
  and additional properties to aid particular samplers.
+ **Derived** parameters do **not** have a prior definition (neither a reference value or pdf),
  but can have a LaTeX label and a ``min`` and/or ``max`` to be used in the sample analysis
  (e.g. to guess the pdf tails correctly if the derived quantity needs to be positive or
  negative -- defaulted to ``-inf``, ``inf`` resp.).

The (optional) **reference** pdf (``ref``) defines the region of the prior which is of most
interest (e.g. where most of the prior mass is expected); samplers needing an initial point
for a chain will attempt to draw it from the ``ref`` if it has been defined (otherwise
from the prior). A good reference pdf will avoid a long *burn-in* stage during the
sampling.

The syntax for priors and ref's has three fields:

+ ``dist`` (default: ``uniform``): any 1-dimensional continuous distribution from
  `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions>`_.
+ ``loc`` and ``scale`` (default: 0 and 1, resp.): the *location* and *scale* of the pdf,
  as they are understood for each particular pdf in :class:`scipy.stats` (e.g. for a
  ``uniform`` pdf, ``loc`` is the lower bound and ``scale`` is the length of the domain,
  whereas in a Gaussian ``norm`` pdf ``loc`` is the centre and ``scale`` is the standard
  deviation).
+ (alternatively) ``min`` and ``max`` (default: 0 and 1 resp.): the boundaries of the pdf
  for a bounded distribution (``uniform``, ``beta``, etc. -- unexpected behaviour for an
  unbounded one).

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

.. note:: ``theory`` is a reserved parameter name!

.. warning::

   Improper priors will produce unexpected errors. So don't try

   .. code-block:: yaml

      params:
        a:
          prior:
            min: -inf
            max:  inf

Changing and redefining parameters -- inheritance
-------------------------------------------------

When mentioning a likelihood, the parameters that appear in its ``defaults.yaml`` file are
inherited without needing to re-state them. Still, you may want to re-define their type or
some of their properties in your input file:

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

As general rules, when trying to redefine anything in a parameter, everything not re-defined
is inherited, except for the prior, which must not be inheritable in order to allow
re-defining sampled parameters into something else.

Custom, multidimensional priors
-------------------------------

The priors described above are
obviously 1-dimensional. You can also define more complicated, multidimensional priors
using a ``prior`` block at the base level (i.e. not indented).

Inside the ``prior`` block, list a pair of priors as ``[name]: [function]``, where the
functions must return **log-priors**.
A prior function can be specified in three different ways:

a) **As a string:** If a string is given, it will be passed to Python's ``eval()`` function. In this case, use Python's ``lambda`` to describe the function, using known parameter names as its arguments. In your string you can access ``scipy.stats`` and  ``numpy`` members under ``stats`` and ``np`` respectively.

b) **As a function:** If your function is more complicated (it takes more than a line of code) and cannot be written as a ``lambda``, you can pass a *function object* directly, whose arguments must be known parameter names. This option is of course not compatible with shell invocation, since a ``yaml`` file can only contain text.

c) **As an import statement:** If your function cannot be written as a ``lambda`` but you  would still like to have your input as a ``yaml`` file, you can code your function into a separate file and use ``import_module`` to load it on the fly.

Let us apply this to the example in section :ref:`in_example`. We will add a
*gaussian ring* prior on both parameters, centred at the origin, with radius 0.5 and sigma of 0.1;
and a simple *gaussian* prior on the sum of both parameters,
centred at 0.5 and with sigma of 0.2.

Using option **(a)** above, we would simply add another block to the input file:

.. code-block:: yaml

   prior:
     ring: "lambda mock_a, mock_b: stats.norm.logpdf(np.sqrt(mock_a**2 + mock_b**2), loc=0.5, scale=0.1)"
     linear: "lambda y: stats.norm.logpdf(mock_a + mock_b, loc=0.5, scale=0.2)"

.. warning::
   Use quotation marks, ``"`` or ``'``, to enclose the prior! Otherwise, the colon of the
   ``lambda`` will be wrongly interpreted by ``yaml``.

We may consider that these priors, or any other, are too complicated for a one-liner,
so we may prefer to code it in a separate file, say ``myprior.py`` in the same folder:

.. code-block:: python

   from scipy.stats import norm
   import numpy as np

   def gaussian_ring(mock_a, mock_b):
       r = np.sqrt(mock_a**2 + mock_b**2)
       return norm.logpdf(r, loc=0.5, scale=0.1)

   def linear_combination(mock_a, mock_b):
       return norm.logpdf(mock_a + mock_b, loc=0.5, scale=0.2)

In this case, which corresponds to option **(c)**, the prior block would look like:

.. code:: yaml

   prior:
     ring: import_module("myprior").gaussian_ring
     linear: import_module("myprior").linear_combination

This way, you can invoke `cobaya` from the command line with the corresponding ``yaml``
file as input, provided that ``myprior.py`` is in the same folder. It is the most
*portable* version, in case you want to pass your sample around, or archive it for future
reference.

CHECK IF DIFFERENT FOLDERS WORK!!!

If you are running `cobaya` within a Python script or a Jupyter notebook, you can
simply pass a dictionary of functions and their names in under
the ``prior`` key of the *information* dicctionary that is passed to ``cobaya.run.run``,
as described in case **(b)**.

Following the example in section :ref:`in_example_script` and using the priors above,
now using case , simply define the ``gaussian_ring`` and
``linear_combination`` functions somewhere early in the script and assign them to the
dictionary containing all the information, before calling ``cobaya.run.run``:

.. code:: python

   info["prior"] = {"ring": gaussian_ring, "linear": linear_combination}



decir que no estan normalised (link a seccion (TODO) sobre normalizar posteriors)

TEST EXAMPLES!!!

YOU MUST STLL SPECIFY THE LIMITS!!!


.. image:: img_prior_ring.svg

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
from collections import OrderedDict as odict
import numpy as np
import numbers
import inspect

# Local
from cobaya.conventions import _prior, _p_dist, _p_ref, _p_label
from cobaya.tools import get_external_function, get_scipy_1d_pdf
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)


class Prior():
    """
    Class managing the prior and reference pdf's.
    """
    def __init__(self, parametrisation, info_prior=None):
        """
        Initialises the prior and reference pdf's from the input information.
        """
        sampled_params_info = parametrisation.sampled_params()
        if not sampled_params_info:
            log.warning("No sampled parameters requested! This will fail for non-mock samplers.")
        # pdf: a list of independent components
        # in principle, separable: one per parameter
        self.name = []
        self.pdf = []
        self.ref_pdf = []
        self.lims = []
        for p in sampled_params_info:
            self.name += [p]
            prior = sampled_params_info[p].get(_prior)
            self.pdf += [get_scipy_1d_pdf({p: prior})]
            # Get the reference (1d) pdf
            ref = sampled_params_info[p].get(_p_ref)
            # Cases: number, pdf (something, but not a number), nothing
            if isinstance(ref, numbers.Number):
                self.ref_pdf += [ref]
            elif ref != None:
                self.ref_pdf += [get_scipy_1d_pdf({p: ref})]
            else:
                self.ref_pdf += [np.nan]
            self.lims += [(-np.inf,np.inf)]
            if self.pdf[-1].__class__.__name__ == "rv_frozen":
                self.lims[-1] = self.pdf[-1].interval(1)
            else:
                log.error("No limits defined for parameter '%s'.", param)
                raise HandledException
        # Process the external prior(s):
        self.external = odict()
        for name in (info_prior if info_prior else {}):
            log.debug("Loading external prior '%s' from: '%s'", name, info_prior[name])
            self.external[name] = {"logp": get_external_function(info_prior[name])}
            self.external[name]["argspec"] = inspect.getargspec(self.external[name]["logp"])
            try:
                self.external[name]["indices"] = [list(sampled_params_info).index(p)
                                                  for p in self.external[name]["argspec"].args]
            except ValueError:
                log.error(
                    "The arguments of the external prior '%s' must be known *sampled* parameters. "
                    "Got %r", name, self.external[name]["argspec"].args)
                raise HandledException
            log.warning("External prior '%s' loaded. Mind that it might not be normalised!", name)

    def d(self):
        """
        Returns:
           Dimensionality of the parameter space.
        """
        return len(self.pdf)

    def limits(self):
        """
        Returns:
           An array of limits ``[min,max]`` for the parameters, in the order given by the
           input.

        NB: If an external prior has been defined, the limits given in the 'prior'
        sub-block of that particular parameter's info may not be faithful to the externally
        defined prior.
        """
        return np.array(self.lims.values())

    def sample(self, n=1, external_error=True):
        """
        Generates samples from the prior pdf.

        If an external prior has been defined, it is not possible to sample from the prior
        directly. In that case, if you want to sample from the "default" pdf (i.e.
        ignoring the external prior), set `external_error` to `False`.

        Returns:
          An array of ``n`` samples from the prior, as vectors ``[value of param 1, ...]``.
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
        return (np.prod([pdf.pdf(xi) for pdf,xi in zip(self.pdf,x)])
                * np.exp(self.logp_external(x)))

    def logp(self, x):
        """
        Returns:
           The log-probability density of the given point or array of points.
        """
        return (sum([pdf.logpdf(xi) for pdf,xi in zip(self.pdf,x)])
                + self.logp_external(x))

    def logp_external(self, x):
        """Evaluates the logprior using the external prior only."""
        return sum([ext["logp"](*x[ext["indices"]]) for ext in self.external.values()])

    def covmat(self, external_error=True):
        """
        Returns:
           The covariance matrix of the prior.
        """
        if external_error and self.external:
            log.error("It is not possible to get the covariance matrix from an external prior.")
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
          The standard deviation of the 1d ref pdf's. For those parameters that do not have
          a ref pdf defined, the standard deviation of the prior is taken instead.
        """
        covmat = np.diag([getattr(ref_pdf, "var", lambda: np.nan)()
                          for i, ref_pdf in enumerate(self.ref_pdf)])
        where_no_ref = np.isnan(covmat)
        if np.any(where_no_ref):
            log.warning("Reference pdf not defined or improper for some parameters. "
                        "Using prior's sigma instead for them.")
            covmat[where_no_ref] = self.covmat(external_error=False)[where_no_ref]
        return covmat

    # Python magic for the "with" statement
    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        pass
