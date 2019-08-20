r"""
.. module:: samplers.minimize

:Synopsis: Posterior/likelihood *maximization* (i.e. -log(post) and chi^2 minimization).
:Author: Jesus Torrado

This is a **maximizator** for posteriors or likelihoods, based on
`scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
and `Py-BOBYQA <https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/index.html>`_
(added in 2.0).

.. note::

   BOBYQA tends to work better on Cosmological problems with the default settings.

.. |br| raw:: html

   <br />

.. note::
   **If you use BOBYQA, please cite it as:**
   |br|
   `C. Cartis, J. Fiala, B. Marteau, L. Roberts,
   "Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers"
   (arXiv:1804.00154) <https://arxiv.org/abs/1804.00154>`_
   |br|
   `C. Cartis, L. Roberts, O. Sheridan-Methven,
   "Escaping local minima with derivative-free methods: a numerical investigation"
   (arXiv:1812.11343) <https://arxiv.org/abs/1812.11343>`_
   |br|
   `M.J.D. Powell,
   "The BOBYQA Algorithm for Bound Constrained Optimization without Derivatives",
   (Technical Report 2009/NA06, DAMTP, University of Cambridge)
   <http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf>`_

   **If you use scipy**, you can find `the appropriate references here
   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

It works more effectively when run on top of a Monte Carlo sample: just change the sampler
for ``minimize`` with the desired options, and it will use as a starting point the
*maximum a posteriori* (MAP) or best fit (maximum likelihood, o minimal :math:`\chi^2`)
found so far, as well as the covariance matrix of the sample for rescaling of the
parameter jumps.

As text output, it produces a ``[output prefix].minimum.txt`` if the MAP was requested, or
``[output prefix].bestfit.txt`` if the maximum likelihood was requested
(``ignore_prior: True``).

When called from a Python script, Cobaya's ``run`` function returns the updated info
and the products described below in the method
:func:`products <samplers.minimize.minimize.products>`.

It is recommended to run a couple of parallel MPI processes:
it will finally pick the best among the results.

.. warning::

   Since Cobaya is often used on likelihoods featuring numerical noise (e.g. Cosmology),
   we have reduced the default accuracy criterion for the minimizers, so that they
   converge in a limited amount of time. If your posterior is fast to evaluate, you may
   want to refine the convergence parameters (see ``override`` options in the ``yaml``
   below).

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import os
import numpy as np
from scipy.optimize import minimize as scpminimize
from collections import OrderedDict as odict

import pybobyqa  # in the py-bobyqa pip package

import logging
from copy import deepcopy

# Local
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi_size, get_mpi_comm, am_single_or_primary_process
from cobaya.collection import OnePoint
from cobaya.log import LoggedError
from cobaya.tools import read_dnumber, choleskyL, recursive_update

# Handling scpiy vs BOBYQA
evals_attr = {"scipy": "fun", "bobyqa": "f"}


class minimize(Sampler):
    def initialize(self):
        """Prepares the arguments for `scipy.minimize`."""
        if am_single_or_primary_process():
            self.log.info("Initializing")
        self.max_evals = read_dnumber(self.max_evals, self.model.prior.d())
        # Configure target
        method = self.model.loglike if self.ignore_prior else self.model.logpost
        kwargs = {"make_finite": True}
        if self.ignore_prior:
            kwargs.update({"return_derived": False})
        self.logp = lambda x: method(x, **kwargs)
        # Try to load info from previous samples.
        # If none, sample from reference (make sure that it has finite like/post)
        initial_point = None
        covmat = None
        if self.output:
            collection_in = self.output.load_collections(
                self.model, skip=0, thin=1, concatenate=True)
            if collection_in:
                initial_point = (
                    collection_in.bestfit() if self.ignore_prior else collection_in.MAP())
                initial_point = initial_point[
                    list(self.model.parameterization.sampled_params())].values
                self.log.info("Starting from %s of previous chain:",
                              "best fit" if self.ignore_prior else "MAP")
                # TODO: if ignore_prior, one should use *like* covariance (this is *post*)
                covmat = collection_in.cov()
        if initial_point is None:
            this_logp = -np.inf
            while not np.isfinite(this_logp):
                initial_point = self.model.prior.reference()
                this_logp = self.logp(initial_point)
            self.log.info("Starting from random initial point:")
        self.log.info(dict(zip(self.model.parameterization.sampled_params(), initial_point)))
        # Cov and affine transformation
        self._affine_transform_matrix = None
        self._inv_affine_transform_matrix = None
        self._affine_transform_baseline = None
        if covmat is None:
            # Use as much info as we have from ref & prior
            covmat = self.model.prior.reference_covmat()
        # Transform to space where initial point is at centre, and cov is normalised
        sigmas_diag, L = choleskyL(covmat, return_scale_free=True)
        self._affine_transform_matrix = np.linalg.inv(sigmas_diag)
        self._inv_affine_transform_matrix = sigmas_diag
        self._affine_transform_baseline = initial_point
        self.affine_transform = lambda x: (
            self._affine_transform_matrix.dot(x - self._affine_transform_baseline))
        self.inv_affine_transform = lambda x: (
                self._inv_affine_transform_matrix.dot(x) + self._affine_transform_baseline)
        bounds = self.model.prior.bounds(
            confidence_for_unbounded=self.confidence_for_unbounded)
        # Re-scale
        self.logp_transf = lambda x: self.logp(self.inv_affine_transform(x))
        initial_point = self.affine_transform(initial_point)
        bounds = np.array([self.affine_transform(bounds[:, i]) for i in range(2)]).T
        # Configure method
        if self.method.lower() == "bobyqa":
            self.minimizer = pybobyqa.solve
            self.kwargs = {
                "objfun": (lambda x: -self.logp_transf(x)),
                "x0": initial_point,
                "bounds": np.array(list(zip(*bounds))),
                "seek_global_minimum": (
                    True if get_mpi_size() in [0, 1] else False),
                "maxfun": int(self.max_evals)}
            self.kwargs = recursive_update(
                deepcopy(self.kwargs), self.override_bobyqa or {})
            self.log.debug("Arguments for pybobyqa.solve:\n%r",
                           {k: v for k, v in self.kwargs.items() if k != "objfun"})
        elif self.method.lower() == "scipy":
            self.minimizer = scpminimize
            self.kwargs = {
                "fun": (lambda x: -self.logp_transf(x)),
                "x0": initial_point,
                "bounds": bounds,
                "options": {
                    "maxiter": self.max_evals,
                    "disp": (self.log.getEffectiveLevel() == logging.DEBUG)}}
            self.kwargs = recursive_update(
                deepcopy(self.kwargs), self.override_scipy or {})
            self.log.debug("Arguments for scipy.optimize.minimize:\n%r",
                           {k: v for k, v in self.kwargs.items() if k != "fun"})
        else:
            methods = ["bobyqa", "scipy"]
            raise LoggedError(
                self.log, "Method '%s' not recognized. Try one of %r.", self.method, methods)

    def run(self):
        """
        Runs `scipy.minimize`
        """
        self.log.info("Starting minimization.")
        try:
            self.result = self.minimizer(**self.kwargs)
        except:
            self.log.error("Minimizer '%s' raised an unexpected error:", self.method)
            raise
        self.success = (self.result.success if self.method.lower() == "scipy" else
                        self.result.flag == self.result.EXIT_SUCCESS)
        if self.success:
            self.log.info("Finished successfully!")
        else:
            if self.method.lower() == "bobyqa":
                reason = {
                    self.result.EXIT_MAXFUN_WARNING:
                        "Maximum allowed objective evaluations reached. "
                        "This is the most likely return value when using multiple restarts.",
                    self.result.EXIT_SLOW_WARNING:
                        "Maximum number of slow iterations reached.",
                    self.result.EXIT_FALSE_SUCCESS_WARNING:
                        "Py-BOBYQA reached the maximum number of restarts which decreased the"
                        " objective, but to a worse value than was found in a previous run.",
                    self.result.EXIT_INPUT_ERROR:
                        "Error in the inputs.",
                    self.result.EXIT_TR_INCREASE_ERROR:
                        "Error occurred when solving the trust region subproblem.",
                    self.result.EXIT_LINALG_ERROR:
                        "Linear algebra error, e.g. the interpolation points produced a "
                        "singular linear system."
                }[self.result.flag]
            else:
                reason = ""
            self.log.error("Finished unsuccessfully." +
                           (" Reason: " + reason if reason else ""))

    def close(self, *args):
        """
        Determines success (or not), chooses best (if MPI)
        and produces output (if requested).
        """
        evals_attr_ = evals_attr[self.method.lower()]
        # If something failed
        if not hasattr(self, "result"):
            return
        if get_mpi_size():
            results = get_mpi_comm().gather(self.result, root=0)
            _inv_affine_transform_matrices = get_mpi_comm().gather(
                self._inv_affine_transform_matrix, root=0)
            _affine_transform_baselines = get_mpi_comm().gather(
                self._affine_transform_baseline, root=0)
            if am_single_or_primary_process():
                i_min = np.argmin([getattr(r, evals_attr_) for r in results])
                self.result = results[i_min]
                self._inv_affine_transform_matrix = _inv_affine_transform_matrices[i_min]
                self._affine_transform_baseline = _affine_transform_baselines[i_min]
        if am_single_or_primary_process():
            if not self.success:
                raise LoggedError(
                    self.log, "Minimization failed! Here is the raw result object:\n%s",
                    str(self.result))
            logp_min = -np.array(getattr(self.result, evals_attr_))
            x_min = self.inv_affine_transform(self.result.x)
            self.log.info("-log(%s) minimized to %g",
                          "likelihood" if self.ignore_prior else "posterior", -logp_min)
            recomputed_post_min = self.model.logposterior(x_min, cached=False)
            recomputed_logp_min = (sum(recomputed_post_min.loglikes) if self.ignore_prior
                                   else recomputed_post_min.logpost)
            if not np.allclose(logp_min, recomputed_logp_min):
                raise LoggedError(
                    self.log,
                    "Cannot reproduce result. Maybe yout likelihood is stochastic? "
                    "Recomputed min: %g (was %g) at %r",
                    recomputed_logp_min, logp_min, x_min)
            self.minimum = OnePoint(
                self.model, self.output, name="",
                extension=("bestfit.txt" if self.ignore_prior else "minimum.txt"))
            self.minimum.add(x_min, derived=recomputed_post_min.derived,
                             logpost=recomputed_post_min.logpost,
                             logpriors=recomputed_post_min.logpriors,
                             loglikes=recomputed_post_min.loglikes)
            self.log.info(
                "Parameter values at minimum:\n%s", self.minimum.data.to_string())
            self.minimum._out_update()
            self.dump_getdist()

    def products(self):
        r"""
        Returns a dictionary containing:

        - ``minimum``: :class:`OnePoint` that maximizes the posterior or likelihood
          (depending on ``ignore_prior``).

        - ``result_object``: instance of results class of
          `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`_
          or `pyBOBYQA
          <https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/userguide.html>`_.

        - ``M``: inverse of the affine transform matrix (see below).
          ``None`` if no transformation applied.

        - ``X0``: offset of the affine transform matrix (see below)
          ``None`` if no transformation applied.

        If non-trivial ``M`` and ``X0`` are returned, this means that the minimizer has
        been working on an affine-transformed parameter space :math:`x^\prime`, from which
        the real space points can be obtained as :math:`x = M x^\prime + X_0`. This inverse
        transformation needs to be applied to the coordinates appearing inside the
        ``result_object``.
        """
        if am_single_or_primary_process():
            return {"minimum": self.minimum, "result_object": self.result,
                    "M": self._inv_affine_transform_matrix,
                    "X0": self._affine_transform_baseline}

    def getdist_point_text(self, params, weight=None, minuslogpost=None):
        lines = []
        if weight is not None:
            lines.append('  weight    = %s' % weight)
        if minuslogpost is not None:
            lines.append(' -log(Like) = %s' % minuslogpost)
            lines.append('  chi-sq    = %s' % (2 * minuslogpost))
        lines.append('')
        labels = self.model.parameterization.labels()
        label_list = list(labels.keys())
        if hasattr(params, 'chi2_names'): label_list += params.chi2_names
        width = max([len(lab) for lab in label_list]) + 2

        def add_section(pars):
            for p, val in pars:
                lab = labels.get(p, p)
                num = label_list.index(p) + 1
                if isinstance(val, (float, np.floating)) and len(str(val)) > 10:
                    lines.append("%5d  %-17.9e %-*s %s" % (num, val, width, p, lab))
                else:
                    lines.append("%5d  %-17s %-*s %s" % (num, val, width, p, lab))

        num_sampled = len(self.model.parameterization.sampled_params())
        num_derived = len(self.model.parameterization.derived_params())
        add_section([[p, params[p]] for p in self.model.parameterization.sampled_params()])
        lines.append('')
        add_section([[p, value] for p, value in self.model.parameterization.constant_params().items()])
        lines.append('')
        add_section([[p, params[p]] for p in self.model.parameterization.derived_params()])
        if hasattr(params, 'chi2_names'):
            from cobaya.conventions import _chi2, _separator
            labels.update(
                odict([[p, r'\chi^2_{\rm %s}' % (p.replace(_chi2 + _separator, '').replace("_", r"\ "))]
                       for p in params.chi2_names]))
            add_section([[chi2, params[chi2]] for chi2 in params.chi2_names])
        return "\n".join(lines)

    def dump_getdist(self):
        if not self.output:
            return
        if not self.ignore_prior:
            # .minimum files currently assumed to be maximum of posterior
            getdist_bf = self.getdist_point_text(self.minimum, minuslogpost=self.minimum['minuslogpost'])
            print("\n" + getdist_bf)
            with open(os.path.join(self.output.folder, self.output.prefix + '.minimum'), 'w') as f:
                f.write(getdist_bf)
