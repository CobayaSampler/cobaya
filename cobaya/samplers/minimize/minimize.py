"""
.. module:: samplers.minimize

:Synopsis: Posterior/likelihood *maximization* (i.e. chi^2 minimization).
:Author: Jesus Torrado (though it's just a wrapper of ``scipy.optimize.minimize``)

This is a **maximizator** for posteriors or likelihoods, using
`scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

It is pretty self-explanatory: just look at the comments on the defaults below.

It is recommended to run a couple of parallel MPI processes:
it will finally pick the best among the results.
"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import numpy as np
from scipy.optimize import minimize as scpminimize
import pybobyqa
import logging

# Local
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi_size, get_mpi_comm, am_single_or_primary_process
from cobaya.collection import OnePoint
from cobaya.log import HandledException
from cobaya.tools import read_dnumber, choleskyL


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
                covmat = collection_in.cov()
        if initial_point is None:
            this_logp = -np.inf
            while not np.isfinite(this_logp):
                initial_point = self.model.prior.reference()
                this_logp = self.logp(initial_point)
            self.log.info("Starting from random initial point:")
        self.log.info(dict(zip(self.model.parameterization.sampled_params(), initial_point)))
        # Cov and affine transformation
        covmat = None
        self._affine_transform_matrix = None
        self._inv_affine_transform_matrix = None
        self._affine_transform_baseline = None
        if covmat is not None:
            sigmas_diag, L = choleskyL(covmat, return_scale_free=True)
            self._affine_transform_matrix = np.linalg.inv(sigmas_diag)
            self._inv_affine_transform_matrix = sigmas_diag
            self._affine_transform_baseline = initial_point
            # Transforms to space where initial point is at centre, and cov is normalised
            self.affine_transform = lambda x: (
                self._affine_transform_matrix.dot(x - self._affine_transform_baseline))
            self.inv_affine_transform = lambda x: (
                self._inv_affine_transform_matrix.dot(x) + self._affine_transform_baseline)
        else:
            self.affine_transform = lambda x: x
            self.inv_affine_transform = lambda x: x
        bounds = self.model.prior.bounds(
            confidence_for_unbounded=self.confidence_for_unbounded)
        # Re-scale
        self.logp_transf = lambda x: self.logp(self.inv_affine_transform(x))
        initial_point = self.affine_transform(initial_point)
        bounds = self.affine_transform(bounds)
        # Configure method
        if self.method.lower() == "bobyqa":
            self.minimizer = pybobyqa.solve
            self.kwargs = {
                "objfun": (lambda x: -self.logp_transf(x)),
                "x0": initial_point,
                "bounds": np.array(list(zip(*bounds))),
                "seek_global_minimum": True,
                "maxfun": int(self.max_evals)}
            self.kwargs.update(self.override or {})
            self.log.debug("Arguments for pybobyqa.solve:\n%r",
                           {k:v for k,v in self.kwargs.items() if k != "objfun"})
        elif self.method.lower() == "scipy":
            self.minimizer = scpminimize
            self.kwargs = {
                "fun": (lambda x: -self.logp_transf(x)),
                "x0": initial_point,
                "bounds": bounds,
                "tol": self.tol,
                "options": {
                    "maxiter": self.max_evals,
                    "disp": (self.log.getEffectiveLevel() == logging.DEBUG)}}
            self.kwargs.update(self.override or {})
            self.log.debug("Arguments for scipy.optimize.minimize:\n%r",
                           {k:v for k,v in self.kwargs.items() if k != "fun"})
        else:
            methods = ["bobyqa", "scipy"]
            self.log.error(
                "Method '%s' not recognized. Try one of %r.", self.method, methods)
            raise HandledException

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
            self.log.info("Finished succesfully!")
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
            self.log.error("Finished unsuccesfully." +
                           ("Reason: " + reason if reason else ""))

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
            if am_single_or_primary_process():
                self.result = results[np.argmin(
                    [getattr(r, evals_attr_) for r in results])]
        if am_single_or_primary_process():
            if not self.success:
                self.log.error("Minimization failed! Here is the `scipy` raw result:\n%r",
                               self.result)
                raise HandledException
            logpost = -np.array(getattr(self.result, evals_attr_))
            x_minimum = self.inv_affine_transform(self.result.x)
            self.log.info("log%s maximized at %g",
                          "likelihood" if self.ignore_prior else "posterior", logpost)
            post = self.model.logposterior(x_minimum)
            recomputed_max = sum(post.loglikes) if self.ignore_prior else post.logpost
            if not np.allclose(logpost, recomputed_max):
                self.log.error("Cannot reproduce result. Something bad happened. "
                               "Recomputed max: %g at %r", recomputed_max, x_minimum)
                raise HandledException
            self.minimum = OnePoint(
                self.model, self.output, name="maximum",
                extension=("likelihood" if self.ignore_prior else "posterior"))
            self.minimum.add(x_minimum, derived=post.derived, logpost=post.logpost,
                             logpriors=post.logpriors, loglikes=post.loglikes)
            self.log.info(
                "Parameter values at maximum:\n%s", self.minimum.data.to_string())
            self.minimum._out_update()

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The :class:`OnePoint` that maximizes the posterior or likelihood (depending on
           ``ignore_prior``), the ``result_object`` instance of `scipy
           <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`_
           or `pyBOBYQA
           <https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/userguide.html>`_,
           and the inverse of the affine transform under which the minimizer has worked, 
           as a matrix ``M`` and a point ``X0``, from which the real space points can be
           obtained as :math:`x = M x^\prime + X0`, where :math:`x^\prime` represents the
           parameter space used in ``result_object`` (returns ``None`` for both ``M`` and
           ``X0`` if no transformation was applied).
        """
        if am_single_or_primary_process():
            return {"minimum": self.minimum, "result_object": self.result,
                    "M": self._inv_affine_transform_matrix,
                    "X0": self._affine_transform_baseline}
