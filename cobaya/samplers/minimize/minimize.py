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
from cobaya.mpi import get_mpi_rank, get_mpi_size, get_mpi_comm
from cobaya.collection import OnePoint
from cobaya.log import HandledException
from cobaya.tools import read_dnumber


# Handling scpiy vs BOBYQA
evals_attr = {"scipy": "fun", "bobyqa": "f"}


class minimize(Sampler):
    def initialize(self):
        """Prepares the arguments for `scipy.minimize`."""
        if not get_mpi_rank():
            self.log.info("Initializing")
        self.max_evals = read_dnumber(self.max_evals, self.model.prior.d())
        # Configure target
        self.logp = (
            (lambda x: self.model.logposterior(x, make_finite=True)[0])
            if not self.ignore_prior else
            (lambda x: sum(self.model.loglikes(x, return_derived=True)[0])))
        # Initial point:
        # BASIC: sample from reference and make sure that it has finite like/post
        this_logp = -np.inf
        while not np.isfinite(this_logp):
            initial_point = self.model.prior.reference()
            this_logp = self.logp(initial_point)
        self.linear_transform = np.eye(self.model.prior.d())
        # Configure transformation for target
        self.logp_transf = lambda x: self.logp(self.transform(x))
        # Configure method
        if self.method.lower() == "bobyqa":
            self.minimizer = pybobyqa.solve
            self.kwargs = {
                "objfun": (lambda x: -self.logp_transf(x)),
                "x0": initial_point,
                "bounds": np.array(list(
                    zip(*self.model.prior.bounds(confidence_for_unbounded=0.999)))),
                "seek_global_minimum": True,
                "scaling_within_bounds": hasattr(self, "covmat"),

                "maxfun": int(self.max_evals),
                }
            self.log.debug("Arguments for pybobyqa.solve:\n%r",
                           {k:v for k,v in self.kwargs.items() if k != "objfun"})
        elif self.method.lower() == "scipy":
            self.minimizer = scpminimize
            self.kwargs = {
                "fun": (lambda x: -self.logp_transf(x)),
                "x0": initial_point,
                "bounds": self.model.prior.bounds(confidence_for_unbounded=0.999),
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

    def transform(self, point):
        return(self.linear_transform.dot(point))

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
            if not get_mpi_rank():
                self.result = results[np.argmin(
                    [getattr(r, evals_attr_) for r in results])]
        if not get_mpi_rank():
            if not self.success:
                self.log.error("Maximization failed! Here is the `scipy` raw result:\n%r",
                               self.result)
                raise HandledException
            logpost = -np.array(getattr(self.result, evals_attr_))
            self.log.info("log%s maximized at %g",
                          "likelihood" if self.ignore_prior else "posterior", logpost)
            post = self.model.logposterior(self.result.x)
            recomputed_max = sum(post.loglikes) if self.ignore_prior else post.logpost
            if not np.allclose(logpost, recomputed_max):
                self.log.error("Cannot reproduce result. Something bad happened. "
                               "Recomputed max: %g at %r", recomputed_max, self.result.x)
                raise HandledException
            self.maximum = OnePoint(
                self.model, self.output, name="maximum",
                extension=("likelihood" if self.ignore_prior else "posterior"))
            self.maximum.add(self.result.x, derived=post.derived, logpost=post.logpost,
                             logpriors=post.logpriors, loglikes=post.loglikes)
            self.log.info("Parameter values at maximum:\n%s"%self.maximum.data.to_string())
            self.maximum._out_update()

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The :class:`OnePoint` that maximizes the posterior or likelihood (depending on
           ``ignore_prior``), and the `scipy.optimize.OptimizeResult
           <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`_
           instance.
        """
        if not get_mpi_rank():
            return {"maximum": self.maximum, "result_object": self.result}
