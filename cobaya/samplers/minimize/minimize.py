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
import logging

# Local
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi_rank, get_mpi_size, get_mpi_comm
from cobaya.collection import OnePoint
from cobaya.log import HandledException


class minimize(Sampler):
    def initialize(self):
        """Prepares the arguments for `scipy.minimize`."""
        if not get_mpi_rank():
            self.log.info("Initializing")
        self.logp = ((lambda x: self.model.logposterior(x, make_finite=True)[0])
                     if not self.ignore_prior else
                     (lambda x: sum(self.model.loglikes(x, return_derived=True)[0])))
        # Initial point: sample from reference and make sure that it has finite lik/post
        this_logp = -np.inf
        while not np.isfinite(this_logp):
            initial_point = self.model.prior.reference()
            this_logp = self.logp(initial_point)
        self.kwargs = {
            "fun": (lambda x: -self.logp(x)),
            "x0": initial_point,
            "bounds": self.model.prior.bounds(confidence_for_unbounded=0.999),
            "tol": self.tol,
            "options": {
                "maxiter": self.maxiter,
                "disp": (self.log.getEffectiveLevel() == logging.DEBUG)}}
        self.kwargs.update(self.override or {})
        self.log.debug("Arguments for scipy.optimize.minimize:\n%r", self.kwargs)

    def run(self):
        """
        Runs `scipy.minimize`
        """
        self.log.info("Starting minimization.")
        self.result = scpminimize(**self.kwargs)
        if self.result.success:
            self.log.info("Finished succesfully.")
        else:
            self.log.error("Finished Unsuccesfully.")

    def close(self, *args):
        """
        Determines success (or not), chooses best (if MPI)
        and produces output (if requested).
        """
        # If something failed
        if not hasattr(self, "result"):
            return
        if get_mpi_size():
            results = get_mpi_comm().gather(self.result, root=0)
            if not get_mpi_rank():
                self.result = results[np.argmin([r.fun for r in results])]
        if not get_mpi_rank():
            if not self.result.success:
                self.log.error("Maximization failed! Here is the `scipy` raw result:\n%r",
                               self.result)
                raise HandledException
            self.log.info("log%s maximized at %g",
                          "likelihood" if self.ignore_prior else "posterior",
                          -self.result.fun)
            post = self.model.logposterior(self.result.x)
            recomputed_max = sum(post.loglikes) if self.ignore_prior else post.logpost
            if not np.allclose(-self.result.fun, recomputed_max):
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
            return {"maximum": self.maximum, "OptimizeResult": self.result}
