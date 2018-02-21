# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import numpy as np
from scipy.optimize import minimize as scpminimize

# Local
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi_rank, get_mpi_size, get_mpi_comm
from cobaya.collection import OnePoint
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)


class minimize(Sampler):
    def initialise(self):
        """Prepares the arguments for `scipy.minimize`."""
        if not get_mpi_rank():
            log.info("Initializing")
        # Initial point: sample from reference and make sure that it has finite lik/post
        logp = -np.inf
        while not np.isfinite(logp):
            initial_point = self.prior.reference()
            logp = self.logposterior(initial_point, ignore_prior=self.ignore_prior)[0]
        self.kwargs = {
            "fun": (lambda x: -self.logposterior(
                x, ignore_prior=self.ignore_prior, make_finite=True)[0]),
            "x0": initial_point,
            "bounds": self.prior.limits(),
            "tol": self.tol,
            "options": {
                "maxiter": self.maxiter,
                "disp": (log.getEffectiveLevel() == logging.DEBUG)}}
        self.kwargs.update(self.override or {})

    def run(self):
        """
        Runs `scipy.minimize`
        """
        log.info("Starting minimization.")
        self.result = scpminimize(**self.kwargs)
        if self.result.success:
            log.info("Finished succesfuly.")
        else:
            log.error("Finished UNsuccesfuly.")

    def close(self):
        """
        Determines success (or not), chooses best (if MPI)
        and produces output (if requested).
        """
        if get_mpi_size():
            results = get_mpi_comm().gather(self.result, root=0)
            if not get_mpi_rank():
                self.result = results[np.argmin([r.fun for r in results])]
        if not get_mpi_rank():
            log.info("log%s maximised at %g",
                     "likelihood" if self.ignore_prior else "posterior",
                     -self.result.fun)
            logpost, logprior, logliks, derived = self.logposterior(self.result.x)
            recomputed_max = sum(logliks) if self.ignore_prior else logpost
            if not np.allclose(-self.result.fun, recomputed_max):
                log.error("Cannot reproduce result. Something bad happened. "
                          "Recomputed max: %g at %r", recomputed_max, self.result.x)
                raise HandledException
            self.maximum = OnePoint(
                self.parametrization, self.likelihood, self.output, name="maximum",
                extension=("likelihood" if self.ignore_prior else "posterior"))
            self.maximum.add(self.result.x, derived=derived, logpost=logpost,
                             logprior=logprior, logliks=logliks)
            self.maximum.out_update()
            log.info("Result written in '%s'", self.maximum.file_name)

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The ``OnePoint`` that maximises the posterior or likelihood (depending on
           `ignore_prior`), and the ``scipy.optimize.OptimizeResult`` instance.
        """
        if not get_mpi_rank():
            return {"maximum": self.maximum, "OptimizeResult": self.result}
