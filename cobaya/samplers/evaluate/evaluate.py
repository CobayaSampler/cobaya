"""
.. module:: samplers.evaluate

:Synopsis: Dummy "sampler": simply evaluates the likelihood.
:Author: Jesus Torrado

"""

# Global
import numpy as np
from typing import Mapping

# Local
from cobaya.sampler import Sampler
from cobaya.collection import Collection
from cobaya.log import LoggedError


class evaluate(Sampler):
    override: Mapping[str, float]

    def initialize(self):
        """
        Creates a 1-point collection to store the point
        at which the posterior is evaluated.
        """
        try:
            self.N = int(self.N)
        except:
            raise LoggedError(
                self.log,
                "Could not convert the number of samples to an integer: %r", self.N)
        self.one_point = Collection(
            self.model, self.output, initial_size=self.N, name="1")
        self.log.info("Initialized!")

    def _run(self):
        """
        First gets a reference point. If a single reference point is not given,
        the point is sampled from the reference pdf. If that one is not defined either,
        the point is sampled from the prior.

        Then it evaluates the prior and likelihood(s) and stores them in the one-member
        sample collection.
        """
        for i in range(self.N):
            if self.N > 1:
                self.log.info("Evaluating sample #%d ------------------------------",
                              i + 1)
            self.log.info("Looking for a reference point with non-zero prior.")
            reference_point = self.model.prior.reference()
            reference_point = dict(
                zip(self.model.parameterization.sampled_params(), reference_point))
            for p, v in (self.override or {}).items():
                if p not in reference_point:
                    raise LoggedError(
                        self.log, "Parameter '%s' used in override not known. "
                                  "Known parameters names are %r.",
                        p, self.model.parameterization.sampled_params())
                reference_point[p] = v
            self.log.info("Reference point:\n   " + "\n   ".join(
                ["%s = %g" % pv for pv in reference_point.items()]))
            self.log.info("Evaluating prior and likelihoods...")
            self.logposterior = self.model.logposterior(reference_point)
            self.one_point.add(
                list(reference_point.values()), derived=self.logposterior.derived,
                logpost=self.logposterior.logpost, logpriors=self.logposterior.logpriors,
                loglikes=self.logposterior.loglikes)
            self.log.info("log-posterior  = %g", self.logposterior.logpost)
            self.log.info("log-prior      = %g", sum(self.logposterior.logpriors))
            for j, name in enumerate(self.model.prior):
                self.log.info(
                    "   logprior_" + name + " = %g", self.logposterior.logpriors[j])
            if sum(self.logposterior.logpriors) > -np.inf:
                self.log.info("log-likelihood = %g", sum(self.logposterior.loglikes))
                for j, name in enumerate(self.model.likelihood):
                    self.log.info(
                        "   chi2_" + name + " = %g", (-2 * self.logposterior.loglikes[j]))
                self.log.info("Derived params:")
                for name, value in zip(self.model.parameterization.derived_params(),
                                       self.logposterior.derived):
                    self.log.info("   " + name + " = %g", value)
            else:
                self.log.info("Likelihoods and derived parameters not computed, "
                              "since the prior is null.")
        # Write the output: the point and its prior, posterior and likelihood.
        self.one_point.out_update()

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the sequentially discarded live points.
        """
        return {"sample": self.one_point}
