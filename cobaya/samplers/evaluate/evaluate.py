"""
.. module:: samplers.evaluate

:Synopsis: Dummy "sampler": simply evaluates the likelihood.
:Author: Jesus Torrado

"""

from collections.abc import Mapping

import numpy as np

import cobaya.mpi as mpi
from cobaya.collection import SampleCollection
from cobaya.log import LoggedError
from cobaya.sampler import Sampler


class Evaluate(Sampler):
    sampler_type = "uncorrelated"
    file_base_name = "evaluate"

    override: Mapping[str, float]
    N: int

    def initialize(self):
        """
        Creates a 1-point collection to store the point
        at which the posterior is evaluated.
        """
        try:
            self.N = int(self.N)
        except ValueError:
            raise LoggedError(
                self.log,
                "Could not convert the number of samples to an integer: %r",
                self.N,
            )
        self.points = SampleCollection(self.model, self.output, name=str(1 + mpi.rank()))
        self.log.info("Initialized!")

    def run(self):
        """
        First gets a reference point. If a single reference point is not given,
        the point is sampled from the reference pdf. If that one is not defined either,
        the point is sampled from the prior.

        Then it evaluates the prior and likelihood(s) and stores them in the one-member
        sample collection.
        """
        for i in range(self.N):
            if self.N > 1:
                self.log.info(
                    "Evaluating sample #%d ------------------------------", i + 1
                )
            self.log.info("Looking for a reference point with non-zero prior.")
            reference_values = self.model.prior.reference(random_state=self._rng)
            reference_point = dict(
                zip(self.model.parameterization.sampled_params(), reference_values)
            )
            for p, v in (self.override or {}).items():
                if p not in reference_point:
                    raise LoggedError(
                        self.log,
                        "Parameter '%s' used in override not known. "
                        "Known parameters names are %r.",
                        p,
                        self.model.parameterization.sampled_params(),
                    )
                reference_point[p] = v
            self.log.info(
                "Reference point:\n   "
                + "\n   ".join(["%s = %g" % pv for pv in reference_point.items()])
            )
            self.log.info("Evaluating prior and likelihoods...")
            self.logposterior = self.model.logposterior(reference_point)
            self.points.add(
                list(reference_point.values()),
                derived=self.logposterior.derived,
                logpost=self.logposterior.logpost,
                logpriors=self.logposterior.logpriors,
                loglikes=self.logposterior.loglikes,
            )
            self.log.info("log-posterior  = %g", self.logposterior.logpost)
            self.log.info("log-prior      = %g", self.logposterior.logprior)
            for j, name in enumerate(self.model.prior):
                self.log.info(
                    "   logprior_" + name + " = %g", self.logposterior.logpriors[j]
                )
            if self.logposterior.logprior > -np.inf:
                self.log.info("log-likelihood = %g", self.logposterior.loglike)
                for j, name in enumerate(self.model.likelihood):
                    self.log.info(
                        "   chi2_" + name + " = %g", (-2 * self.logposterior.loglikes[j])
                    )
                self.log.info("Derived params:")
                for name, value in zip(
                    self.model.parameterization.derived_params(),
                    self.logposterior.derived,
                ):
                    self.log.info("   " + name + " = %g", value)
            else:
                self.log.info(
                    "Likelihoods and derived parameters not computed, "
                    "since the prior is null."
                )
        # Write the output: the point and its prior, posterior and likelihood.
        self.points.out_update()

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``SampleCollection`` containing the
           sequentially discarded live points.
        """
        return {"sample": self.points}
