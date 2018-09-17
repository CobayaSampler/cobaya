"""
.. module:: samplers.evaluate

:Synopsis: Dummy "sampler": simply evaluates the likelihood.
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import division, print_function, absolute_import

# Global
import numpy as np
from collections import OrderedDict as odict

# Local
from cobaya.sampler import Sampler
from cobaya.collection import Collection
from cobaya.log import HandledException


class evaluate(Sampler):

    def initialize(self):
        """
        Creates a 1-point collection to store the point
        at which the posterior is evaluated.
        """
        self.one_point = Collection(self.model, self.output, initial_size=1, name="1")
        self.log.info("Initialized!")

    def run(self):
        """
        First gets a reference point. If a single reference point is not given,
        the point is sampled from the reference pdf. If that one is not defined either,
        the point is sampled from the prior.

        Then it evaluates the prior and likelihood(s) and stores them in the one-member
        sample collection.
        """
        self.log.info("Looking for a reference point with non-zero prior.")
        reference_point = self.model.prior.reference()
        reference_point = odict(
            zip(self.model.parameterization.sampled_params(), reference_point))
        for p, v in (self.override or {}).items():
            if p not in reference_point:
                self.log.error("Parameter '%s' used in override not known. "
                               "Known parameters names are %r.",
                               p, self.model.parameterization.sampled_params())
                raise HandledException
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
        for i, name in enumerate(self.model.prior):
            self.log.info("   logprior_" + name + " = %g" % self.logposterior.logpriors[i])
        if sum(self.logposterior.logpriors) > -np.inf:
            self.log.info("log-likelihood = %g", sum(self.logposterior.loglikes))
            for i, name in enumerate(self.model.likelihood):
                self.log.info("   chi2_" + name + " = %g" % (-2 * self.logposterior.loglikes[i]))
            self.log.info("Derived params:")
            for name, value in zip(self.model.parameterization.derived_params(),
                                   self.logposterior.derived):
                self.log.info("   " + name + " = %g" % value)
        else:
            self.log.info("Likelihoods and derived parameters not computed, "
                          "since the prior is null.")

    def close(self, exception_type, exception_value, traceback):
        """
        Writes the output: the point and its prior, posterior and likelihood.
        """
        self.one_point._out_update()

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the sequentially discarded live points.
        """
        return {"sample": self.one_point}
