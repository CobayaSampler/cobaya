"""
.. module:: samplers.evaluate

:Synopsis: Dummy "sampler": simply evaluates the likelihood.
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import division

# Local
from cobaya.sampler import Sampler
from cobaya.collection import Collection


class evaluate(Sampler):

    def initialise(self):
        """
        Creates a 1-point collection to store the point
        at which the posterior is evaluated.
        """
        self.one_point = Collection(self.parametrization, self.likelihood,
                                    self.output, initial_size=1, name="1")
        self.log.info("Initialised!")

    def run(self):
        """
        First gets a reference point. If a single reference point is not given,
        the point is sampled from the reference pdf, and if that one is not defined either,
        the point is sampled from the prior.

        Then it evaluates the prior and likelihood(s) and stores them in the one-member
        sample collection.
        """
        reference_point = self.prior.reference()
        self.log.info("Reference point:\n   " +
                      "\n   ".join(["%s = %g" % (p,reference_point[i])
                                    for i,p in enumerate(self.parametrization.sampled_params())]))
        self.log.info("Evaluating prior and likelihoods...")
        logpost, logprior, logliks, derived = self.logposterior(reference_point)
        self.one_point.add(reference_point, derived=derived,
                           logpost=logpost, logprior=logprior, logliks=logliks)
        self.log.info("log-posterior  = %g", logpost)
        self.log.info("log-prior      = %g", logprior)
        self.log.info("log-likelihood = %g", sum(logliks))
        self.log.info("   "+"\n   ".join(["chi2_"+name+" = %g"%(-2*logliks[i])
                                          for i,name in enumerate(self.likelihood)]))
        self.log.info("Done!")

    def close(self):
        """
        Writes the output: the point and its prior, posterior and likelihood.
        """
        self.one_point.out_update()

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the sequentially discarded live points.
        """
        return {"sample": self.one_point}
