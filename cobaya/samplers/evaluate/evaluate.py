"""
.. module:: samplers.evaluate

:Synopsis: Dummy "sampler": simply evaluates the likelihood.
:Author: Jesus Torrado

This is a *dummy* sampler that just evaluates the likelihood in a given (or sampled)
reference point.

You can use it to test your likelihoods.

To use it, simply make the ``sampler`` block:

.. code-block:: yaml

   sampler:
     evaluate:

If you want to fix the evaluation point, give just a value as a *reference*
for the parameters in the input file. For example:

.. code-block:: yaml

   params:
     a:
       prior:
         min: -1
         max:  1
       ref: 0.5
     b:
       prior:
         min: -1
         max:  1
       ref:
         dist: norm
         loc: 0
         scale: 0.1
     c:
       prior:
         min: -1
         max:  1

In this case, the posterior will be evaluated at ``a=0.5``, ``b`` sampled from a normal
pdf centred at ``0`` with standard deviation ``0.1``, and ``c`` will be sampled uniformly
between ``-1`` and ``1``.

"""

# Python 2/3 compatibility
from __future__ import division

# Local
from cobaya.sampler import Sampler
from cobaya.collection import Collection

# Logger
import logging
log = logging.getLogger(__name__)


class evaluate(Sampler):

    def initialise(self):
        """
        Creates a 1-point collection to store the point
        at which the posterior is evaluated.
        """ 
        self.one_point = Collection(self.parametrisation, self.likelihood,
                                    self.output, initial_size=1, name="1")
        log.info("Initialised!")
        
    def run(self):
        """
        First gets a reference point. If a single reference point is not given,
        the point is sampled from the reference pdf, and if that one is not defined either,
        the point is sampled from the prior.

        Then it evaluates the prior and likelihood(s) and stores them in the one-member 
        sample collection.
        """
        reference_point = self.prior.reference()
        log.info("Reference point:\n   "+
               "\n   ".join(["%s = %g"%(p,reference_point[i])
                             for i,p in enumerate(self.parametrisation.sampled_params())]))
        log.info("Evaluating prior and likelihoods...")
        logpost, logprior, logliks, derived = self.logposterior(reference_point)
        self.one_point.add(reference_point, derived=derived,
                           logpost=logpost, logprior=logprior, logliks=logliks)
        log.info("log-posterior  = %g", logpost)
        log.info("log-prior      = %g", logprior)
        log.info("log-likelihood = %g", sum(logliks))
        log.info("   "+"\n   ".join(["chi2_"+name+" = %g"%(-2*logliks[i])
                  for i,name in enumerate(self.likelihood)]))
        log.info("Done!")

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
