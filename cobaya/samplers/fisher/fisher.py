"""
.. module:: samplers.fisher

:Synopsis: Fisher matrix estimator
:Author: ???
"""

# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

# Global
import numpy as np
import logging

# Local
from cobaya.sampler import Sampler
from cobaya.log import HandledException
from cobaya.mpi import get_mpi_rank, get_mpi_size, get_mpi_comm
from cobaya.collection import Collection


class fisher(Sampler):
    def initialize(self):
        self.log.info("Initializing.")
        # Example for setting default options
        print("+++ For option '%s' got value '%s'. Add more options in fisher.yaml." % (
            "example_option", self.example_option))
        # Prepare the list of "samples" (posterior evaluations)
        self.collection = Collection(self.model, self.output)
        # Prepare vectors to store derivatives
        # ... up to you
        # Prepare the matrix
        self.fisher_matrix = np.eye(self.model.prior.d())

    def run(self):
        self.log.info("Starting Fisher matrix estimation.")
        # Getting info about parameters
        param_names = list(self.model.parameterization.sampled_params())
        print("Parameter names:", param_names)
        bounds = self.model.prior.bounds()
        print("Parameter bounds:", zip(param_names, bounds))
        # Get n samples from the prior, evaluate the likelihood,
        # and adding them to our collection of points
        #    (mind that self.model.prior.sample always returns an array, so,
        #     for a single sample do `sample = self.model.prior.sample()[0]`)
        initial_sample = self.model.prior.sample(4)
        for point in initial_sample:
            logpost, logpriors, loglikes, derived = self.model.logposterior(point)
            self.collection.add(
                point, logpost=logpost, logpriors=logpriors, loglikes=loglikes)
        print("--------------------")
        print("Our collection so far")
        print(self.collection)
        # Access one of the stored points and its values for parameters and logprobs
        i_point = 2  # the 3rd point!
        param = "x_01"
        print("For point #%d, " % i_point, "the value of parameter '%s' is %g" %
              (param, self.collection[i_point][param]))
        print("and the values of the logprior and loglikelihood are respectively %g, %g" %
              (-self.collection[i_point]["minuslogprior"],
               -0.5*self.collection[i_point]["chi2"]))
        # Accessing a column:
        print("All the values for parameter", param, "are\n", self.collection[param])
        # If you want to modify values for parameter/posteriors/etc, you have to COPY them
        # otherwise they'll be modified inside the table too!
        copy_of_minuslogpost = self.collection["minuslogpost"].values.copy()
        # Let me know if you need to know how to do anything else

    def close(self, *args):
        """
        Anything needed to be finished once convergence has been achieved (or not)
        """
        # Make sure that the last points are written
        self.collection._out_update()

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.
        """
        return {"evaluations": self.collection, "fisher_matrix": self.fisher_matrix}
