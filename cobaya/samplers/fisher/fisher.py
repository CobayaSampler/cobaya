"""
.. module:: samplers.fisher

:Synopsis: Fisher matrix estimator
:Author: ???
"""

# Python 2/3 compatibility
from __future__ import absolute_import, division

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
        self.collection = Collection(
            self.parameterization, self.likelihood, self.output)
        # Prepare vectors to store derivatives
        # ... up to you
        # Prepare the matrix
        self.fisher_matrix = np.eye(self.prior.d())

    def run(self):
        self.log.info("Starting Fisher matrix estimation.")
        # Get the boundaries of the prior
        bounds = self.prior.bounds()
        # Get n samples from the prior
        n = 5 * self.prior.d()  # e.g. # samples = 5 * dimension
        initial_sample = self.prior.sample(n)
        # Evaluate the posterior at those points and add them to the samples collection
        for point in initial_sample:
            logpost, logpriors, loglikes, derived = self.logposterior(point)
            self.collection.add(
                point, logpost=logpost, logpriors=logpriors, loglikes=loglikes)
        # Access one of the stored points (e.g. the 3rd one),
        # and its value for one of the parameters and the posterior
        i_point = 2
        i_param = 0
        point = self.collection.data.iloc[i_point]
        param_names = list(self.parameterization.sampled_params())
        print "Point #%d is ", point
        print "and its value for param", param_names[i_param], "is", point[param_names[i_param]]
        print "and its evidence is", point["minuslogpost"]
        # Accessing a column:
        print "All the values for parameter", param_names[i_param], "are"
        print self.collection[param_names[i_param]]
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
