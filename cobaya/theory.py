"""
.. module:: theory

:Synopsis: Prototype theory class and theory loader
:Author: Jesus Torrado

If you are using a experimental likelihood, chances are the you will need a theoretical
code to compute the observables needed to compute the likelihood.

This module contains the prototype of the theory code and the loader for the requested
code.

.. note::

   At this moment, of all modules of cobaya, this is the one with the least fixed
   structure. Don't pay much attention to it for now. Just go on to the documentation of
   :doc:`CAMB <theory_camb>` and :doc:`CLASS <theory_class>`.

"""

# Python 2/3 compatibility
from __future__ import division

# Global
import logging

# Default options for all subclasses
class_options = {"speed": -1}


# Theory code prototype
class Theory(object):
    """Prototype of the theory class."""

    def initialize(self):
        """
        Initializes the theory code: imports the theory code, if it is an external one,
        and makes any necessary preparations.
        """
        pass

    def needs(self, arguments):
        """
        Function to be called by the likelihoods at their initialization,
        to specify their requests.
        Its specific behaviour for a code must be defined.
        """
        pass

    def compute(self, **parameter_values_and_derived_dict):
        """
        Takes a dictionary of parameter values and computes the products needed by the
        likelihood.
        If passed a keyword `derived` with an empty dictionary, it populates it with the
        value of the derived parameters for the present set of sampled and fixed parameter
        values.
        """
        pass

    def close(self):
        """Finalizes the theory code, if something needs to be done
        (releasing memory, etc.)"""
        pass

    # Generic methods: do not touch these

    def __init__(self, info_input_params, info_output_params, info_theory, modules=None,
                 timing=None):
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        self.input_params = info_input_params
        self.output_params = info_output_params
        self.path_install = modules
        # Load info of the code
        for k in info_theory:
            setattr(self, k, info_theory[k])
        # Timing
        self.timing = timing
        self.n = 0
        self.time_avg = 0
        self.initialize()

    def d(self):
        return len(self.sampled)

    def __exit__(self, exception_type, exception_value, traceback):
        if self.timing:
            self.log.info("Average 'compute' evaluation time: %g s  (%d evaluations)" %
                          (self.time_avg, self.n))
        self.close()
