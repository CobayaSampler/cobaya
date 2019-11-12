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

# Local
from cobaya.conventions import _input_params, _output_params
from cobaya.component import CobayaComponent

# Default options for all subclasses
class_options = {"speed": -1}


# Theory code prototype
class Theory(CobayaComponent):
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

    # Generic methods: do not touch these

    def d(self):
        """
        Dimension of the input vector.

        NB: Different from dimensionality of the sampling problem, e.g. this may include
        fixed input parameters.
        """
        return len(self.input_params)
