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
   structure. Don't pay much attention to it for now. Just go on to the installation
   instructions for :doc:`CAMB <theory_camb>` and :doc:`CLASS <theory_class>`.

"""

# Python 2/3 compatibility
from __future__ import division

# Global
import sys
import os
from collections import OrderedDict as odict
from importlib import import_module

# Local
from cobaya.conventions import *
from cobaya.tools import get_folder
from cobaya.input import load_input_and_defaults, load_params, get_updated_params_info
from cobaya.log import HandledException


# Theory code prototype
class Theory():
    """Prototype of the theory class."""
    
    def initialise(self):
        """
        Initialises the theory code: imports the theory code, if it is an external one, 
        and makes any necessary preparations.
        """
        pass

    def needs(self, arguments):
        """
        Function to be called by the likelihoods at their initialisation,
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
        """Finalises the theory code, if something needs to be done
        (releasing memory, etc.)"""
        pass

    # Generic methods: do not touch these

    def __init__(self, info_theory, info_params=None):
        # Create class-level default options
        self._parent_defaults = odict([["speed", 1]])
        # Instead of using a defaults.yaml, we will simply initialise a 'path' attribute
        self.path = None
        # Load info of the code
        self._updated_info = load_input_and_defaults(self, info_theory, kind=input_theory)
        load_params(self, info_params, allow_unknown_prefixes=[""])
        self._updated_info_params = get_updated_params_info(self)
        self.initialise()

#    def args(self):
#        return self._args

    def updated_info(self):
        return self._updated_info
    def updated_info_params(self):
        return self._updated_info_params

    def d(self):
        return len(self.sampled)

    def sampled_params(self):
        return self.sampled

    def sampled_params_names(self):
        return self.sampled.keys()

#    def params_values(self):
#        """Returns current params values as dictionary."""
#        return self._current
    
#    def params_dict(self, params_values):
#        return dict([[p,v] for p,v in zip(self.params_names(),params_values)])

#    def args_and_params(self, params_values):
#        a_p = self.args().copy()
#        a_p.update(self.params_dict(params_values))
#        return a_p


def get_Theory(info_theory, info_params=None):
    """
    Auxiliary function to retrieve and initialise the requested theory code.
    """
    if not info_theory:
        return None
    name = info_theory.keys()[0]
    try:
        class_folder = get_folder(name, input_theory, sep=".", absolute=False)
        Theory_class = getattr(
            import_module(class_folder, package=package), name.lower())
    except ImportError:
        log.error("Theory '%s' not found.", name.lower())
        raise HandledException
    return Theory_class(info_theory[name], info_params=info_params)
