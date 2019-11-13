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
from cobaya.conventions import _external, _theory
from cobaya.component import CobayaComponent, ComponentCollection
from cobaya.tools import get_class


# Theory code prototype
class Theory(CobayaComponent):
    """Prototype of the theory class."""
    # Default options for all subclasses
    class_options = {"speed": -1}

    def initialize(self):
        """
        Initializes the theory code: imports the theory code, if it is an external one,
        and makes any necessary preparations.
        """
        pass

    def needs(self, **arguments):
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


class TheoryCollection(ComponentCollection):
    """
    Initializes the list of theory codes.
    """

    def __init__(self, info_theory, path_install=None, timing=None):
        super(TheoryCollection, self).__init__()
        self.set_logger("theory")
        # TODO: multiple theory dependence, requirements etc.
        assert not info_theory or len(info_theory) < 2, "Currently only one theory actually supported"

        if info_theory:
            for name, info in info_theory.items():
                # If it has an "external" key, wrap it up. Else, load it up
                if _external in info:
                    theory_class = info[_external]
                else:
                    theory_class = get_class(name, kind=_theory)
                self[name] = theory_class(info, path_install=path_install, timing=timing, name=name)

    def __getattribute__(self, name):
        if not name.startswith('__'):
            # support old single-theory syntax
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                pass
            for theory in self.values():
                try:
                    return getattr(theory, name)
                except AttributeError:
                    pass
        return object.__getattribute__(self, name)
