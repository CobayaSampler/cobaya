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

import inspect
from collections import deque
# Local
from cobaya.conventions import _external, kinds, _requires, _params
from cobaya.component import CobayaComponent, ComponentCollection
from cobaya.tools import get_class
from cobaya.log import LoggedError
from cobaya.tools import get_class_methods


class Theory(CobayaComponent):
    """Base class theory that can calculate something."""
    # Default options for all subclasses
    class_options = {"speed": -1, "stop_at_error": False}

    def __init__(self, info={}, name=None, timing=None, path_install=None,
                 initialize=True, standalone=True):

        super(Theory, self).__init__(info, name=name, timing=timing,
                                     path_install=path_install, initialize=initialize,
                                     standalone=standalone)

        self.provider = None  # set to Provider instance before calculations
        # Generate cache states, to avoid recomputing.
        # Default 3, but can be changed by sampler
        self.set_cache_size(3)

    def get_requirements(self):
        """
        Get a dictionary of requirements (e.g. calculated by a another component)
        :return: dictionary of requirements
        """
        return {}

    def needs(self, **requirements):
        """
        Function to be called specifying any output products that are needed and hence
        should be calculated by this component.
        Requirements is a dictionary of requirement names with optional parameters for
        each.
        """
        # set-set states whenever needs change
        self._states.clear()

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Do the actual calculation and store results in state dict

        :param state: dictionary to store results
        :param want_derived: whether to set state['derived'] derived parameters
        :param params_values_dict: parameter values
        :return: None or True if success, False for fail
        """

    def initialize_with_params(self):
        """
        Additional initialization after requirements called and input_params and
        output_params have been assigned (but provider and needs assigned).
        """

    def initialize_with_provider(self, provider):
        """
        Final initialization after parameters, provider and needs assigned.
        The provider is used to get the requirements of this theory using provider.get_X()
        and provider.get_params('Y').
        """
        self.provider = provider

    def get_param(self, p):
        """
        Interface function for likelihoods to get derived parameters.
        """
        return self._current_state["derived"][p]

    def get_can_provide_methods(self):
        """
        Get a dictionary of quantities X that can be retrieved using get_X methods.

        :return: dictionary of the form {X: get_X method}
        """
        return get_class_methods(self.get_provider().__class__, not_base=Theory)

    def get_can_provide_params(self):
        """
        Get a list of derived parameters that this component can calculate.

        :return: list of parameter names
        """
        params = getattr(self, _params, None)
        if params:
            return [k for k, v in params.items() if
                    hasattr(v, 'get') and v.get('derived')]
        else:
            return []

    def get_allow_agnostic(self):
        """
        Whether it is allowed to pass all unassigned input and output parameters to this
        component (True) or whether parameters must be explicitly specified (False).

        :return: True or False
        """
        return False

    def set_cache_size(self, n):
        """
        Set how many states to cache
        """
        self._states = deque(maxlen=n)

    def check_cache_and_compute(self, dependency_params=None, want_derived=False,
                                cached=True, **params_values_dict):
        """
        Takes a dictionary of parameter values and computes the products needed by the
        likelihood, or uses the cached value if that exists for these parameters.
        If want_derived, the derived parameters are saved in the computed state
        (retrieved using get_current_derived()).
        """
        params_values_dict = params_values_dict.copy()
        self.log.debug("Got parameters %r", params_values_dict)

        for set_param in getattr(self, _requires, []):
            # mess handling optional parameters that may be computed elsewhere, eg. YHe
            params_values_dict[set_param] = self.provider.get_param(set_param)
        state = None
        if cached:
            for _state in self._states:
                if _state["params"] == params_values_dict and \
                        _state["dependency_params"] == dependency_params \
                        and (not want_derived or _state["derived"] is not None):
                    state = _state
                    self.log.debug("Re-using computed results")
                    self._states.remove(_state)
                    break
        if not state:
            self.log.debug("Computing new state")
            state = {"params": params_values_dict,
                     "dependency_params": dependency_params,
                     "derived": {} if want_derived else None, "derived_extra": None}
            if self.timer:
                self.timer.start()
            try:
                if self.calculate(state, want_derived, **params_values_dict) is False:
                    return False
            except LoggedError:
                raise
            except Exception as e:
                if self.stop_at_error:
                    raise LoggedError(self.log, "Error at evaluation: %r", e)
                else:
                    raise

            if self.timer:
                self.timer.increment(self.log)

        # make this state the current one
        self._states.appendleft(state)
        self._current_state = state
        return True

    def get_current_derived(self):
        return self._current_state.get("derived", {})

    def get_provider(self):
        """
        Return object containing get_X, get_param methods to get computed results.
        This defaults to self, but can change to delegate provision to another object

        :return: object instance
        """
        return self


class TheoryCollection(ComponentCollection):
    """
    Initializes the list of theory codes.
    """

    def __init__(self, info_theory, path_install=None, timing=None):
        super(TheoryCollection, self).__init__()
        self.set_logger("theory")

        if info_theory:
            for name, info in info_theory.items():
                # If it has an "external" key, wrap it up. Else, load it up
                if isinstance(info, Theory):
                    self[name] = info
                else:
                    if _external in info:
                        theory_class = info[_external]
                        if not inspect.isclass(theory_class) or \
                                not issubclass(theory_class, Theory):
                            raise LoggedError(self.log,
                                              "Theory %s is not a Theory subclass", name)
                    else:
                        theory_class = get_class(name, kind=kinds.theory)
                    self[name] = theory_class(info, path_install=path_install,
                                              timing=timing, name=name)

    def __getattribute__(self, name):
        if not name.startswith('_'):
            try:
                return super(TheoryCollection, self).__getattribute__(name)
            except AttributeError:
                self.log.warn("No attribute %s of TheoryCollection. Use model.provider "
                              "if you want to access computed requests" % name)
                pass
        return object.__getattribute__(self, name)
