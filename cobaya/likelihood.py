"""
.. module:: likelihood

:Synopsis: Likelihood class and likelihood collection
:Author: Jesus Torrado and Antony Lewis

This module defines the main :class:`Likelihood` class, from which every likelihood
usually inherits, and the :class:`LikelihoodCollection` class, which groups and manages
all the individual likelihoods.

Likelihoods inherit from :class:`~theory.Theory`, adding an additional method
to return the likelihood. As with all theories, likelihoods cache results, and the
function :meth:`LikelihoodInterface.current_logp` is used by :class:`model.Model` to
calculate the total likelihood. The default Likelihood implementation does the actual
calculation of the log likelihood in the `logp` function, which is then called
by :meth:`Likelihood.calculate` to save the result into the current state.

Subclasses typically just provide the `logp` function to define their likelihood result,
and use :meth:`~theory.Theory.get_requirements` to specify which inputs are needed from
other theory codes (or likelihoods). Other methods of the :class:`~theory.Theory` base
class can be used as and when needed.

"""

# Global
from time import sleep
from typing import Mapping, Optional, Union, Dict
from itertools import chain
import numpy as np
import numbers

# Local
from cobaya.typing import LikesDict, LikeDictIn, ParamValuesDict, empty_dict
from cobaya.tools import get_external_function, getfullargspec, str_to_list
from cobaya.log import LoggedError
from cobaya.component import ComponentCollection, get_component_class
from cobaya.theory import Theory


class LikelihoodInterface:
    """
    Interface function for likelihoods. Can descend from a :class:`~theory.Theory` class
    and this to make a likelihood (where the calculate() method stores state['logp'] for
    the current parameters), or likelihoods can directly inherit from :class:`Likelihood`
    instead.

    The current_logp property returns the current state's logp, and does not normally
    need to be changed.
    """

    current_state: Dict

    @property
    def current_logp(self) -> float:
        """
        Gets log likelihood for the current point

        :return:  log likelihood from the current state
        """
        return self.current_state["logp"]


def is_LikelihoodInterface(class_instance):
    """
    Checks for `current_logp` property. `hasattr()` cannot safely be used in this case
    because `self._current_state` has not yet been defined.

    Works for both classes and instances.
    """
    # NB: This is much faster than "<method> in dir(class)"
    cls = class_instance if class_instance.__class__ is type \
        else class_instance.__class__
    return isinstance(getattr(cls, "current_logp", None), property)


class Likelihood(Theory, LikelihoodInterface):
    """Base class for likelihoods. Extends from :class:`LikelihoodInterface` and the
    general :class:`~theory.Theory` class by adding functions to return likelihoods
    functions (logp function for a given point)."""

    type: Optional[Union[list, str]] = []

    def __init__(self, info: LikeDictIn = empty_dict,
                 name: Optional[str] = None,
                 timing: Optional[bool] = None,
                 packages_path: Optional[str] = None,
                 initialize=True, standalone=True):
        self.delay = 0
        super().__init__(info, name=name, timing=timing,
                         packages_path=packages_path, initialize=initialize,
                         standalone=standalone)

    @property
    def theory(self):
        # for backwards compatibility
        return self.provider

    def logp(self, **params_values):
        """
        Computes and returns the log likelihood value.
        Takes as keyword arguments the parameter values.
        To get the derived parameters, pass a `_derived` keyword with an empty dictionary.

        Alternatively you can just implement calculate() and save the log likelihood into
        state['logp']; this may be more convenient if you also need to also calculate
        other quantities.
        """
        return None

    def marginal(self, directions=None, params_values=None):
        """
        (For analytic likelihoods only.)
        Computes the marginal likelihood.
        If nothing is specified, returns the total marginal likelihood.
        If some directions are specified (as a list, tuple or array), returns the marginal
        likelihood pdf over those directions evaluated at the given parameter values.
        """
        raise LoggedError(self.log, "Exact marginal likelihood not defined.")

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Calculates the likelihood and any derived parameters or needs.
        Return False is the calculation fails.

        """
        derived: Optional[ParamValuesDict] = {} if want_derived else None
        state["logp"] = -np.inf  # in case of exception
        state["logp"] = self.logp(_derived=derived, **params_values_dict)
        self.log.debug("Computed log-likelihood = %g", state["logp"])
        if derived is not None:
            state["derived"] = derived.copy()

    def wait(self):
        if self.delay:
            self.log.debug("Sleeping for %f seconds.", self.delay)
            sleep(self.delay)


class AbsorbUnusedParamsLikelihood(Likelihood):
    pass


class LikelihoodExternalFunction(Likelihood):
    def __init__(self, info, name, timing=None):
        Theory.__init__(self, info, name=name, timing=timing, standalone=False)
        # Store the external function and assign its arguments
        self.external_function = get_external_function(info["external"], name=name)
        self._self_arg = "_self"
        argspec = getfullargspec(self.external_function)
        self.input_params = str_to_list(self.input_params)
        ignore_args = [self._self_arg]
        if argspec.defaults:
            required_args = argspec.args[:-len(argspec.defaults)]
        else:
            required_args = argspec.args
        self.params = {p: None for p in required_args if p not in ignore_args}
        # MARKED FOR DEPRECATION IN v3.0
        if "_derived" in argspec.args:
            raise LoggedError(
                self.log, "The use of a `_derived` argument to deal with derived "
                          "parameters has been deprecated. From now on please list your "
                          "derived parameters in a list as the value of %r in the "
                          "likelihood info (see documentation) and have your function "
                          "return a tuple `(logp, {derived_param_1: value_1, ...})`.",
                "output_params")
        # END OF DEPRECATION BLOCK
        if self.output_params:
            self.output_params = str_to_list(self.output_params) or []
        # Required quantities from other components
        self._uses_self_arg = self._self_arg in argspec.args
        if info.get("requires") and not self._uses_self_arg:
            raise LoggedError(
                self.log, "If a likelihood has external requirements, declared under %r, "
                          "it needs to accept a keyword argument %r.", "requires",
                self._self_arg)
        self._requirements = info.get("requires") or {}
        # MARKED FOR DEPRECATION IN v3.0
        if "_theory" in argspec.args:
            raise LoggedError(
                self.log, "The use of a `_theory` argument to deal with requirements has "
                          "been deprecated. From now on please indicate your requirements"
                          " as the value of field %r in the likelihood info (see "
                          "documentation) and have your function take a parameter "
                          "`_self`.", "requires")
        # END OF DEPRECATION BLOCK

        self._optional_args = \
            [p for p, val in chain(zip(argspec.args[-len(argspec.defaults):],
                                       argspec.defaults) if argspec.defaults else [],
                                   (argspec.kwonlydefaults or {}).items())
             if p not in ignore_args and
             (isinstance(val, numbers.Number) or val is None)]
        self._args = set(chain(self._optional_args, self.params))
        if argspec.varkw:
            self._args.update(self.input_params)
        self.log.info("Initialized external likelihood.")

    def get_requirements(self):
        return self._requirements

    def get_can_support_params(self):
        return self._optional_args

    def logp(self, **params_values):
        # Remove non-input params (except _derived)
        _derived = params_values.pop("_derived", None)
        for p in list(params_values):
            if p not in self._args:
                params_values.pop(p)
        if self._uses_self_arg:
            params_values[self._self_arg] = self
        try:
            return_value = self.external_function(**params_values)
        except:
            self.log.debug("External function failed at evaluation.")
            raise
        bad_return_msg = "Expected return value `(logp, {derived_params_dict})`."
        if hasattr(return_value, "__len__"):
            logp = return_value[0]
            if self.output_params:
                try:
                    if _derived is not None:
                        _derived.update(return_value[1])
                        params_values["_derived"] = _derived
                except:
                    raise LoggedError(self.log, bad_return_msg)
        elif self.output_params:
            raise LoggedError(self.log, bad_return_msg)
        else:  # no return.__len__ and output_params expected
            logp = return_value
        return logp


class LikelihoodCollection(ComponentCollection):
    """
    A dictionary storing experimental likelihood :class:`Likelihood` instances index
    by their names.
    """

    def __init__(self, info_likelihood: LikesDict, packages_path=None, timing=None,
                 theory=None):
        super().__init__()
        self.set_logger("likelihood")
        self.theory = theory
        # Get the individual likelihood classes
        for name, info in info_likelihood.items():
            if isinstance(name, Theory):
                name, info = name.get_name(), info
            if isinstance(info, Theory):
                self.add_instance(name, info)
            elif isinstance(info, Mapping) and "external" in info:
                external = info["external"]
                if isinstance(external, Theory):
                    self.add_instance(name, external)
                elif isinstance(external, type):
                    if not is_LikelihoodInterface(external) or \
                            not issubclass(external, Theory):
                        raise LoggedError(self.log, "%s: external class likelihood must "
                                                    "be a subclass of Theory and have "
                                                    "logp, current_logp attributes",
                                          external.__name__)
                    self.add_instance(name, external(info, packages_path=packages_path,
                                                     timing=timing, standalone=False,
                                                     name=name))
                else:
                    # If it has an "external" key, wrap it up. Else, load it up
                    self.add_instance(name, LikelihoodExternalFunction(info, name,
                                                                       timing=timing))
            else:
                assert isinstance(info, Mapping)
                like_class: type = get_component_class(
                    name, kind="likelihood",
                    component_path=info.get("python_path", None),
                    class_name=info.get("class"), logger=self.log)
                self.add_instance(name, like_class(info, packages_path=packages_path,
                                                   timing=timing, standalone=False,
                                                   name=name))

            if not is_LikelihoodInterface(self[name]):
                raise LoggedError(self.log, "'Likelihood' %s is not actually a "
                                            "likelihood (no current_logp attribute)",
                                  name)

    def get_helper_theory_collection(self):
        return self.theory

    @property
    def all_types(self):
        if not hasattr(self, "_all_types"):
            self._all_types = set(chain(*[like.type_list for like in self.values()]))
        return self._all_types
