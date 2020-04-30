"""
.. module:: likelihood

:Synopsis: Likelihood class and likelihood collection
:Author: Jesus Torrado and Antony Lewis

This module defines the main :class:`Likelihood` class, from which every likelihood
usually inherits, and the :class:`LikelihoodCollection` class, which groups and manages
all the individual likelihoods.

Likelihoods inherit from :class:`~theory.Theory`, adding an additional method
to return the likelihood. As with all theories, likelihoods cache results, and the
function :meth:`LikelihoodInterface.get_current_logp` is used by :class:`model.Model` to
calculate the total likelihood. The default Likelihood implementation does the actual
calculation of the log likelihood in the `logp` function, which is then called
by :meth:`Likelihood.calculate` to save the result into the current state.

Subclasses typically just provide the `logp` function to define their likelihood result,
and use :meth:`~theory.Theory.get_requirements` to specify which inputs are needed from
other theory codes (or likelihoods). Other methods of the :class:`~theory.Theory` base
class can be used as and when needed.

"""

# Global
import sys
import traceback
import inspect
from time import sleep
from typing import Mapping, Optional, Union
from itertools import chain
import numpy as np

# Local
from cobaya.conventions import kinds, _external, _component_path, empty_dict, \
    _input_params, _output_params, _requires
from cobaya.tools import get_class, get_external_function, getfullargspec, str_to_list
from cobaya.log import LoggedError
from cobaya.component import ComponentCollection
from cobaya.theory import Theory


class LikelihoodInterface:
    """
    Interface function for likelihoods. Can descend from a :class:`~theory.Theory` class
    and this to make a likelihood (where the calculate() method stores state['logp'] for
    the current parameters), or likelihoods can directly inherit from :class:`Likelihood`
    instead.

    The get_current_logp function returns the current state's logp, and does not normally
    need to be changed.
    """

    _current_state: Mapping[str, Mapping]

    def get_current_logp(self):
        """
        Gets log likelihood for the current point

        :return:  log likelihood from the current state
        """
        return self._current_state["logp"]


class Likelihood(Theory, LikelihoodInterface):
    """Base class for likelihoods. Extends from :class:`LikelihoodInterface` and the
    general :class:`~theory.Theory` class by adding functions to return likelihoods
    functions (logp function for a given point)."""

    type: Optional[Union[list, str]] = []

    def __init__(self, info=empty_dict, name=None, timing=None, packages_path=None,
                 initialize=True, standalone=True):
        self.delay = 0
        super().__init__(info, name=name, timing=timing,
                         packages_path=packages_path, initialize=initialize,
                         standalone=standalone)
        # Make sure `types` is a list of data types, for aggregated chi2
        self.type = str_to_list(getattr(self, "type", []) or [])

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
        derived = {} if want_derived else None
        state["logp"] = -np.inf  # in case of exception
        state["logp"] = self.logp(_derived=derived, **params_values_dict)
        self.log.debug("Computed log-likelihood = %g", state["logp"])
        if want_derived:
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
        self.external_function = get_external_function(info[_external], name=name)
        self._self_arg = "_self"
        argspec = getfullargspec(self.external_function)
        if info.get(_input_params, []):
            setattr(self, _input_params, str_to_list(info.get(_input_params)))
        else:
            ignore_args = [self._self_arg]
            # MARKED FOR DEPRECATION IN v3.0
            ignore_args += ["_derived", "_theory"]
            # END OF DEPRECATION BLOCK
            setattr(self, _input_params,
                    [p for p in argspec.args if p not in ignore_args])
        # MARKED FOR DEPRECATION IN v3.0
        self._derived_through_arg = "_derived" in argspec.args
        # END OF DEPRECATION BLOCK
        if info.get(_output_params, []):
            setattr(self, _output_params, str_to_list(info.get(_output_params)))
        # MARKED FOR DEPRECATION IN v3.0
        elif self._derived_through_arg:
            self.log.warning(
                "The use of a `_derived` argument to deal with derived parameters will be"
                " deprecated in a future version. From now on please list your derived "
                "parameters in a list as the value of %r in the likelihood info (see "
                "documentation) and have your function return a tuple "
                "`(logp, {derived_param_1: value_1, ...})`.", _output_params)
            # BEHAVIOUR TO BE REPLACED BY ERROR:
            derived_kw_index = argspec.args[-len(argspec.defaults):].index("_derived")
            setattr(self, _output_params, argspec.defaults[derived_kw_index])
        # END OF DEPRECATION BLOCK
        else:
            setattr(self, _output_params, [])
        # Required quantities from other components
        self._uses_self_arg = self._self_arg in argspec.args
        if info.get(_requires) and not self._uses_self_arg:
            raise LoggedError(
                self.log, "If a likelihood has external requirements, declared under %r, "
                          "it needs to accept a keyword argument %r.", _requires,
                self._self_arg)
        # MARKED FOR DEPRECATION IN v3.0
        self._uses_old_theory = "_theory" in argspec.args
        if self._uses_old_theory:
            self.log.warning(
                "The use of a `_theory` argument to deal with requirements will be"
                " deprecated in a future version. From now on please indicate your "
                "requirements as the value of field %r in the likelihood info (see "
                "documentation) and have your function take a parameter `_self`.",
                _requires)
            # BEHAVIOUR TO BE REPLACED BY ERROR:
            info[_requires] = argspec.defaults[
                argspec.args[-len(argspec.defaults):].index("_theory")]
        # END OF DEPRECATION BLOCK
        self._requirements = info.get(_requires, {}) or {}
        self.log.info("Initialized external likelihood.")

    def get_requirements(self):
        return self._requirements

    def logp(self, **params_values):
        # Remove non-input params (except _derived)
        # TODO: this lines should be removed whenever input_params/reqs split is fixed
        for p in list(params_values):
            if p not in self.input_params and p != "_derived":
                params_values.pop(p)
        _derived = params_values.pop("_derived", None)
        if self._uses_self_arg:
            params_values[self._self_arg] = self
        # MARKED FOR DEPRECATION IN v3.0
        # BLOCK TO BE REMOVED
        if self._derived_through_arg:
            params_values["_derived"] = _derived
        if self._uses_old_theory:
            params_values["_theory"] = self.provider
        # END OF DEPRECATION BLOCK
        try:
            return_value = self.external_function(**params_values)
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
            # MARKED FOR DEPRECATION IN v3.0 --> just after the `not` below
            elif self.output_params and not self._derived_through_arg:
                raise LoggedError(self.log, bad_return_msg)
            else:  # no return.__len__ and output_params expected
                logp = return_value
            return logp
        except Exception as ex:
            if isinstance(ex, LoggedError):
                # Assume proper error info was written before raising LoggedError
                pass
            else:
                # Print traceback
                self.log.error("".join(
                    ["-"] * 16 + ["\n\n"] +
                    list(traceback.format_exception(*sys.exc_info())) +
                    ["\n"] + ["-"] * 37))
            raise LoggedError(
                self.log, "The external likelihood '%s' failed at evaluation. "
                          "See error info on top of this message.", self.get_name())


class LikelihoodCollection(ComponentCollection):
    """
    A dictionary storing experimental likelihood :class:`Likelihood` instances index
    by their names.
    """

    def __init__(self, info_likelihood, packages_path=None, timing=None, theory=None):
        super().__init__()
        self.set_logger("likelihood")
        self.theory = theory
        # Get the individual likelihood classes
        for name, info in info_likelihood.items():
            if isinstance(name, Theory):
                name, info = name.get_name(), info
            if isinstance(info, Theory):
                self.add_instance(name, info)
            elif _external in info:
                if isinstance(info[_external], Theory):
                    self.add_instance(name, info[_external])
                elif inspect.isclass(info[_external]):
                    if not hasattr(info[_external], "get_current_logp") or \
                            not issubclass(info[_external], Theory):
                        raise LoggedError(self.log, "%s: external class likelihood must "
                                                    "be a subclass of Theory and have "
                                                    "logp, get_current_logp functions",
                                          info[_external].__name__)
                    self.add_instance(name,
                                      info[_external](info, packages_path=packages_path,
                                                      timing=timing,
                                                      standalone=False,
                                                      name=name))
                else:
                    # If it has an "external" key, wrap it up. Else, load it up
                    self.add_instance(name, LikelihoodExternalFunction(info, name,
                                                                       timing=timing))
            else:
                like_class = get_class(name, kind=kinds.likelihood,
                                       component_path=info.pop(_component_path, None))
                self.add_instance(name, like_class(info, packages_path=packages_path,
                                                   timing=timing, standalone=False,
                                                   name=name))

            if not hasattr(self[name], "get_current_logp"):
                raise LoggedError(self.log, "'Likelihood' %s is not actually a "
                                            "likelihood (no get_current_logp function)",
                                  name)

    @property
    def all_types(self):
        if not hasattr(self, "_all_types"):
            self._all_types = set(chain(
                *[str_to_list(getattr(self[like], "type", []) or []) for like in self]))
        return self._all_types
