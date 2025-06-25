"""
.. module:: likelihood

:Synopsis: Likelihood class and likelihood collection
:Author: Jesus Torrado and Antony Lewis

This module defines the main :class:`Likelihood` class, from which every likelihood
usually inherits, and the :class:`LikelihoodCollection` class, which groups and manages
all the individual likelihoods.

Likelihoods inherit from :class:`~theory.Theory`, adding an additional method
to return the likelihood. As with all theories, likelihoods cache results, and the
property :meth:`LikelihoodInterface.current_logp` is used by :class:`model.Model` to
calculate the total likelihood. The default Likelihood implementation does the actual
calculation of the log likelihood in the `logp` function, which is then called
by :meth:`Likelihood.calculate` to save the result into the current state.

Subclasses typically just provide the `logp` function to define their likelihood result,
and use :meth:`~theory.Theory.get_requirements` to specify which inputs are needed from
other theory codes (or likelihoods). Other methods of the :class:`~theory.Theory` base
class can be used as and when needed.

"""

import numbers
from collections.abc import Mapping
from itertools import chain
from time import sleep

import numpy as np

from cobaya.component import ComponentCollection, get_component_class
from cobaya.log import LoggedError
from cobaya.theory import Theory
from cobaya.tools import get_external_function, getfullargspec, str_to_list
from cobaya.typing import LikeDictIn, LikesDict, ParamValuesDict, empty_dict


class LikelihoodInterface:
    """
    Interface function for likelihoods. Can descend from a :class:`~theory.Theory` class
    and this to make a likelihood (where the calculate() method stores state['logp'] for
    the current parameters), or likelihoods can directly inherit from :class:`Likelihood`
    instead.

    The current_logp property returns the current state's logp as a scalar, and does not
    normally need to be changed.
    """

    current_state: dict

    @property
    def current_logp(self) -> float:
        """
        Gets log likelihood for the current point.

        :return:  log likelihood from the current state as a scalar
        """
        value = self.current_state["logp"]
        if hasattr(value, "__len__"):
            value = value[0]
        return value


def is_LikelihoodInterface(class_instance):
    """
    Checks for `current_logp` property. `hasattr()` cannot safely be used in this case
    because `self._current_state` has not yet been defined.

    Works for both classes and instances.
    """
    # NB: This is much faster than "<method> in dir(class)"
    cls = class_instance if class_instance.__class__ is type else class_instance.__class__
    return isinstance(getattr(cls, "current_logp", None), property)


class Likelihood(Theory, LikelihoodInterface):
    """
    Base class for likelihoods. Extends from :class:`LikelihoodInterface` and the
    general :class:`~theory.Theory` class by adding functions to return likelihoods
    functions (logp function for a given point).
    """

    type: list | str | None = []

    def __init__(
        self,
        info: LikeDictIn = empty_dict,
        name: str | None = None,
        timing: bool | None = None,
        packages_path: str | None = None,
        initialize=True,
        standalone=True,
    ):
        self.delay = 0
        super().__init__(
            info,
            name=name,
            timing=timing,
            packages_path=packages_path,
            initialize=initialize,
            standalone=standalone,
        )

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
        derived: ParamValuesDict | None = {} if want_derived else None
        state["logp"] = -np.inf  # in case of exception
        state["logp"] = self.logp(_derived=derived, **params_values_dict)
        self.log.debug("Computed log-likelihood = %s", state["logp"])
        if derived is not None:
            state["derived"] = derived.copy()

    def wait(self):
        if self.delay:
            self.log.debug("Sleeping for %f seconds.", self.delay)
            sleep(self.delay)


class AbsorbUnusedParamsLikelihood(Likelihood):
    pass


class LikelihoodExternalFunction(Likelihood):
    def __init__(
        self,
        info: LikeDictIn,
        name: str | None = None,
        timing: bool | None = None,
        **kwargs,
    ):
        if kwargs:
            self.log.warning(
                "The following kwargs are ignored for external likelihood functions: %r",
                kwargs,
            )
        super().__init__(
            info,
            name=name,
            timing=timing,
            packages_path=None,
            initialize=True,
            standalone=False,
        )
        self.input_params = str_to_list(self.input_params)
        # Store the external function and assign its arguments
        self.external_function = get_external_function(info["external"], name=name)
        self._self_arg = "_self"
        argspec = getfullargspec(self.external_function)
        # NB: unnamed args are not supported
        has_unnamed_args = bool(argspec.varargs)
        if has_unnamed_args:
            raise LoggedError(
                self.log, "External likelihoods with unnamed args are not supported."
            )
        ignore_args = [self._self_arg]
        if argspec.defaults:
            required_args = set(argspec.args[: -len(argspec.defaults)])
        else:
            required_args = set(argspec.args)
        # Allows for passing a class method
        # (Do not mistake for the use of _self to get quantities from provider, see below)
        if hasattr(self.external_function, "__self__"):
            required_args.remove("self")
        self.params = {p: None for p in required_args if p not in ignore_args}
        if self.output_params:
            self.output_params = str_to_list(self.output_params) or []
        # Required quantities from other components
        self._uses_self_arg = self._self_arg in argspec.args
        if info.get("requires") and not self._uses_self_arg:
            raise LoggedError(
                self.log,
                "If a likelihood has external requirements, declared under %r, "
                "it needs to accept a keyword argument %r.",
                "requires",
                self._self_arg,
            )
        self._requirements = info.get("requires") or {}
        self._optional_args = [
            p
            for p, val in chain(
                zip(argspec.args[-len(argspec.defaults) :], argspec.defaults)
                if argspec.defaults
                else [],
                (argspec.kwonlydefaults or {}).items(),
            )
            if p not in ignore_args and (isinstance(val, numbers.Number) or val is None)
        ]
        self._args = set(chain(self._optional_args, self.params))
        # If has unnamed kwargs, assume these are the ones declared in input_params
        has_unnamed_kwargs = bool(argspec.varkw)
        if has_unnamed_kwargs:
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
        except Exception:
            self.log.debug("External function failed at evaluation.")
            raise
        bad_return_msg = "Expected return value `(logp, {derived_params_dict})`."
        if hasattr(return_value, "__len__"):
            logp = return_value[0]  # type: ignore
            if self.output_params:
                try:
                    if _derived is not None:
                        _derived.update(return_value[1])  # type: ignore
                        params_values["_derived"] = _derived
                except (AttributeError, TypeError, IndexError) as excpt:
                    raise LoggedError(self.log, bad_return_msg) from excpt
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

    def __init__(
        self, info_likelihood: LikesDict, packages_path=None, timing=None, theory=None
    ):
        super().__init__()
        self.set_logger("likelihood")
        self.theory = theory
        # Get the individual likelihood classes
        for name, info in info_likelihood.items():
            if isinstance(name, Theory):
                name = name.get_name()
            if isinstance(info, Theory):
                self.add_instance(name, info)
            elif isinstance(info, Mapping) and "external" in info:
                external = info["external"]
                if isinstance(external, Theory):
                    self.add_instance(name, external)
                elif isinstance(external, type):
                    if not is_LikelihoodInterface(external) or not issubclass(
                        external, Theory
                    ):
                        raise LoggedError(
                            self.log,
                            "%s: external class likelihood must "
                            "be a subclass of Theory and have "
                            "logp, current_logp attributes",
                            external.__name__,
                        )
                    self.add_instance(
                        name,
                        external(
                            info,
                            packages_path=packages_path,
                            timing=timing,
                            standalone=False,
                            name=name,
                        ),
                    )
                else:
                    # If it has an "external" key, wrap it up. Else, load it up
                    self.add_instance(
                        name, LikelihoodExternalFunction(info, name, timing=timing)
                    )
            else:
                assert isinstance(info, Mapping)
                like_class = get_component_class(
                    name,
                    kind="likelihood",
                    component_path=info.get("python_path"),
                    class_name=info.get("class"),
                    logger=self.log,
                )
                assert like_class is not None
                self.add_instance(
                    name,
                    like_class(
                        info,
                        packages_path=packages_path,
                        timing=timing,
                        standalone=False,
                        name=name,
                    ),
                )

            if not is_LikelihoodInterface(self[name]):
                raise LoggedError(
                    self.log,
                    "'Likelihood' %s is not actually a "
                    "likelihood (no current_logp attribute)",
                    name,
                )

    def get_helper_theory_collection(self):
        return self.theory

    @property
    def all_types(self):
        if not hasattr(self, "_all_types"):
            self._all_types = set(chain(*[like.type_list for like in self.values()]))
        return self._all_types
