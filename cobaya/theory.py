"""
.. module:: theory

:Synopsis: :class:`Theory` is a base class for theory codes and likelihoods.

Both likelihoods and theories calculate something. Likelihoods are distinguished
because they calculate a log likelihood. Both theory codes and likelihoods can calculate
other things, and they may have complex dependencies between them: e.g. the likelihood
depends on observable A that is computed by theory code B than in turn requires
calculation of input calculation by code C.

This module contains the base class for all of these calculation components. It handles
caching of results, so that calculations do not need to be redone when the parameters on
which a component directly (or indirectly) depends have not changed.

Subclasses generally provide the :meth:`Theory.get_requirements`,
:meth:`Theory.calculate` and initialization methods as required. The
:meth:`Theory.must_provide` method is used to tell a code which requirements are
actually needed by other components, and may return a dictionary of additional conditional
requirements based on those passed.

The :meth:`Theory.calculate` method saves all needed results in the state dictionary
(which is cached and reused as needed). Subclasses define ``get_X`` or ``get_result(X)``
methods to return the actual result of the calculation for X for the current cache state.
The :meth:`Theory.get_param` method returns the value of a derived parameter for the
current state.

For details and examples of how to handle multiple theory codes with complex dependencies
see :doc:`theories_and_dependencies`.

"""

from collections import deque
from typing import Sequence, Optional, Union, Tuple, Dict, Iterable, Set, Any, List
# Local
from cobaya.typing import TheoryDictIn, TheoriesDict, InfoDict, ParamValuesDict, \
    ParamsDict, empty_dict, unset_params
from cobaya.component import CobayaComponent, ComponentCollection, get_component_class
from cobaya.tools import str_to_list
from cobaya.log import LoggedError, always_stop_exceptions
from cobaya.tools import get_class_methods


class Theory(CobayaComponent):
    """Base class theory that can calculate something."""

    speed: float = -1
    stop_at_error: bool = False
    version: Optional[Union[dict, str]] = None
    params: ParamsDict

    # special components set by the dependency resolver;
    # (for Theory included in updated yaml but not in defaults)
    input_params: Sequence[str] = unset_params
    output_params: Sequence[str] = unset_params

    _states: deque

    def __init__(self, info: TheoryDictIn = empty_dict,
                 name: Optional[str] = None, timing: Optional[bool] = None,
                 packages_path: Optional[str] = None,
                 initialize=True, standalone=True):

        self._measured_speed = None
        super().__init__(info, name=name, timing=timing,
                         packages_path=packages_path, initialize=initialize,
                         standalone=standalone)

        # set to Provider instance before calculations
        self.provider: Any = None
        # Generate cache states, to avoid recomputing.
        # Default 3, but can be changed by sampler
        self.set_cache_size(3)
        self._helpers: Dict[str, 'Theory'] = {}
        self._input_params_extra: Set[str] = set()

    def get_requirements(self) -> Union[InfoDict, Sequence[str],
                                        Sequence[Tuple[str, InfoDict]]]:
        """
        Get a dictionary of requirements (or a list of requirement name, option tuples)
        that are always needed (e.g. must be calculated by another component
        or provided as input parameters).

        :return: dictionary or list of tuples of requirement names and options
                (or iterable of requirement names if no optional parameters are needed)
        """
        return str_to_list(getattr(self, "requires", []))

    def must_provide(self, **requirements) -> Union[None, InfoDict, Sequence[str],
                                                    Sequence[Tuple[str, InfoDict]]]:
        """
        Function to be called specifying any output products that are needed and hence
        should be calculated by this component depending..

        The ``requirements`` argument is a requirement name with any optional parameters.
        This function may be called more than once with different requirements.

        :return: optional dictionary (or list of requirement name, option tuples) of
                 conditional requirements for the ones requested.
        """
        # reset states whenever requirements change
        self._states.clear()
        # MARKED FOR DEPRECATION IN v3.0
        # This code will only run if needs() is defined but not must_provide()
        if hasattr(self, "needs"):
            raise LoggedError(
                self.log,
                "The .needs() method has been deprecated in favour of .must_provide()")
        # END OF DEPRECATION BLOCK
        return None

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
        output_params have been assigned (but provider and assigned requirements not yet
        set).
        """

    def initialize_with_provider(self, provider: 'Provider'):
        """
        Final initialization after parameters, provider and assigned requirements set.
        The provider is used to get the requirements of this theory using provider.get_X()
        and provider.get_param('Y').

        :param provider: the :class:`theory.Provider` instance that should be used by
                         this component to get computed requirements
        """
        self.provider = provider

    def get_param(self, p: str) -> float:
        """
        Interface function for likelihoods and other theory components to get derived
        parameters.
        """
        return self.current_state["derived"][p]

    def get_result(self, result_name, **kwargs):
        """
        Interface function for likelihood and other theory components to get
        quantities calculated by this component. By default assumes the quantity
        is just directly saved into the current state (i.e. returns
        ``state[result_name]``).

        :param result_name: name of quantity
        :param kwargs: options specific to X or this component
        :return: result
        """
        return self.current_state[result_name]

    def get_can_provide_methods(self):
        """
        Get a dictionary of quantities X that can be retrieved using get_X methods.

        :return: dictionary of the form {X: get_X method}
        """
        return get_class_methods(self.get_provider().__class__, not_base=Theory)

    def get_can_provide(self) -> Iterable[str]:
        """
        Get a list of names of quantities that can be retrieved using the general
        get_result(X) method.

        :return: iterable of quantity names
        """
        return []

    def get_can_provide_params(self) -> Iterable[str]:
        """
        Get a list of derived parameters that this component can calculate.
        The default implementation returns the result based on the params attribute set
        via the .yaml file or class params (with derived:True for derived parameters).

        :return: iterable of parameter names
        """
        params = getattr(self, "params", None)
        if params:
            return [k for k, v in params.items() if
                    hasattr(v, 'get') and v.get('derived') is True]
        else:
            return []

    def get_can_support_params(self) -> Iterable[str]:
        """
        Get a list of parameters supported by this component, can be used to support
        parameters that don't explicitly appear in the .yaml or class params attribute
        or are otherwise explicitly supported (e.g. via requirements)

        :return: iterable of names of parameters
        """
        return []

    def get_allow_agnostic(self):
        """
        Whether it is allowed to pass all unassigned input parameters to this
        component (True) or whether parameters must be explicitly specified (False).

        :return: True or False
        """
        return False

    @property
    def input_params_extra(self):
        """
        Parameters required from other components, to be passed as input parameters.
        """
        return self._input_params_extra

    def set_cache_size(self, n):
        """
        Set how many states to cache
        """
        self._states = deque(maxlen=n)

    def check_cache_and_compute(self, params_values_dict,
                                dependency_params=None, want_derived=False, cached=True):
        """
        Takes a dictionary of parameter values and computes the products needed by the
        likelihood, or uses the cached value if that exists for these parameters.
        If want_derived, the derived parameters are saved in the computed state
        (retrieved using current_derived).

        params_values_dict can be safely modified and stored.
        """

        if self._input_params_extra:
            params_values_dict.update(
                zip(self._input_params_extra,
                    self.provider.get_param(self._input_params_extra)))
        self.log.debug("Got parameters %r", params_values_dict)
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
                     "derived": {} if want_derived else None}
            if self.timer:
                self.timer.start()
            try:
                if self.calculate(state, want_derived, **params_values_dict) is False:
                    return False
            except always_stop_exceptions:
                raise
            except Exception as excpt:
                if self.stop_at_error:
                    self.log.error("Error at evaluation. See error information below.")
                    raise
                else:
                    self.log.debug(
                        "Ignored error at evaluation and assigned 0 likelihood "
                        "(set 'stop_at_error: True' as an option for this component "
                        "to stop here and print a traceback). Error message: %r", excpt)
                    return False
            if self.timer:
                self.timer.increment(self.log)
        # make this state the current one
        self._states.appendleft(state)
        self._current_state = state
        return True

    @property
    def current_state(self) -> Dict:
        try:
            return self._current_state
        except AttributeError:
            raise LoggedError(self.log, "Cannot retrieve calculated quantities: "
                                        "nothing has been computed yet "
                                        "(maybe the prior was -infinity?)")

    @property
    def current_derived(self) -> ParamValuesDict:
        return self.current_state.get("derived", {})

    @property
    def type_list(self) -> List[str]:
        # list of labels that classify this component
        # not usually used for Theory, can be used for aggregated chi2 in likelihoods
        return str_to_list(getattr(self, "type", []) or [])

    # MARKED FOR DEPRECATION IN v3.1
    def get_current_derived(self):
        self.log.warning("'Theory.get_current_derived()' method will soon be deprecated "
                         "in favour of 'Theory.current_derived' attribute. Please, "
                         "rename your call.")
        # BEHAVIOUR TO BE REPLACED BY AN ERROR
        return self.current_derived

    # END OF DEPRECATION BLOCK

    def get_provider(self):
        """
        Return object containing get_X, get_param, get_result methods to get computed
        results.
        This defaults to self, but can change to delegate provision to another object

        :return: object instance
        """
        return self

    def get_helper_theories(self) -> Dict[str, 'Theory']:
        """
        Return dictionary of optional names and helper Theory instances that should be
        used in conjunction with this component. The helpers can be created here
        as only called once, and before any other use of helpers.

        :return: dictionary of names and Theory instances
        """
        return {}

    def update_for_helper_theories(self, helpers: Dict[str, 'Theory']):
        self._helpers = helpers
        if helpers:
            components: List[Theory] = list(helpers.values()) + [self]
            for output, attr in enumerate(("input_params", "output_params")):
                pars = getattr(self, attr, unset_params)
                if pars is not unset_params:
                    for component in components:
                        if not component.get_allow_agnostic():
                            if output:
                                supported = component.get_can_provide_params()
                            else:
                                supported = component.get_can_support_params()
                            setattr(component, attr, [p for p in pars if p in supported])
                            pars = [p for p in pars if p not in supported]
                    for component in components:
                        if component.get_allow_agnostic():
                            setattr(component, attr, pars)

    def get_attr_list_with_helpers(self, attr):
        """
        Get combined list of self.attr and helper.attr for all helper theories

        :param attr: attr name
        :return: combined list
        """
        values = list(getattr(self, attr))
        for helper in self._helpers.values():
            values.extend(getattr(helper, attr))
        return values

    def get_speed(self):
        return self._measured_speed or self.speed

    def set_measured_speed(self, speed):
        self.speed = speed


class TheoryCollection(ComponentCollection):
    """
    Initializes the list of theory codes.
    """

    def __init__(self, info_theory: TheoriesDict, packages_path=None, timing=None):
        super().__init__()
        self.set_logger("theory")

        if info_theory:
            for name, info in info_theory.items():
                info = info or {}
                # If it has an "external" key, wrap it up. Else, load it up
                if isinstance(info, Theory):
                    self.add_instance(name, info)
                elif isinstance(info.get("external"), Theory):
                    self.add_instance(name, info["external"])
                else:
                    if "external" in info:
                        theory_class = info["external"]
                        if not isinstance(theory_class, type) or \
                                not issubclass(theory_class, Theory):
                            raise LoggedError(self.log,
                                              "Theory %s is not a Theory subclass", name)
                    else:
                        theory_class = get_component_class(
                            name, kind="theory", class_name=info.get("class"),
                            logger=self.log)
                    self.add_instance(
                        name, theory_class(
                            info, packages_path=packages_path, timing=timing, name=name))

    def __getattribute__(self, name):
        if not name.startswith('_'):
            try:
                return super().__getattribute__(name)
            except AttributeError:
                self.log.warn("No attribute %s of TheoryCollection. Use model.provider "
                              "if you want to access computed requests" % name)
                pass
        return object.__getattribute__(self, name)


class HelperTheory(Theory):

    def has_version(self):
        # assume the main component handles all version checking
        return False


class Provider:
    """
    Class used to retrieve computed requirements.
    Just passes on get_X and get_param methods to the component assigned to compute them.

    For get_param it will also take values directly from the current sampling parameters
    if the parameter is defined there.
    """

    params: ParamValuesDict

    def __init__(self, model, requirement_providers: Dict[str, Theory]):
        self.model = model
        self.requirement_providers = requirement_providers
        self.params = {}

    def set_current_input_params(self, params):
        self.params = params

    def get_param(self, param: Union[str, Iterable[str]]) -> Union[float, List[float]]:
        """
        Returns the value of a derived (or sampled) parameter. If it is not a sampled
        parameter it calls :meth:`Theory.get_param` on component assigned to compute
        this derived parameter.

        :param param: parameter name, or a list of parameter names
        :return: value of parameter, or list of parameter values
        """
        if not isinstance(param, str):
            return [(self.params[p] if p in self.params else
                     self.requirement_providers[p].get_param(p)) for p in param]
        if param in self.params:
            return self.params[param]
        else:
            return self.requirement_providers[param].get_param(param)

    def get_result(self, result_name: str, **kwargs) -> Any:
        return self.requirement_providers[result_name].get_result(result_name, **kwargs)

    def __getattr__(self, name):
        if name.startswith('get_'):
            requirement = name[4:]
            try:
                return getattr(self.requirement_providers[requirement], name)
            except KeyError:  # requirement not listed (parameter or result)
                raise AttributeError
        return object.__getattribute__(self, name)
