"""
.. module:: model

:Synopsis: Wrapper for models: parameterization+prior+likelihood+theory
:Author: Jesus Torrado and Antony Lewis

"""
# Global
import logging
from contextlib import contextmanager
from copy import deepcopy
from itertools import chain
from typing import NamedTuple, Sequence, Mapping, Iterable, Optional, \
    Union, List, Any, Dict, Set
import numpy as np
import os

# Local
from cobaya.conventions import overhead_time, debug_default, get_chi2_name
from cobaya.typing import InfoDict, InputDict, LikesDict, TheoriesDict, \
    ParamsDict, PriorsDict, ParamValuesDict, empty_dict, unset_params
from cobaya.input import update_info, load_input_dict
from cobaya.parameterization import Parameterization
from cobaya.prior import Prior
from cobaya.likelihood import LikelihoodCollection, AbsorbUnusedParamsLikelihood, \
    is_LikelihoodInterface
from cobaya.theory import TheoryCollection, Theory, Provider
from cobaya.log import LoggedError, logger_setup, HasLogger
from cobaya.yaml import yaml_dump
from cobaya.tools import deepcopy_where_possible, are_different_params_lists, \
    str_to_list, sort_parameter_blocks, recursive_update, sort_cosmetic
from cobaya import mpi


@contextmanager
def timing_on(model: 'Model'):
    was_on = model.timing
    if not was_on:
        model.set_timing_on(True)
    try:
        yield
    finally:
        if not was_on:
            model.set_timing_on(False)


# Log-posterior namedtuple
class LogPosterior(NamedTuple):
    logpost: float
    logpriors: Sequence[float]
    loglikes: Sequence[float]
    derived: Sequence[float]


class Requirement(NamedTuple):
    name: str
    options: Optional[InfoDict]

    def __eq__(self, other):
        return self.name == other.name and _dict_equal(self.options, other.options)

    def __repr__(self):
        return "{%r:%r}" % (self.name, self.options)


def as_requirement_list(requirements):
    if isinstance(requirements, Mapping):
        return [Requirement(name, options) for name, options in requirements.items()]
    elif isinstance(requirements, str):
        return [Requirement(requirements, None)]
    elif isinstance(requirements, Iterable):
        if all(isinstance(term, str) for term in requirements):
            return [Requirement(name, None) for name in requirements]
        result = []
        for item in requirements:
            if isinstance(item, Sequence) and len(item) == 2:
                result.append(Requirement(item[0], item[1]))
            else:
                break
        else:
            return result

    raise ValueError('Requirements must be a dict of names and options, a list of names, '
                     'or an iterable of requirement (name, option) pairs')


def _dict_equal(d1, d2):
    # dict/None equality test accounting for numpy arrays not supporting standard eq
    if type(d1) != type(d2):
        return False
    if isinstance(d1, np.ndarray):
        return np.array_equal(d1, d2)
    if not d1 and not d2:
        return True
    if bool(d1) != bool(d2):
        return False
    if isinstance(d1, str):
        return d1 == d2
    if isinstance(d1, Mapping):
        if set(d1) != set(d2):
            return False
        for k, v in d1.items():
            if not _dict_equal(v, d2[k]):
                return False
        return True
    if hasattr(d1, '__len__'):
        if len(d1) != len(d2):
            return False
        for k1, k2 in zip(d1, d2):
            if not _dict_equal(k1, k2):
                return False
        return True
    return d1 == d2


class Model(HasLogger):
    """
    Class containing all the information necessary to compute the unnormalized posterior.

    Allows for low-level interaction with the theory code, prior and likelihood.

    **NB:** do not initialize this class directly; use :func:`~model.get_model` instead,
    with some info as input.
    """

    provider: Any

    def __init__(self, info_params: ParamsDict, info_likelihood: LikesDict,
                 info_prior: Optional[PriorsDict] = None,
                 info_theory: Optional[TheoriesDict] = None,
                 packages_path=None, timing=None, allow_renames=True, stop_at_error=False,
                 post=False, skip_unused_theories=False,
                 dropped_theory_params: Optional[Iterable[str]] = None):
        self.set_logger()
        self._updated_info: InputDict = {
            "params": deepcopy_where_possible(info_params),
            "likelihood": deepcopy_where_possible(info_likelihood)}
        if not self._updated_info["likelihood"]:
            raise LoggedError(self.log, "No likelihood requested!")
        for k, v in (("prior", info_prior), ("theory", info_theory),
                     ("packages_path", packages_path), ("timing", timing)):
            if v not in (None, {}):
                self._updated_info[k] = deepcopy_where_possible(v)  # type: ignore
        self.parameterization = Parameterization(self._updated_info["params"],
                                                 allow_renames=allow_renames,
                                                 ignore_unused_sampled=post)
        self.prior = Prior(self.parameterization, self._updated_info.get("prior"))
        self.timing = timing
        info_theory = self._updated_info.get("theory")
        self.theory = TheoryCollection(info_theory or {}, packages_path=packages_path,
                                       timing=timing)
        info_likelihood = self._updated_info["likelihood"]
        self.likelihood = LikelihoodCollection(info_likelihood, theory=self.theory,
                                               packages_path=packages_path, timing=timing)
        if stop_at_error:
            for component in self.components:
                component.stop_at_error = stop_at_error
        # Assign input/output parameters
        self._assign_params(info_likelihood, info_theory, dropped_theory_params)
        self._set_dependencies_and_providers(skip_unused_theories=skip_unused_theories)
        # Add to the updated info some values that are only available after initialisation
        self._updated_info = recursive_update(
            self._updated_info, self.get_versions(add_version_field=True))
        # Overhead per likelihood evaluation, approximately ind from # input params
        # Evaluation of non-uniform priors will add some overhead per parameter.
        self.overhead = overhead_time

    def info(self) -> InputDict:
        """
        Returns a copy of the information used to create the model, including defaults
        and some new values that are only available after initialisation.
        """
        return deepcopy_where_possible(self._updated_info)

    def _to_sampled_array(self, params_values) -> np.ndarray:
        """
        Internal method to interact with the prior.
        Needs correct (not renamed) parameter names.
        """
        if hasattr(params_values, "keys"):
            params_values_array = np.array(list(params_values.values()))
        else:
            params_values_array = np.atleast_1d(params_values)
            if params_values_array.shape[0] != self.prior.d():
                raise LoggedError(
                    self.log, "Wrong dimensionality: it's %d and it should be %d.",
                    len(params_values_array), self.prior.d())
        if len(params_values_array.shape) >= 2:
            raise LoggedError(
                self.log, "Cannot take arrays of points as inputs, just single points.")
        return params_values_array

    def logpriors(self, params_values, make_finite=False) -> np.ndarray:
        """
        Takes an array or dictionary of sampled parameter values.
        If the argument is an array, parameters must have the same order as in the input.
        When in doubt, you can get the correct order as
        ``list([your_model].parameterization.sampled_params())``.

        Returns the log-values of the priors, in the same order as it is returned by
        ``list([your_model].prior)``. The first one, named ``0``, corresponds to the
        product of the 1-dimensional priors specified in the ``params`` block, and it's
        normalized (in general, the external prior densities aren't).

        If ``make_finite=True``, it will try to represent infinities as the largest real
        numbers allowed by machine precision.
        """
        if hasattr(params_values, "keys"):
            params_values = self.parameterization.check_sampled(**params_values)
        params_values_array = self._to_sampled_array(params_values)
        logpriors = np.asarray(self.prior.logps(params_values_array))
        if make_finite:
            return np.nan_to_num(logpriors)
        return logpriors

    def logprior(self, params_values, make_finite=False):
        """
        Takes an array or dictionary of sampled parameter values.
        If the argument is an array, parameters must have the same order as in the input.
        When in doubt, you can get the correct order as
        ``list([your_model].parameterization.sampled_params())``.

        Returns the log-value of the prior (in general, unnormalized, unless the only
        priors specified are the 1-dimensional ones in the ``params`` block).

        If ``make_finite=True``, it will try to represent infinities as the largest real
        numbers allowed by machine precision.
        """
        logprior = np.sum(self.logpriors(params_values))
        if make_finite:
            return np.nan_to_num(logprior)
        return logprior

    def logps(self, input_params, return_derived=True, cached=True, make_finite=False):
        # Calculate required results and returns likelihoods
        derived_dict: ParamValuesDict = {}
        compute_success = True
        self.provider.set_current_input_params(input_params)
        self.log.debug("Got input parameters: %r", input_params)
        loglikes = np.zeros(len(self.likelihood))
        need_derived = self.requires_derived or return_derived
        for (component, like_index), param_dep in zip(self._component_order.items(),
                                                      self._params_of_dependencies):
            depend_list = [input_params[p] for p in param_dep]
            params = {p: input_params[p] for p in component.input_params}
            compute_success = component.check_cache_and_compute(
                params, want_derived=need_derived,
                dependency_params=depend_list, cached=cached)
            if not compute_success:
                loglikes[:] = -np.inf
                self.log.debug("Calculation failed, skipping rest of calculations ")
                break
            if return_derived:
                derived_dict.update(component.current_derived)
            # Add chi2's to derived parameters
            if like_index is not None:
                try:
                    loglikes[like_index] = float(component.current_logp)  # type: ignore
                except TypeError:
                    raise LoggedError(
                        self.log,
                        "Likelihood %s has not returned a valid log-likelihood, "
                        "but %r instead.", component,
                        component.current_logp)  # type: ignore
        if make_finite:
            loglikes = np.nan_to_num(loglikes)
        if return_derived:
            # Turn the derived params dict into a list and return
            if not compute_success:
                derived_list = [np.nan] * len(self.output_params)
            else:
                for chi2_name, indices in self._chi2_names:
                    derived_dict[chi2_name] = -2 * sum(loglikes[i] for i in indices)
                derived_list = [derived_dict[p] for p in self.output_params]
            return loglikes, derived_list
        return loglikes

    def loglikes(self, params_values=None, return_derived=True, make_finite=False,
                 cached=True):
        """
        Takes an array or dictionary of sampled parameter values.
        If the argument is an array, parameters must have the same order as in the input.
        When in doubt, you can get the correct order as
        ``list([your_model].parameterization.sampled_params())``.

        Returns a tuple ``(loglikes, derived_params)``, where ``loglikes`` are the
        log-values of the likelihoods (unnormalized, in general) in the same order as
        it is returned by ``list([your_model].likelihood)``, and ``derived_params``
        are the values of the derived parameters in the order given by
        ``list([your_model].parameterization.derived_params())``.

        To return just the list of log-likelihood values, make ``return_derived=False``.

        If ``make_finite=True``, it will try to represent infinities as the largest real
        numbers allowed by machine precision.

        If ``cached=False`` (default: True), it ignores previously computed results that
        could be reused.
        """
        if params_values is None:
            params_values = []
        elif hasattr(params_values, "keys"):
            params_values = self.parameterization.check_sampled(**params_values)

        input_params = self.parameterization.to_input(params_values)
        return self._loglikes_input_params(input_params, return_derived=return_derived,
                                           cached=cached, make_finite=make_finite)

    def _loglikes_input_params(self, input_params, return_derived=True, make_finite=False,
                               cached=True):
        result = self.logps(input_params, return_derived=return_derived, cached=cached,
                            make_finite=make_finite)
        if return_derived:
            loglikes, derived_list = result
            derived_sampler = self.parameterization.to_derived(derived_list)
            if self.log.getEffectiveLevel() <= logging.DEBUG:
                self.log.debug(
                    "Computed derived parameters: %s", derived_sampler)
            return loglikes, list(derived_sampler.values())
        return result

    def loglike(self, params_values=None, return_derived=True, make_finite=False,
                cached=True):
        """
        Takes an array or dictionary of sampled parameter values.
        If the argument is an array, parameters must have the same order as in the input.
        When in doubt, you can get the correct order as
        ``list([your_model].parameterization.sampled_params())``.

        Returns a tuple ``(loglike, derived_params)``, where ``loglike`` is the log-value
        of the likelihood (unnormalized, in general), and ``derived_params``
        are the values of the derived parameters in the order given by
        ``list([your_model].parameterization.derived_params())``.
        If the model contains multiple likelihoods, the sum of the loglikes is returned.

        To return just the list of log-likelihood values, make ``return_derived=False``.

        If ``make_finite=True``, it will try to represent infinities as the largest real
        numbers allowed by machine precision.

        If ``cached=False`` (default: True), it ignores previously computed results that
        could be reused.
        """
        ret_value = self.loglikes(params_values, return_derived=return_derived,
                                  cached=cached)
        if return_derived:
            loglike = np.sum(ret_value[0])
            if make_finite:
                return np.nan_to_num(loglike), ret_value[1]
            return loglike, ret_value[1]
        else:
            loglike = np.sum(ret_value)
            if make_finite:
                return np.nan_to_num(loglike)
            return loglike

    def logposterior(self, params_values, return_derived=True,
                     make_finite=False, cached=True, _no_check=False) -> LogPosterior:
        """
        Takes an array or dictionary of sampled parameter values.
        If the argument is an array, parameters must have the same order as in the input.
        When in doubt, you can get the correct order as
        ``list([your_model].parameterization.sampled_params())``.

        Returns the a ``logposterior`` ``NamedTuple``, with the following fields:

        - ``logpost``: log-value of the posterior.
        - ``logpriors``: log-values of the priors, in the same order as in
          ``list([your_model].prior)``. The first one, corresponds to the
          product of the 1-dimensional priors specified in the ``params``
          block. Except for the first one, the priors are unnormalized.
        - ``loglikes``: log-values of the likelihoods (unnormalized, in general),
          in the same order as in ``list([your_model].likelihood)``.
        - ``derived``: values of the derived parameters in the order given by
          ``list([your_model].parameterization.derived_params())``.

        Only computes the log-likelihood and the derived parameters if the prior is
        non-null (otherwise the fields ``loglikes`` and ``derived`` are empty lists).

        To ignore the derived parameters, make ``return_derived=False``.

        If ``make_finite=True``, it will try to represent infinities as the largest real
        numbers allowed by machine precision.

        If ``cached=False`` (default: True), it ignores previously computed results that
        could be reused.
        """
        if _no_check:
            params_values_array = params_values
        else:
            if hasattr(params_values, "keys"):
                params_values = self.parameterization.check_sampled(**params_values)
            params_values_array = self._to_sampled_array(params_values)
            if self.log.getEffectiveLevel() <= logging.DEBUG:
                self.log.debug(
                    "Posterior to be computed for parameters %s",
                    dict(zip(self.parameterization.sampled_params(),
                             params_values_array)))
            if not np.all(np.isfinite(params_values_array)):
                raise LoggedError(
                    self.log, "Got non-finite parameter values: %r",
                    dict(zip(self.parameterization.sampled_params(),
                             params_values_array)))

        # Notice that we don't use the make_finite in the prior call,
        # to correctly check if we have to compute the likelihood
        logps = self.prior.logps_internal(params_values_array)
        if logps == -np.inf:
            logpriors = [-np.inf] * (1 + len(self.prior.external))
            logpost = -np.inf
        else:
            input_params = self.parameterization.to_input(params_values_array)
            logpriors = [logps]
            if self.prior.external:
                logpriors.extend(self.prior.logps_external(input_params))
                logpost = sum(logpriors)
            else:
                logpost = logps

        if logps != -np.inf:
            # noinspection PyUnboundLocalVariable
            like = self._loglikes_input_params(input_params,
                                               return_derived=return_derived,
                                               cached=cached, make_finite=make_finite)
            loglikes, derived_sampler = like if return_derived else (like, [])
            logpost += sum(loglikes)
        else:
            loglikes = []
            derived_sampler = []
        if make_finite:
            logpriors = np.nan_to_num(logpriors)
            logpost = np.nan_to_num(logpost)
        return LogPosterior(logpost=logpost, logpriors=logpriors,
                            loglikes=loglikes, derived=derived_sampler)

    def logpost(self, params_values, make_finite=False, cached=True) -> float:
        """
        Takes an array or dictionary of sampled parameter values.
        If the argument is an array, parameters must have the same order as in the input.
        When in doubt, you can get the correct order as
        ``list([your_model].parameterization.sampled_params())``.

        Returns the log-value of the posterior.

        If ``make_finite=True``, it will try to represent infinities as the largest real
        numbers allowed by machine precision.

        If ``cached=False`` (default: True), it ignores previously computed results that
        could be reused.
        """
        return self.logposterior(params_values, make_finite=make_finite,
                                 return_derived=False, cached=cached).logpost

    def get_valid_point(self, max_tries, ignore_fixed_ref=False, random_state=None):
        """
        Finds a point with finite posterior, sampled from from the reference pdf.

        It will fail if no valid point is found after `max_tries`.

        If `ignored_fixed_ref=True` (default: `False`), fixed reference values will be
        ignored in favor of the full prior, ensuring some randomness for all parameters
        (useful e.g. to prevent caching when measuring speeds).

        Returns (point, LogPosterior(logpost, logpriors, loglikes, derived))
        """
        for loop in range(max(1, max_tries // self.prior.d())):
            initial_point = self.prior.reference(max_tries=max_tries,
                                                 ignore_fixed=ignore_fixed_ref,
                                                 warn_if_no_ref=not loop,
                                                 random_state=random_state)
            results = self.logposterior(initial_point)
            if results.logpost != -np.inf:
                break
        else:
            if self.prior.reference_is_pointlike():
                raise LoggedError(self.log, "The reference point provided has null "
                                            "likelihood. Set 'ref' to a different point "
                                            "or a pdf.")
            raise LoggedError(self.log, "Could not find random point giving finite "
                                        "posterior after %g tries", max_tries)
        return initial_point, results

    def dump_timing(self):
        """
        Prints the average computation time of the theory code and likelihoods.

        It's more reliable the more times the likelihood has been evaluated.
        """
        self.likelihood.dump_timing()
        self.theory.dump_timing()

    # Python magic for the "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type=None, exception_value=None, traceback=None):
        self.likelihood.__exit__(exception_type, exception_value, traceback)
        self.theory.__exit__(exception_type, exception_value, traceback)

    def close(self):
        self.__exit__()

    def get_versions(self, add_version_field=False):
        return dict(theory=self.theory.get_versions(add_version_field=add_version_field),
                    likelihood=self.likelihood.get_versions(
                        add_version_field=add_version_field))

    def get_speeds(self, ignore_sub=False):
        return dict(theory=self.theory.get_speeds(ignore_sub=ignore_sub),
                    likelihood=self.likelihood.get_speeds(ignore_sub=ignore_sub))

    def _set_component_order(self, components, dependencies):
        dependence_order: List[Theory] = []
        deps = {p: s.copy() for p, s in dependencies.items()}
        comps = [c for c in components if not isinstance(c, AbsorbUnusedParamsLikelihood)]
        target_length = len(comps)
        _last = 0
        while len(dependence_order) < target_length:
            for component in list(comps):
                if not deps.get(component):
                    dependence_order.append(component)
                    comps.remove(component)
                    for p, dep in deps.items():
                        dep.discard(component)
            if len(dependence_order) == _last:
                raise LoggedError(self.log, "Circular dependency, cannot calculate "
                                            "%r" % comps)
            _last = len(dependence_order)
        likes = list(self.likelihood.values())
        self._component_order = {c: likes.index(c) if c in likes else None
                                 for c in dependence_order}

    def _set_dependencies_and_providers(self, manual_requirements=empty_dict,
                                        skip_unused_theories=False):
        components: List[Theory] = self.components
        direct_param_dependence: Dict[Theory, Set[str]] = {c: set() for c in components}

        def _tidy_requirements(_require, _component=None):
            # take input requirement and return tuples of
            # requirement names and requirement options
            if not _require:
                return []
            _require = as_requirement_list(_require)
            requirements_in_input_params = set(
                req.name for req in _require).intersection(self.input_params)
            if requirements_in_input_params and _component is not None:
                # Save parameters dependence
                direct_param_dependence[_component].update(requirements_in_input_params)
                # requirements that are sampled parameters automatically satisfied
                return [_req for _req in _require
                        if _req.name not in requirements_in_input_params]
            else:
                return _require

        # ## 1. Get the requirements and providers ##
        # requirements of each component
        requirements: Dict[Theory, List[Requirement]] = {}
        # providers of each *available* requirement (requested or not)
        providers: Dict[str, List[Theory]] = {}
        # set of requirement names that may be required derived parameters
        requirements_are_params: Set[str] = set()
        for component in components:
            # MARKED FOR DEPRECATION IN v3.0
            if hasattr(component, "add_theory"):
                raise LoggedError(self.log,
                                  "Please remove add_theory from %r and return "
                                  "requirement dictionary from get_requirements() "
                                  "instead" % component)
            # END OF DEPRECATION BLOCK
            component.initialize_with_params()
            requirements[component] = \
                _tidy_requirements(component.get_requirements(), component)
            # Component params converted to requirements if not explicitly sampled
            requirements[component] += \
                [Requirement(p, None) for p in (getattr(component, "params", {}) or []) if
                 p not in self.input_params and p not in component.output_params]
            # Gather what this component can provide
            can_provide = (list(component.get_can_provide()) +
                           list(component.get_can_provide_methods()))
            # Parameters that can be provided
            # Corner case: some components can either take some parameters as input OR
            # provide their own calculation of them. Pop those if required as input.
            provide_params = set(p for p in component.get_can_provide_params() if
                                 all(p != req.name for req in requirements[component]))
            provide_params.update(component.output_params)
            requirements_are_params.update(provide_params)
            # Invert to get the provider(s) of each available product/parameter
            for k in chain(can_provide, provide_params):
                providers[k] = providers.get(k, []) + [component]
        # Add requirements requested by hand
        manual_theory = Theory(name='_manual')
        if manual_requirements:
            self._manual_requirements = getattr(self, "_manual_requirements", []) \
                                        + _tidy_requirements(manual_requirements)
            requirements[manual_theory] = deepcopy(self._manual_requirements)

        # ## 2. Assign each requirement to a provider ##
        # store requirements assigned to each provider:
        self._must_provide: Dict[Theory, List[Requirement]] = {c: [] for c in components}
        # inverse of the one above, *minus conditional requirements* --
        # this is the one we need to initialise the provider
        requirement_providers = {}
        dependencies: Dict[
            Theory, Set[Theory]] = {}  # set of components of which each component depends
        used_suppliers = set(c for c in components if c.output_params)
        there_are_more_requirements = True
        # temp list of dictionary of requirements for each component
        must_provide = {c: [Requirement(p, None) for p in c.output_params] for c in
                        components}
        while there_are_more_requirements:
            # Check supplier for each requirement, get dependency and must_provide
            for component, requires in requirements.items():
                for requirement in requires:
                    suppliers = providers.get(requirement.name)
                    if not suppliers:
                        raise LoggedError(self.log,
                                          "Requirement %s of %r is not provided by any "
                                          "component, nor sampled directly",
                                          requirement.name, component)
                    supplier: Optional[Theory]
                    if len(suppliers) == 1:
                        supplier = suppliers[0]
                    else:
                        supplier = None
                        for sup in suppliers:
                            provide = str_to_list(getattr(sup, 'provides', []))
                            if requirement.name in provide:
                                if supplier:
                                    raise LoggedError(
                                        self.log, "more than one component provides %s",
                                        requirement.name)
                                supplier = sup
                        if not supplier:
                            raise LoggedError(
                                self.log, "requirement %s is provided by more than one "
                                          "component: %s. Use 'provides' keyword to "
                                          "specify which provides it",
                                requirement.name, suppliers)
                    if supplier is component:
                        raise LoggedError(
                            self.log, "Component %r cannot provide %s to itself!",
                            component, requirement.name)
                    requirement_providers[requirement.name] = supplier.get_provider()
                    used_suppliers.add(supplier)
                    declared_requirements_for_this_supplier = \
                        self._must_provide[supplier] + must_provide[supplier]
                    if requirement not in declared_requirements_for_this_supplier:
                        must_provide[supplier] += [requirement]
                    dependencies[component] = \
                        dependencies.get(component, set()) | {supplier}
                    # Requirements per component excluding input params
                    # -- saves some overhead in theory.check_cache_and_compute
                    if component is not manual_theory and \
                            requirement.name not in component.input_params and \
                            requirement.name in requirements_are_params:
                        component.input_params_extra.add(requirement.name)
            # tell each component what it must provide, and collect the
            # conditional requirements for those requests
            there_are_more_requirements = False
            for component, requires in requirements.items():
                # empty the list of requirements, since they have already been assigned,
                # and store here new (conditional) ones
                requires[:] = []
                # .get here accounts for the dummy components of manual reqs
                for request in must_provide.get(component) or []:
                    conditional_requirements = \
                        _tidy_requirements(
                            component.must_provide(
                                **{request.name: request.options}), component)
                    self._must_provide[component].append(request)
                    if conditional_requirements:
                        there_are_more_requirements = True
                        requires += conditional_requirements
            # set component compute order and raise error if circular dependence
            self._set_component_order(components, dependencies)
            # TODO: it would be nice that after this loop we, in some way, have
            # component.get_requirements() return the conditional reqs actually used too,
            # or maybe assign conditional used ones to a property?
            # Reset list
            must_provide = {c: [] for c in components}
        # Expunge manual requirements
        requirements.pop(manual_theory, None)
        # Check that unassigned input parameters are at least required by some component
        if self._unassigned_input:
            self._unassigned_input.difference_update(*direct_param_dependence.values())
            if self._unassigned_input:
                unassigned = self._unassigned_input - self.prior.external_dependence
                if unassigned:
                    raise LoggedError(
                        self.log, "Could not find anything to use input parameter(s) %r.",
                        unassigned)
                else:
                    self.log.warning("Parameter(s) %s are only used by the prior",
                                     self._unassigned_input)

        unused_theories = set(self.theory.values()) - used_suppliers
        if unused_theories:
            if skip_unused_theories:
                self.log.debug('Theories %s do not need to be computed '
                               'and will be skipped', unused_theories)
                for theory in unused_theories:
                    self._component_order.pop(theory, None)
                    components.remove(theory)
            else:
                self.log.warning('Theories %s do not appear to be actually used '
                                 'for anything', unused_theories)

        if self.log.getEffectiveLevel() <= logging.DEBUG:
            self.log.debug("Components will be computed in the order:")
            self.log.debug(" - %r" % list(self._component_order))

        def dependencies_of(_component):
            deps = set()
            for c in dependencies.get(_component, []):
                deps.add(c)
                deps.update(dependencies_of(c))
            return deps

        # ## 3. Save dependencies on components and their parameters ##
        self._dependencies = {c: dependencies_of(c) for c in components}
        # this next one is not a dict to save a lookup per iteration
        self._params_of_dependencies: List[Set[str]] \
            = [set() for _ in self._component_order]
        for component, param_dep in zip(self._component_order,
                                        self._params_of_dependencies):
            param_dep.update(direct_param_dependence.get(component) or [])
            for dep in self._dependencies.get(component, []):
                param_dep.update(
                    set(dep.input_params).union(direct_param_dependence.get(dep) or []))
            param_dep -= set(component.input_params)
            if not len(component.input_params) and not param_dep \
                    and component.get_name() != 'one':
                raise LoggedError(
                    self.log, "Component '%r' seems not to depend on any parameters "
                              "(neither directly nor indirectly)", component)
        # Store the input params and components on which each sampled params depends.
        sampled_input_dependence = self.parameterization.sampled_input_dependence()
        sampled_dependence: Dict[str, List[Theory]] = {p: []
                                                       for p in sampled_input_dependence}
        for p, i_s in sampled_input_dependence.items():
            for component in components:
                if p in component.input_params or i_s and \
                        any(p_i in component.input_params for p_i in i_s):
                    sampled_dependence[p].append(component)
                    for comp in components:
                        if comp is not component and \
                                component in self._dependencies.get(comp, []):
                            sampled_dependence[p].append(comp)
        self.sampled_dependence = sampled_dependence
        self.requires_derived: Set[str] = \
            requirements_are_params.intersection(requirement_providers)

        # ## 4. Initialize the provider and pass it to each component ##
        if self.log.getEffectiveLevel() <= logging.DEBUG:
            if requirement_providers:
                self.log.debug("Requirements will be calculated by these components:")
                for req, provider in requirement_providers.items():
                    self.log.debug("- %s: %s", req, provider)
            else:
                self.log.debug("No requirements need to be computed")

        self.provider = Provider(self, requirement_providers)
        for component in components:
            component.initialize_with_provider(self.provider)

    def add_requirements(self, requirements):
        """
        Adds quantities to be computed by the pipeline, for testing purposes.
        """
        self._set_dependencies_and_providers(manual_requirements=requirements)

    def requested(self):
        """
        Get all the requested requirements that will be computed.

        :return: dictionary giving list of requirements calculated by
                each component name
        """
        return {"%r" % c: v for c, v in self._must_provide.items() if v}

    def _assign_params(self, info_likelihood, info_theory=None,
                       dropped_theory_params=None):
        """
        Assign input and output parameters to theories and likelihoods, following the
        algorithm explained in :doc:`DEVEL`.
        """
        self.input_params = [p for p in self.parameterization.input_params() if p not in
                             self.parameterization.dropped_param_set()]
        self.output_params = list(self.parameterization.output_params())
        input_assign: Dict[str, List[Theory]] = {p: [] for p in self.input_params}
        output_assign: Dict[str, List[Theory]] = {p: [] for p in self.output_params}
        # Go through all components.
        # NB: self.components iterates over likelihoods first, and then theories
        # so unassigned can by default go to theories
        assign_components = [c for c in self.components
                             if not isinstance(c, AbsorbUnusedParamsLikelihood)]
        for assign, option, prefix, derived_param in (
                (input_assign, "input_params", "input_params_prefix", False),
                (output_assign, "output_params", "output_params_prefix", True)):
            agnostic_likes = []
            for component in assign_components:
                if derived_param:
                    required_params = set(str_to_list(getattr(component, "provides", [])))
                else:
                    required_params = set(
                        p for p, v in as_requirement_list(component.get_requirements())
                        # ignore non-params; it's ok if some non-param goes through
                        if v is None)
                # Identify parameters understood by this likelihood/theory
                # 1a. Does it have input/output params list?
                #     (takes into account that for callables, we can ignore elements)
                if getattr(component, option) is not unset_params:
                    for p in getattr(component, option):
                        try:
                            assign[p] += [component]
                        except KeyError:
                            if not derived_param:
                                raise LoggedError(
                                    self.log,
                                    "Parameter '%s' needed as input for '%s', "
                                    "but not provided.", p, component.get_name())
                # 2. Is there a params prefix?
                elif getattr(component, prefix, None) is not None:
                    for p in assign:
                        if p.startswith(getattr(component, prefix)):
                            assign[p] += [component]
                # 3. Does it have a general (mixed) list of params? (set from default)
                # 4. or otherwise required
                elif getattr(component, "params", None) or required_params:
                    if getattr(component, "params", None):
                        for p, options in getattr(component, "params", {}).items():
                            if not isinstance(options, Mapping) and not derived_param or \
                                    isinstance(options, Mapping) and \
                                    options.get('derived', False) is derived_param:
                                if p in assign:
                                    assign[p] += [component]
                    elif component.get_allow_agnostic():
                        agnostic_likes += [component]
                    if required_params:
                        for p in required_params:
                            if p in assign and component not in assign[p]:
                                assign[p] += [component]
                # 5. No parameter knowledge: store as parameter agnostic
                elif component.get_allow_agnostic():
                    agnostic_likes += [component]

            # 6. If parameter not already assigned give to any component that supports it
            #    Input parameters always assigned to any supporting component.
            unassigned = [p for p in assign if not assign[p]]
            for component in assign_components:
                if derived_param:
                    supports_params = component.get_can_provide_params()
                else:
                    supports_params = component.get_can_support_params()
                pars_to_assign = set(supports_params)
                if dropped_theory_params and not is_LikelihoodInterface(component):
                    pars_to_assign.difference_update(dropped_theory_params)
                for p in (unassigned if derived_param else assign):
                    if p in pars_to_assign and component not in assign[p]:
                        assign[p] += [component]
            # Check that there is only one non-knowledgeable element, and assign
            # unused params
            if len(agnostic_likes) > 1 and not all(assign.values()):
                raise LoggedError(
                    self.log, "More than one parameter-agnostic likelihood/theory "
                              "with respect to %s: %r. Cannot decide "
                              "parameter assignments.", option, agnostic_likes)
            elif agnostic_likes:  # if there is only one
                component = agnostic_likes[0]
                for p, assigned in assign.items():
                    if not assigned:
                        assign[p] += [component]

        # If unit likelihood is present, assign all unassigned inputs to it
        for like in self.likelihood.values():
            if isinstance(like, AbsorbUnusedParamsLikelihood):
                for p, assigned in input_assign.items():
                    if not assigned:
                        assigned.append(like)
                break
        # If there are unassigned input params, check later that they are used by
        # *conditional* requirements of a component (and if not raise error)
        self._unassigned_input = set(
            p for p, assigned in input_assign.items() if not assigned).difference(
            chain(*(self.parameterization.input_dependencies.get(p, []) for p, assigned
                    in input_assign.items() if assigned)))

        chi2_names: Dict[str, List[int]] = {}
        # Add aggregated chi2 for likelihood types
        for i, like in enumerate(self.likelihood.values()):
            for tp in like.type_list:
                name = get_chi2_name(tp)
                if name not in chi2_names:
                    chi2_names[name] = []
                chi2_names[name].append(i)
        # Remove aggregated chi2 that may have been picked up by an agnostic component
        for chi2_name in chi2_names:
            output_assign.pop(chi2_name, None)
        # If chi2__like single-likelihood parameters are explicitly requested, include
        # in derived outputs for use in on output derived parameters
        for p in output_assign:
            if p.startswith(get_chi2_name("")):
                like = p[len(get_chi2_name("")):]
                index = list(self.likelihood).index(like)
                if index is None:
                    raise LoggedError(
                        self.log, "Your derived parameters depend on an unknown "
                                  "likelihood: '%s'", like)
                if p in chi2_names:
                    raise LoggedError(
                        self.log, "Your have likelihoods with type labels that are the "
                                  "same as a likelihood", like)
                chi2_names[p] = [index]
                # They may have been already assigned to an agnostic likelihood,
                # so purge first: no "=+"
                output_assign[p] = [self.likelihood[like]]
        self._chi2_names = tuple(chi2_names.items())
        # Check that there are no unassigned parameters (with the exception of aggr chi2)
        unassigned_output = [p for p, assigned in output_assign.items() if not assigned]
        if unassigned_output:
            raise LoggedError(
                self.log, "Could not find whom to assign output parameters %r.",
                unassigned_output)
        # Check that output parameters are assigned exactly once
        multiassigned_output = {p: assigned for p, assigned in output_assign.items()
                                if len(assigned) > 1}
        if multiassigned_output:
            raise LoggedError(
                self.log,
                "Output params can only be computed by one likelihood/theory, "
                "but some were claimed by more than one: %r.", multiassigned_output)
        # Finished! Assign and update infos
        for assign, option in ((input_assign, "input_params"),
                               (output_assign, "output_params")):
            for component in self.components:
                setattr(component, option,
                        [p for p, assign in assign.items() if component in assign])
                # Update infos! (helper theory parameters stored in yaml with host)
                inf = (info_likelihood if component in self.likelihood.values() else
                       info_theory)
                inf = inf.get(component.get_name())
                if inf:
                    inf.pop("params", None)
                    inf[option] = component.get_attr_list_with_helpers(option)
        if self.log.getEffectiveLevel() <= logging.DEBUG:
            self.log.debug("Parameters were assigned as follows:")
            for component in self.components:
                self.log.debug("- %r:", component)
                self.log.debug("     Input:  %r", component.input_params)
                self.log.debug("     Output: %r", component.output_params)

    @property
    def components(self) -> List[Theory]:
        return list(chain(self.likelihood.values(), self.theory.values()))

    def get_param_blocking_for_sampler(self, split_fast_slow=False, oversample_power=0):
        """
        Separates the sampled parameters in blocks according to the component(s)
        re-evaluation(s) that changing each one of them involves. Then sorts these blocks
        in an optimal way using the speed (i.e. inverse evaluation time in seconds)
        of each component.

        Returns tuples of ``(params_in_block), (oversample_factor)``,
        sorted by descending variation cost-per-parameter.

        Set ``oversample_power`` to some value between 0 and 1 to control the amount of
        oversampling (default: 0 -- no oversampling). If close enough to 1, it chooses
        oversampling factors such that the same amount of time is spent in each block.

        If ``split_fast_slow=True``, it separates blocks in two sets, only the second one
        having an oversampling factor >1. In that case, the oversampling factor should be
        understood as the total one for all the fast blocks.
        """
        # Get a list of components and their speeds
        speeds = {c.get_name(): getattr(c, "speed", -1) for c in self.components}
        # Add overhead to defined ones (positives)
        # and clip undefined ones to the slowest one
        try:
            min_speed = min(speed for speed in speeds.values() if speed > 0)
        except ValueError:
            # No speeds defined <-- empty sequence passed to min
            min_speed = 1
        for comp in speeds:
            speeds[comp] = max(speeds[comp], min_speed)
            # For now, overhead is constant re # params and very small
            speeds[comp] = (speeds[comp] ** -1 + self.overhead) ** -1
        # Compute "footprint"
        # i.e. likelihoods (and theory) that we must recompute when each parameter changes
        footprints = np.zeros((len(self.sampled_dependence), len(speeds)), dtype=int)
        sampled_dependence_names = {k: [c.get_name() for c in v] for
                                    k, v in self.sampled_dependence.items()}
        for i, ls in enumerate(sampled_dependence_names.values()):
            for j, comp in enumerate(speeds):
                footprints[i, j] = comp in ls
        # Group parameters by footprint
        different_footprints = list(set(tuple(row) for row in footprints))
        blocks = [[p for ip, p in enumerate(self.sampled_dependence)
                   if all(footprints[ip] == fp)] for fp in different_footprints]
        # a) Multiple blocks
        if not split_fast_slow:
            i_optimal_ordering, costs, oversample_factors = sort_parameter_blocks(
                blocks, np.array(list(speeds.values()), dtype=float),
                different_footprints, oversample_power=oversample_power)
            blocks_sorted = [blocks[i] for i in i_optimal_ordering]
        # b) 2-block slow-fast separation
        else:
            if len(blocks) == 1:
                raise LoggedError(self.log, "Requested fast/slow separation, "
                                            "but all parameters have the same speed.")
            # First sort them optimally (w/o oversampling)
            i_optimal_ordering, costs, oversample_factors = sort_parameter_blocks(
                blocks, np.array(list(speeds.values()), dtype=float),
                different_footprints, oversample_power=0)
            blocks_sorted = [blocks[i] for i in i_optimal_ordering]
            footprints_sorted = np.array(different_footprints)[list(i_optimal_ordering)]
            # Then, find the split that maxes cost LOG-differences.
            # Since costs are already "accumulated down",
            # we need to subtract those below each one
            costs_per_block = costs - np.concatenate([costs[1:], [0]])
            # Split them so that "adding the next block to the slow ones" has max cost
            log_differences = np.log(costs_per_block[:-1]) - np.log(costs_per_block[1:])
            i_last_slow: int = np.argmax(log_differences)  # type: ignore
            blocks_split = (lambda l: [list(chain(*l[:i_last_slow + 1])),
                                       list(chain(*l[i_last_slow + 1:]))])(blocks_sorted)
            footprints_split = (
                    [np.array(footprints_sorted[:i_last_slow + 1]).sum(axis=0)] +
                    [np.array(footprints_sorted[i_last_slow + 1:]).sum(axis=0)])
            footprints_split = np.clip(np.array(footprints_split), 0, 1)  # type: ignore
            # Recalculate oversampling factor with 2 blocks
            _, _, oversample_factors = sort_parameter_blocks(
                blocks_split, np.array(list(speeds.values()), dtype=float),
                footprints_split, oversample_power=oversample_power)
            # If no oversampling, slow-fast separation makes no sense: warn and set to 2
            if oversample_factors[1] == 1:
                min_factor = 2
                self.log.warning(
                    "Oversampling would be trivial due to small speed difference or "
                    "small `oversample_power`. Set to %d.", min_factor)
            # Finally, unfold `oversampling_factors` to have the right number of elements,
            # taking into account that that of the fast blocks should be interpreted as a
            # global one for all of them.
            # NB: the int() below forces the copy of the factors.
            #     Otherwise the yaml_representer prints references to a single object.
            oversample_factors = (
                    [int(oversample_factors[0])] * (1 + i_last_slow) +
                    [int(oversample_factors[1])] * (len(blocks) - (1 + i_last_slow)))
            self.log.debug("Doing slow/fast split. The oversampling factors for the fast "
                           "blocks should be interpreted as a global one for all of them")
        self.log.debug(
            "Cost, oversampling factor and parameters per block, in optimal order:")
        for c, o, b in zip(costs, oversample_factors, blocks_sorted):
            self.log.debug("* %g : %r : %r", c, o, b)
        return blocks_sorted, oversample_factors

    def check_blocking(self, blocking):
        """
        Checks the correct formatting of the given parameter blocking and oversampling:
        that it consists of tuples `(oversampling_factor, (param1, param2, etc))`, with
        growing oversampling factors, and containing all parameters.

        Returns the input, once checked as ``(blocks), (oversampling_factors)``.

        If ``draggable=True`` (default: ``False``), checks that the oversampling factors
        are compatible with dragging.
        """
        try:
            oversampling_factors, blocks = zip(*list(blocking))
        except:
            raise LoggedError(
                self.log, "Manual blocking not understood. Check documentation.")
        sampled_params = list(self.sampled_dependence)
        check = are_different_params_lists(list(chain(*blocks)), sampled_params)
        duplicate = check.pop("duplicate_A", None)
        missing = check.pop("B_but_not_A", None)
        unknown = check.pop("A_but_not_B", None)
        if duplicate:
            raise LoggedError(
                self.log, "Manual blocking: repeated parameters: %r", duplicate)
        if missing:
            raise LoggedError(
                self.log, "Manual blocking: missing parameters: %r", missing)
        if unknown:
            raise LoggedError(
                self.log, "Manual blocking: unknown parameters: %r", unknown)
        oversampling_factors = np.array(oversampling_factors)
        if (oversampling_factors != np.sort(oversampling_factors)).all():
            self.log.warning(
                "Manual blocking: speed-blocking *apparently* non-optimal: "
                "oversampling factors must go from small (slow) to large (fast).")
        return blocks, oversampling_factors

    def set_cache_size(self, n_states):
        """
        Sets the number of different parameter points to cache for all theories
        and likelihoods.

        :param n_states: number of cached points
        """
        for theory in self.components:
            theory.set_cache_size(n_states)

    def get_auto_covmat(self, params_info=None, random_state=None):
        """
        Tries to get an automatic covariance matrix for the current model and data.

        ``params_info`` should include the set of parameters for which the covmat will be
        searched (default: None, meaning all sampled parameters).
        """
        if params_info is None:
            params_info = self.parameterization.sampled_params_info()
        try:
            for theory in self.theory.values():
                if hasattr(theory, 'get_auto_covmat'):
                    return theory.get_auto_covmat(
                        params_info, self.info()["likelihood"], random_state=random_state)
        except Exception as e:
            self.log.warning("Something went wrong when looking for a covmat: %r", str(e))
            return None

    def set_timing_on(self, on):
        self.timing = on
        for component in self.components:
            component.set_timing_on(on)

    def measure_and_set_speeds(self, n=None, discard=1, max_tries=np.inf,
                               random_state=None):
        """
        Measures the speeds of the different components (theories and likelihoods). To do
        that it evaluates the posterior at `n` points (default: 1 per MPI process, or 3 if
        single process), discarding `discard` points (default: 1) to mitigate possible
        internal caching.

        Stops after encountering `max_tries` points (default: inf) with non-finite
        posterior.
        """
        self.mpi_info("Measuring speeds... (this may take a few seconds)")
        if n is None:
            n = 1 if mpi.more_than_one_process() else 3
        n_done = 0
        with timing_on(self):
            while n_done < int(n) + int(discard):
                point = self.prior.reference(random_state=random_state,
                                             max_tries=max_tries, ignore_fixed=True,
                                             warn_if_no_ref=False)
                if self.loglike(point, cached=False)[0] != -np.inf:
                    n_done += 1
            self.log.debug("Computed %d points to measure speeds.", n_done)
            times = [component.timer.get_time_avg() or 0  # type: ignore
                     for component in self.components]
        if mpi.more_than_one_process():
            # average for different points
            times = np.average(mpi.allgather(times), axis=0)
        measured_speeds = [1 / (1e-7 + time) for time in times]
        self.mpi_info('Setting measured speeds (per sec): %r',
                      {component: float("%.3g" % speed) for component, speed in
                       zip(self.components, measured_speeds)})

        for component, speed in zip(self.components, measured_speeds):
            component.set_measured_speed(speed)


def get_model(info_or_yaml_or_file: Union[InputDict, str, os.PathLike],
              debug: Optional[bool] = None,
              stop_at_error: Optional[bool] = None,
              packages_path: Optional[str] = None,
              override: Optional[InputDict] = None
              ) -> Model:
    info = load_info_overrides(info_or_yaml_or_file, debug, stop_at_error,
                               packages_path, override)
    logger_setup(info.pop("debug", debug_default), info.pop("debug_file", None))

    # Inform about ignored info keys
    ignored_info = []
    for k in list(info):
        if k not in ["params", "likelihood", "prior", "theory", "packages_path",
                     "timing", "stop_at_error", "auto_params"]:
            value = info.pop(k)  # type: ignore
            if value is not None and (not isinstance(value, Mapping) or value):
                ignored_info.append(k)
    # Create the updated input information, including defaults for each component.
    updated_info = update_info(info)
    if ignored_info:
        logging.getLogger(__name__.split(".")[-1]).warning(
            "Ignored blocks/options: %r", ignored_info)
    if logging.root.getEffectiveLevel() <= logging.DEBUG:
        logging.getLogger(__name__.split(".")[-1]).debug(
            "Input info updated with defaults (dumped to YAML):\n%s",
            yaml_dump(sort_cosmetic(updated_info)))
    # Initialize the parameters and posterior
    return Model(updated_info["params"], updated_info["likelihood"],
                 updated_info.get("prior"), updated_info.get("theory"),
                 packages_path=info.get("packages_path"),
                 timing=updated_info.get("timing"),
                 stop_at_error=info.get("stop_at_error", False))


def load_info_overrides(info_or_yaml_or_file, debug, stop_at_error,
                        packages_path, override=None) -> InputDict:
    info = load_input_dict(info_or_yaml_or_file)  # makes deep copy if dict

    if override:
        if "post" in override:
            info["resume"] = False
        info = recursive_update(info, override, copied=False)
    if packages_path:
        info["packages_path"] = packages_path
    if debug is not None:
        info["debug"] = bool(debug)
    if stop_at_error is not None:
        info["stop_at_error"] = bool(stop_at_error)
    return info
