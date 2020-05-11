"""
.. module:: model

:Synopsis: Wrapper for models: parameterization+prior+likelihood+theory
:Author: Jesus Torrado

"""
# Global
import logging
from itertools import chain
from typing import NamedTuple, Sequence, Mapping
import numpy as np

# Local
from cobaya.conventions import kinds, _prior, _timing, _params, _provides, \
    _overhead_time, _packages_path, _debug, _debug_default, _debug_file, _input_params, \
    _output_params, _get_chi2_name, _input_params_prefix, \
    _output_params_prefix, empty_dict
from cobaya.input import update_info
from cobaya.parameterization import Parameterization
from cobaya.prior import Prior
from cobaya.likelihood import LikelihoodCollection, LikelihoodExternalFunction, \
    AbsorbUnusedParamsLikelihood
from cobaya.theory import TheoryCollection
from cobaya.log import LoggedError, logger_setup, HasLogger
from cobaya.yaml import yaml_dump
from cobaya.tools import deepcopy_where_possible, are_different_params_lists, \
    str_to_list, sort_parameter_blocks, recursive_update, sort_cosmetic, ensure_dict
from cobaya.component import Provider
from cobaya.mpi import more_than_one_process, get_mpi_comm


# Log-posterior namedtuple
class LogPosterior(NamedTuple):
    logpost: float = None
    logpriors: Sequence[float] = None
    loglikes: Sequence[float] = []
    derived: Sequence[float] = []


class Requirement(NamedTuple):
    name: str
    options: dict

    def __eq__(self, other):
        return self.name == other.name and _dict_equal(self.options, other.options)

    def __repr__(self):
        return "{%r:%r}" % (self.name, self.options)


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
        if set(list(d1)) != set(list(d2)):
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


def get_model(info):
    assert isinstance(info, Mapping), (
        "The first argument must be a dictionary with the info needed for the model. "
        "If you were trying to pass the name of an input file instead, "
        "load it first with 'cobaya.input.load_input', "
        "or, if you were passing a yaml string, load it with 'cobaya.yaml.yaml_load'.")
    info = deepcopy_where_possible(info)
    logger_setup(info.pop(_debug, _debug_default), info.pop(_debug_file, None))
    # Inform about ignored info keys
    ignored_info = {}
    for k in list(info):
        if k not in [_params, kinds.likelihood, _prior, kinds.theory, _packages_path,
                     _timing, "stop_at_error"]:
            ignored_info[k] = info.pop(k)
    if ignored_info:
        logging.getLogger(__name__.split(".")[-1]).warning(
            "Ignored blocks/options: %r", list(ignored_info))
    # Create the updated input information, including defaults for each component.
    updated_info = update_info(info)
    if logging.root.getEffectiveLevel() <= logging.DEBUG:
        logging.getLogger(__name__.split(".")[-1]).debug(
            "Input info updated with defaults (dumped to YAML):\n%s",
            yaml_dump(sort_cosmetic(updated_info)))
    # Initialize the parameters and posterior
    return Model(updated_info[_params], updated_info[kinds.likelihood],
                 updated_info.get(_prior), updated_info.get(kinds.theory),
                 packages_path=info.get(_packages_path), timing=updated_info.get(_timing),
                 stop_at_error=info.get("stop_at_error", False))


class Model(HasLogger):
    """
    Class containing all the information necessary to compute the unnormalized posterior.

    Allows for low-level interaction with the theory code, prior and likelihood.

    **NB:** do not initialize this class directly; use :func:`~model.get_model` instead,
    with some info as input.
    """

    def __init__(self, info_params, info_likelihood, info_prior=None, info_theory=None,
                 packages_path=None, timing=None, allow_renames=True, stop_at_error=False,
                 post=False, prior_parameterization=None):
        self.set_logger(lowercase=True)
        self._updated_info = {
            _params: deepcopy_where_possible(info_params),
            kinds.likelihood: deepcopy_where_possible(info_likelihood)}
        if not self._updated_info[kinds.likelihood]:
            raise LoggedError(self.log, "No likelihood requested!")
        for k, v in ((_prior, info_prior), (kinds.theory, info_theory),
                     (_packages_path, packages_path), (_timing, timing)):
            if v not in (None, {}):
                self._updated_info[k] = deepcopy_where_possible(v)
        self.parameterization = Parameterization(self._updated_info[_params],
                                                 allow_renames=allow_renames,
                                                 ignore_unused_sampled=post)
        self.prior = Prior(prior_parameterization or self.parameterization,
                           self._updated_info.get(_prior, None))
        self.timing = timing
        info_theory = self._updated_info.get(kinds.theory)
        self.theory = TheoryCollection(info_theory, packages_path=packages_path,
                                       timing=timing)
        info_likelihood = self._updated_info[kinds.likelihood]
        self.likelihood = LikelihoodCollection(info_likelihood, theory=self.theory,
                                               packages_path=packages_path, timing=timing)
        if stop_at_error:
            for component in self.components:
                component.stop_at_error = stop_at_error
        # Assign input/output parameters
        self._assign_params(info_likelihood, info_theory)
        self._set_dependencies_and_providers()
        # Add to the updated info some values that are only available after initialisation
        self._updated_info = recursive_update(
            self._updated_info, self.get_versions(add_version_field=True))
        # Overhead per likelihood evaluation, approximately ind from # input params
        # Evaluation of non-uniform priors will add some overhead per parameter.
        self.overhead = _overhead_time

    def info(self):
        """
        Returns a copy of the information used to create the model, including defaults
        and some new values that are only available after initialisation.
        """
        return deepcopy_where_possible(self._updated_info)

    def _to_sampled_array(self, params_values):
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

    def logpriors(self, params_values, make_finite=False):
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
        logpriors = self.prior.logps(params_values_array)
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
        derived_dict = {}
        compute_success = True
        self.provider.set_current_input_params(input_params)
        self.log.debug("Got input parameters: %r", input_params)
        n_theory = len(self.theory)
        loglikes = np.empty(len(self.likelihood))
        for (component, index), param_dep in zip(self._component_order.items(),
                                                 self._params_of_dependencies):
            depend_list = \
                [input_params[p] for p in param_dep]
            params = {p: input_params[p] for p in component.input_params}
            compute_success = component.check_cache_and_compute(
                want_derived=return_derived,
                dependency_params=depend_list,
                cached=cached, **params)
            if not compute_success:
                loglikes[:] = -np.inf
                self.log.debug(
                    "Calculation failed, skipping rest of calculations ")
                break
            if return_derived:
                derived_dict.update(component.get_current_derived())
            # Add chi2's to derived parameters
            if hasattr(component, "get_current_logp"):
                try:
                    loglikes[index - n_theory] = float(component.get_current_logp())
                except TypeError:
                    raise LoggedError(
                        self.log,
                        "Likelihood %s has not returned a valid log-likelihood, "
                        "but %r instead.", str(component), component.get_current_logp())
                if return_derived:
                    derived_dict[_get_chi2_name(component.get_name().replace(".", "_"))] \
                        = -2 * loglikes[index - n_theory]
                    for this_type in getattr(component, "type", []) or []:
                        aggr_chi2_name = _get_chi2_name(this_type)
                        if aggr_chi2_name not in derived_dict:
                            derived_dict[aggr_chi2_name] = 0
                        derived_dict[aggr_chi2_name] += -2 * loglikes[index - n_theory]
        if make_finite:
            loglikes = np.nan_to_num(loglikes)
        if return_derived:
            # Turn the derived params dict into a list and return
            if not compute_success:
                derived_list = [np.nan] * len(self.output_params)
            else:
                derived_list = [derived_dict[p] for p in self.output_params]
            return loglikes, derived_list
        return loglikes

    def loglikes(self, params_values=None, return_derived=True, make_finite=False,
                 cached=True, _no_check=False):
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
        elif hasattr(params_values, "keys") and not _no_check:
            params_values = self.parameterization.check_sampled(**params_values)

        input_params = self.parameterization.to_input(params_values, copied=False)

        result = self.logps(input_params, return_derived=return_derived,
                            cached=cached, make_finite=make_finite)
        if return_derived:
            loglikes, derived_list = result
            derived_sampler = self.parameterization.to_derived(derived_list)
            if self.log.getEffectiveLevel() <= logging.DEBUG:
                self.log.debug(
                    "Computed derived parameters: %s",
                    dict(zip(self.parameterization.derived_params(), derived_sampler)))
            return loglikes, derived_sampler
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
        ret_value = self.loglikes(
            params_values, return_derived=return_derived, cached=cached)
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

    def logposterior(
            self, params_values, return_derived=True, make_finite=False, cached=True):
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
        if hasattr(params_values, "keys"):
            params_values = self.parameterization.check_sampled(**params_values)
        params_values_array = self._to_sampled_array(params_values)
        if self.log.getEffectiveLevel() <= logging.DEBUG:
            self.log.debug(
                "Posterior to be computed for parameters %s",
                dict(zip(self.parameterization.sampled_params(), params_values_array)))
        if not np.all(np.isfinite(params_values_array)):
            raise LoggedError(
                self.log, "Got non-finite parameter values: %r",
                dict(zip(self.parameterization.sampled_params(), params_values_array)))
        # Notice that we don't use the make_finite in the prior call,
        # to correctly check if we have to compute the likelihood
        logpriors = self.prior.logps(params_values_array)
        logpost = sum(logpriors)
        if -np.inf not in logpriors:
            like = self.loglikes(params_values, return_derived=return_derived,
                                 make_finite=make_finite, cached=cached, _no_check=True)
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

    def logpost(self, params_values, make_finite=False, cached=True):
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
                                 return_derived=False, cached=cached)[0]

    def get_valid_point(self, max_tries, ignore_fixed_ref=False):
        """
        Finds a point with finite posterior, sampled from from the reference pdf.

        It will fail if no valid point is found after `max_tries`.

        If `ignored_fixed_ref=True` (default: `False`), fixed reference values will be
        ignored in favor of the full prior, ensuring some randomness for all parameters
        (useful e.g. to prevent caching when measuring speeds).

        Returns (point, logpost, logpriors, loglikes, derived)
        """
        for loop in range(max(1, max_tries // self.prior.d())):
            initial_point = self.prior.reference(max_tries=max_tries,
                                                 ignore_fixed=ignore_fixed_ref,
                                                 warn_if_no_ref=not loop)
            logpost, logpriors, loglikes, derived = self.logposterior(initial_point)
            if -np.inf not in loglikes:
                break
        else:
            if self.prior.reference_is_pointlike():
                raise LoggedError(self.log, "The reference point provided has null "
                                  "likelihood. Set 'ref' to a different point or a pdf.")
            raise LoggedError(self.log, "Could not find random point giving finite "
                                        "likelihood after %g tries", max_tries)
        return initial_point, logpost, logpriors, loglikes, derived

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
        dependence_order = []
        deps = {p: s.copy() for p, s in dependencies.items()}
        comps = components[:]
        _last = 0
        while len(dependence_order) < len(components):
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

        self._component_order = {c: components.index(c) for c in dependence_order}

    def _set_dependencies_and_providers(self, manual_requirements=empty_dict):
        # TODO: does it matter that theories come first, or can we use self.components?
        components = list(self.theory.values()) + list(self.likelihood.values())
        direct_param_dependence = {c: set() for c in components}

        def _tidy_requirements(_require, _component=None):
            # take input requirement dictionary and split into list of tuples of
            # requirement names and requirement options
            if not _require:
                return []
            if isinstance(_require, Mapping):
                _require = dict(_require)
            else:
                _require = dict.fromkeys(_require)
            # Save parameters dependence
            for par in self.input_params:
                if par in _require and _component is not None:
                    direct_param_dependence[_component].add(par)
                    # requirements that are sampled parameters automatically satisfied
                    _require.pop(par, None)
            return [Requirement(p, v) for p, v in _require.items()]

        ### 1. Get the requirements and providers ###
        requirements = {}  # requirements of each component
        providers = {}  # providers of each *available* requirement (requested or not)
        for component in components:
            # MARKED FOR DEPRECATION IN v3.0
            if hasattr(component, "add_theory"):
                raise LoggedError(self.log,
                                  "Please remove add_theory from %r and return "
                                  "requirement dictionary from get_requirements() "
                                  "instead" % component)
            # END OF DEPRECATION BLOCK
            component.initialize_with_params()
            requirements[component] = _tidy_requirements(
                component.get_requirements(), component)
            # Gather what this component can provide
            can_provide = (
                    list(component.get_can_provide()) +
                    list(component.get_can_provide_methods()))
            # Parameters that can be provided but not already explicitly assigned
            # (i.e. it is not a declared output param of that component)
            provide_params = [p for p in component.get_can_provide_params() if
                              p not in self.output_params]
            # Corner case: some components can either take some parameters as input OR
            # provide their own calculation of them. Pop those if required as input.
            for p in provide_params.copy():  # iterating over copy
                if p in component.get_requirements():  # no need to know which are params
                    provide_params.remove(p)
            # Invert to get the provider(s) of each available product/parameter
            for k in can_provide + component.output_params + provide_params:
                providers[k] = providers.get(k, []) + [component]
        # Add requirements requested by hand
        if manual_requirements:
            requirements[None] = _tidy_requirements(manual_requirements)

        ### 2. Assign each requirement to a provider ###
        # store requirements assigned to each provider:
        self._must_provide = {component: [] for component in components}
        # inverse of the one above, *minus conditional requirements* --
        # this is the one we need to initialise the provider
        requirement_providers = {}
        dependencies = {}  # set of components of which each component depends
        there_are_more_requirements = True
        while there_are_more_requirements:
            # temp list of dictionary of requirements for each component
            must_provide = {c: [] for c in components}
            # Check supplier for each requirement, get dependency and must_provide
            for component, requires in requirements.items():
                for requirement in requires:
                    suppliers = providers.get(requirement.name)
                    if not suppliers:
                        raise LoggedError(
                            self.log, "Requirement %s of %r is not provided by any "
                                      "component", requirement.name, component)
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
                    declared_requirements_for_this_supplier = \
                        self._must_provide[supplier] + must_provide[supplier]
                    if requirement not in declared_requirements_for_this_supplier:
                        must_provide[supplier] += [requirement]
                    dependencies[component] = \
                        dependencies.get(component, set()) | {supplier}
                    # Requirements per component excluding input params
                    # -- saves some overhead in theory.check_cache_and_compute
                    if component and requirement.name not in component.input_params and \
                            requirement.options is None:
                        component._input_params_extra.add(requirement.name)
            # tell each component what it must provide, and collect the
            # conditional requirements for those requests
            there_are_more_requirements = False
            for component, requires in requirements.items():
                # empty the list of requirements, since they have already been assigned,
                # and store here new (conditional) ones
                requires[:] = []
                # .get here accounts for the null component of manual reqs
                if must_provide.get(component, False):
                    for request in must_provide[component]:
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
        # Expunge manual requirements
        requirements.pop(None, None)
        # Check that unassigned input parameters are at least required by some component
        if self._unassigned_input:
            self._unassigned_input.difference_update(*direct_param_dependence.values())
            if self._unassigned_input:
                raise LoggedError(
                    self.log, "Could not find anything to use input parameter(s) %r.",
                    self._unassigned_input)
        if self.log.getEffectiveLevel() <= logging.DEBUG:
            self.log.debug("Components will be computed in the order:")
            self.log.debug(" - %r" % list(self._component_order))

        def dependencies_of(_component):
            deps = set()
            for c in dependencies.get(_component, []):
                deps.add(c)
                deps.update(dependencies_of(c))
            return deps

        ### 3. Save dependencies on components and their parameters ###
        self._dependencies = {c: dependencies_of(c) for c in components}
        # this next one is not a dict to save a lookup per iteration
        self._params_of_dependencies = [set() for _ in self._component_order]
        for component, param_dep in zip(self._component_order,
                                        self._params_of_dependencies):
            param_dep.update(direct_param_dependence.get(component))
            for dep in self._dependencies.get(component, []):
                param_dep.update(
                    set(dep.input_params).union(direct_param_dependence.get(dep)))
            param_dep -= set(component.input_params)
            if not len(component.input_params) and not param_dep \
                    and component.get_name() != 'one':
                raise LoggedError(
                    self.log, "Component '%r' seems not to depend on any parameters "
                              "(neither directly nor indirectly)", component)
        # Store the input params and components on which each sampled params depends.
        sampled_input_dependence = self.parameterization.sampled_input_dependence()
        sampled_dependence = {p: [] for p in sampled_input_dependence}
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

        ### 4. Initialize the provider and pass it to each component ###
        if self.log.getEffectiveLevel() <= logging.DEBUG and requirement_providers:
            self.log.debug("Requirements will be calculated by these components:")
            for requirement, provider in requirement_providers.items():
                self.log.debug("- %s: %s", requirement, provider)
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
        return dict(("%r" % c, v) for c, v in self._must_provide.items() if v)

    def _assign_params(self, info_likelihood, info_theory=None):
        """
        Assign parameters to theories and likelihoods, following the algorithm explained
        in :doc:`DEVEL`.
        """
        self.input_params = list(self.parameterization.input_params())
        self.output_params = list(self.parameterization.output_params())
        params_assign = {"input": {p: [] for p in self.input_params},
                         "output": {p: [] for p in self.output_params}}
        agnostic_likes = {"input": [], "output": []}
        # Go through all components.
        # NB: self.components iterates over likelihoods first, and then theories
        # so unassigned can by default go to theories
        for io_kind, option, prefix, derived_param in (
                ["input", _input_params, _input_params_prefix, False],
                ["output", _output_params, _output_params_prefix, True]):
            for component in self.components:
                if isinstance(component, AbsorbUnusedParamsLikelihood):
                    # take leftover parameters
                    continue
                if component.get_allow_agnostic():
                    supports_params = None
                elif io_kind == 'output':
                    supports_params = set(component.get_can_provide_params())
                    provide = str_to_list(getattr(component, _provides, []))
                    supports_params |= set(provide)
                else:
                    required_params = set(
                        p for p, v in ensure_dict(component.get_requirements()).items()
                        # ignore non-params; it's ok if some non-param goes through
                        if v is None)
                    supports_params = required_params.union(
                        set(component.get_can_support_params()))
                # Identify parameters understood by this likelihood/theory
                # 1a. Does it have input/output params list?
                #     (takes into account that for callables, we can ignore elements)
                if getattr(component, option, None) is not None:
                    for p in getattr(component, option):
                        try:
                            params_assign[io_kind][p] += [component]
                        except KeyError:
                            if io_kind == "input":
                                # If external function, no problem: it may have
                                # default value
                                if not isinstance(component,
                                                  LikelihoodExternalFunction):
                                    raise LoggedError(
                                        self.log,
                                        "Parameter '%s' needed as input for '%s', "
                                        "but not provided.", p, component.name)
                # 2. Is there a params prefix?
                elif getattr(component, prefix, None) is not None:
                    for p in params_assign[io_kind]:
                        if p.startswith(getattr(component, prefix)):
                            params_assign[io_kind][p] += [component]
                # 3. Does it have a general (mixed) list of params? (set from default)
                elif getattr(component, _params, None):
                    for p, options in getattr(component, _params).items():
                        if p in params_assign[io_kind]:
                            if not hasattr(options, 'get') or \
                                    options.get('derived',
                                                derived_param) is derived_param:
                                params_assign[io_kind][p] += [component]
                # 4. otherwise explicitly supported?
                elif supports_params:
                    # outputs this parameter unless explicitly told
                    # another component provides it
                    for p in supports_params:
                        if p in params_assign[io_kind]:
                            if not any((c is not component and p in
                                        str_to_list(getattr(c, _provides, []))) for c in
                                       self.components):
                                params_assign[io_kind][p] += [component]
                # 5. No parameter knowledge: store as parameter agnostic
                elif supports_params is None:
                    agnostic_likes[io_kind] += [component]
            # Check that there is only one non-knowledgeable element, and assign
            # unused params
            if (len(agnostic_likes[io_kind]) > 1 and not all(
                    params_assign[io_kind].values())):
                raise LoggedError(
                    self.log, "More than one parameter-agnostic likelihood/theory "
                              "with respect to %s parameters: %r. Cannot decide "
                              "parameter assignments.", io_kind, agnostic_likes)
            elif agnostic_likes[io_kind]:  # if there is only one
                component = agnostic_likes[io_kind][0]
                for p, assigned in params_assign[io_kind].items():
                    if not assigned or not derived_param and \
                            p in component.get_requirements():
                        params_assign[io_kind][p] += [component]
        # If unit likelihood is present, assign all unassigned inputs to it
        for like in self.likelihood.values():
            if isinstance(like, AbsorbUnusedParamsLikelihood):
                for p, assigned in params_assign["input"].items():
                    if not assigned:
                        assigned.append(like)
                break
        # If there are unassigned input params, check later that they are used by
        # *conditional* requirements of a component (and if not raise error)
        self._unassigned_input = set(p for p, assigned in params_assign["input"].items()
                                     if not assigned)
        # Remove aggregated chi2 that may have been picked up by an agnostic component
        aggr_chi2_names = [_get_chi2_name(t) for t in self.likelihood.all_types]
        for p in aggr_chi2_names:
            params_assign["output"].pop(p, None)
        # Assign the single-likelihood "chi2__" output parameters
        for p in params_assign["output"]:
            if p.startswith(_get_chi2_name("")):
                if p in aggr_chi2_names:
                    continue  # it's an aggregated likelihood
                like = p[len(_get_chi2_name("")):]
                if like not in [lik.replace(".", "_") for lik in self.likelihood]:
                    raise LoggedError(
                        self.log, "Your derived parameters depend on an unknown "
                                  "likelihood: '%s'", like)
                # They may have been already assigned to an agnostic likelihood,
                # so purge first: no "=+"
                params_assign["output"][p] = [self.likelihood[like]]
        # Check that there are no unassigned parameters (with the exception of aggr chi2)
        unassigned_output = [
            p for p, assigned in params_assign["output"].items()
            if not assigned and p not in aggr_chi2_names]
        if unassigned_output:
            raise LoggedError(
                self.log, "Could not find whom to assign output parameters %r.",
                unassigned_output)
        # Check that output parameters are assigned exactly once
        multiassigned_output = {
            p: assigned for p, assigned in params_assign["output"].items()
            if len(assigned) > 1}
        if multiassigned_output:
            raise LoggedError(
                self.log,
                "Output params can only be computed by one likelihood/theory, "
                "but some were claimed by more than one: %r.",
                multiassigned_output)
        # Finished! Assign and update infos
        for io_kind, option, attr in (
                ["input", _input_params, "input_params"],
                ["output", _output_params, "output_params"]):
            for component in self.components:
                setattr(component, attr,
                        [p for p, assign in params_assign[io_kind].items() if
                         component in assign])
                # Update infos! (helper theory parameters stored in yaml with host)
                inf = (info_theory, info_likelihood)[
                    component in self.likelihood.values()]
                inf = inf.get(component.get_name())
                if inf:
                    inf.pop(_params, None)
                    inf[option] = component.get_attr_list_with_helpers(attr)
        if self.log.getEffectiveLevel() <= logging.DEBUG:
            self.log.debug("Parameters were assigned as follows:")
            for component in self.components:
                self.log.debug("- %r:", component)
                self.log.debug("     Input:  %r", component.input_params)
                self.log.debug("     Output: %r", component.output_params)

    @property
    def components(self):
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
        different_footprints = list(set(tuple(row) for row in footprints.tolist()))
        blocks = [[p for ip, p in enumerate(self.sampled_dependence)
                   if all(footprints[ip] == fp)] for fp in different_footprints]
        # a) Multiple blocks
        if not split_fast_slow:
            i_optimal_ordering, costs, oversample_factors = sort_parameter_blocks(
                blocks, np.array(list(speeds.values()), dtype=np.float),
                different_footprints, oversample_power=oversample_power)
            blocks_sorted = [blocks[i] for i in i_optimal_ordering]
        # b) 2-block slow-fast separation
        else:
            if len(blocks) == 1:
                raise LoggedError(self.log, "Requested fast/slow separation, "
                                            "but all parameters have the same speed.")
            # First sort them optimally (w/o oversampling)
            i_optimal_ordering, costs, oversample_factors = sort_parameter_blocks(
                blocks, np.array(list(speeds.values()), dtype=np.float),
                different_footprints, oversample_power=0)
            blocks_sorted = [blocks[i] for i in i_optimal_ordering]
            footprints_sorted = np.array(different_footprints)[list(i_optimal_ordering)]
            # Then, find the split that maxes cost LOG-differences.
            # Since costs are already "accumulated down",
            # we need to subtract those below each one
            costs_per_block = costs - np.concatenate([costs[1:], [0]])
            # Split them so that "adding the next block to the slow ones" has max cost
            log_differences = np.log(costs_per_block[:-1]) - np.log(costs_per_block[1:])
            i_last_slow = np.argmax(log_differences)
            blocks_split = (lambda l: [list(chain(*l[:i_last_slow + 1])),
                                       list(chain(*l[i_last_slow + 1:]))])(blocks_sorted)
            footprints_split = (
                    [np.array(footprints_sorted[:i_last_slow + 1]).sum(axis=0)] +
                    [np.array(footprints_sorted[i_last_slow + 1:]).sum(axis=0)])
            footprints_split = np.clip(np.array(footprints_split), 0, 1)
            # Recalculate oversampling factor with 2 blocks
            _, _, oversample_factors = sort_parameter_blocks(
                blocks_split, np.array(list(speeds.values()), dtype=np.float),
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
        check = are_different_params_lists(
            list(chain(*blocks)), sampled_params)
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

    def get_auto_covmat(self, params_info=None):
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
                        params_info, self.info()[kinds.likelihood])
        except Exception as e:
            self.log.warning("Something went wrong when looking for a covmat: %r", str(e))
            return None

    def set_timing_on(self, on):
        self.timing = on
        for component in self.components:
            component.set_timing_on(on)

    def measure_and_set_speeds(self, n=None, discard=1, max_tries=np.inf):
        """
        Measures the speeds of the different components (theories and likelihoods). To do
        that it evaluates the posterior at `n` points (default: 1 per MPI process, or 3 if
        single process), discarding `discard` points (default: 1) to mitigate possible
        internal caching.

        Stops after encountering `max_tries` points (default: inf) with non-finite
        posterior.
        """
        timing_on = self.timing
        if not timing_on:
            self.set_timing_on(True)
        self.mpi_info("Measuring speeds... (this may take a few seconds)")
        if n is None:
            n = 1 if more_than_one_process() else 3
        n_done = 0
        while n_done < int(n) + int(discard):
            point = self.prior.reference(
                max_tries=max_tries, ignore_fixed=True, warn_if_no_ref=False)
            if self.loglike(point, cached=False, return_derived=False) != -np.inf:
                n_done += 1
        self.log.debug("Computed %d points to measure speeds.", n_done)
        times = [component.timer.get_time_avg() for component in self.components]
        if more_than_one_process():
            # average for different points
            times = np.average(get_mpi_comm().allgather(times), axis=0)
        measured_speeds = [1 / (1e-7 + time) for time in times]
        self.mpi_info('Setting measured speeds (per sec): %r',
                      {component: float("%.3g" % speed) for component, speed in
                       zip(self.components, measured_speeds)})
        if not timing_on:
            self.set_timing_on(False)
        for component, speed in zip(self.components, measured_speeds):
            component.set_measured_speed(speed)
