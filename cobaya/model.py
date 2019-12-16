"""
.. module:: model

:Synopsis: Wrapper for models: parameterization+prior+likelihood+theory
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import numpy as np
from collections import namedtuple, OrderedDict as odict
from itertools import chain, permutations
import logging
import six

# Local
from cobaya.conventions import kinds, _prior, _timing, _aliases
from cobaya.conventions import _params, _overhead_time, _provides
from cobaya.conventions import _path_install, _debug, _debug_default, _debug_file
from cobaya.conventions import _input_params, _output_params, _chi2, _separator
from cobaya.conventions import _input_params_prefix, _output_params_prefix, _requires
from cobaya.input import update_info
from cobaya.parameterization import Parameterization
from cobaya.prior import Prior
from cobaya.likelihood import LikelihoodCollection, LikelihoodExternalFunction, \
    AbsorbUnusedParamsLikelihood
from cobaya.theory import TheoryCollection
from cobaya.log import LoggedError, logger_setup, HasLogger
from cobaya.yaml import yaml_dump
from cobaya.tools import gcd, deepcopy_where_possible, are_different_params_lists, \
    str_to_list
from cobaya.component import Provider
from cobaya.mpi import more_than_one_process, get_mpi_comm

# Log-posterior namedtuple
LogPosterior = namedtuple("LogPosterior", ["logpost", "logpriors", "loglikes", "derived"])
LogPosterior.__new__.__defaults__ = (None, None, [], [])


class Requirement(namedtuple("Requirement", ["name", "options"])):

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
    if isinstance(d1, six.string_types):
        return d1 == d2
    if isinstance(d1, dict):
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


def get_model(info, stop_at_error=None):
    assert hasattr(info, "keys"), (
        "The first argument must be a dictionary with the info needed for the model. "
        "If you were trying to pass the name of an input file instead, "
        "load it first with 'cobaya.input.load_input', "
        "or, if you were passing a yaml string, load it with 'cobaya.yaml.yaml_load'.")
    # Configure the logger ASAP
    # TODO: Just a dummy import before configuring the logger, until I fix root/individual level
    import getdist
    info = deepcopy_where_possible(info)
    # Create the updated input information, including defaults for each module.
    logger_setup(info.pop(_debug, _debug_default), info.pop(_debug_file, None))
    info_stop = info.pop("stop_at_error", False)
    ignored_info = {}
    for k in list(info):
        if k not in [_params, kinds.likelihood, _prior, kinds.theory, _path_install,
                     _timing]:
            ignored_info.update({k: info.pop(k)})
    if ignored_info:
        logging.getLogger(__name__.split(".")[-1]).warning(
            "Ignored blocks/options: %r", list(ignored_info))
    updated_info = update_info(info)
    if logging.root.getEffectiveLevel() <= logging.DEBUG:
        logging.getLogger(__name__.split(".")[-1]).debug(
            "Input info updated with defaults (dumped to YAML):\n%s",
            yaml_dump(updated_info))
    # Initialize the parameters and posterior
    return Model(updated_info[_params], updated_info[kinds.likelihood],
                 updated_info.get(_prior), updated_info.get(kinds.theory),
                 path_install=info.get(_path_install), timing=updated_info.get(_timing),
                 stop_at_error=info_stop if stop_at_error is None else stop_at_error)


class Model(HasLogger):
    """
    Class containing all the information necessary to compute the unnormalized posterior.

    Allows for low-level interaction with the theory code, prior and likelihood.

    **NB:** do not initialize this class directly; use :func:`~model.get_model` instead,
    with some info as input.
    """

    def __init__(self, info_params, info_likelihood, info_prior=None, info_theory=None,
                 path_install=None, timing=None, allow_renames=True, stop_at_error=False,
                 post=False, prior_parameterization=None):
        self.set_logger(lowercase=True)
        self._updated_info = {
            _params: deepcopy_where_possible(info_params),
            kinds.likelihood: deepcopy_where_possible(info_likelihood)}
        if not self._updated_info[kinds.likelihood]:
            raise LoggedError(self.log, "No likelihood requested!")
        for k, v in ((_prior, info_prior), (kinds.theory, info_theory),
                     (_path_install, path_install), (_timing, timing)):
            if v not in (None, {}):
                self._updated_info[k] = deepcopy_where_possible(v)
        self.parameterization = Parameterization(self._updated_info[_params],
                                                 allow_renames=allow_renames,
                                                 ignore_unused_sampled=post)
        self.prior = Prior(prior_parameterization or self.parameterization,
                           self._updated_info.get(_prior, None))

        self.timing = timing

        info_theory = self._updated_info.get(kinds.theory)
        self.theory = TheoryCollection(info_theory, path_install=path_install,
                                       timing=timing)

        info_likelihood = self._updated_info[kinds.likelihood]
        self.likelihood = LikelihoodCollection(info_likelihood, theory=self.theory,
                                               path_install=path_install, timing=timing)

        for component in self.components:
            self.theory.update(component.get_helper_theories())

        if stop_at_error:
            for component in self.components:
                component.stop_at_error = stop_at_error

        # Assign input/output parameters
        self._assign_params(info_likelihood, info_theory)

        self._set_dependencies_and_providers()

        self._measured_speeds = None

        # Overhead per likelihood evaluation
        self.overhead = _overhead_time

    def info(self):
        """
        Returns a copy of the information used to create the model, including defaults.
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

        for (component, index), dependence in zip(self._component_order.items(),
                                                  self._ordered_param_dependence):

            depend_list = [input_params[p] for p in dependence]
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
                loglikes[index - n_theory] = component.get_current_logp()
                if return_derived:
                    derived_dict[_chi2 + _separator +
                                 component.get_name().replace(".", "_")] \
                        = -2 * loglikes[index - n_theory]
                    for this_type in getattr(component, "type", []):
                        aggr_chi2_name = _chi2 + _separator + this_type
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
        if hasattr(params_values, "keys") and not _no_check:
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

    def get_version(self):
        return dict(theory=self.theory.get_version(),
                    likelihood=self.likelihood.get_version())

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

        self._component_order = odict(zip(dependence_order, (components.index(c) for c in
                                                             dependence_order)))

    def _set_dependencies_and_providers(self):
        requirements = []
        dependencies = {}
        self._needs = {}
        providers = {}
        components = list(self.theory.values()) + list(self.likelihood.values())
        direct_param_dependence = {c: set() for c in components}

        def _tidy_requirements(_component, _require):
            # take input requirement dictionary and split into list of tuples of
            # requirement names and requirement options
            if not _require:
                return []
            if isinstance(_require, (set, tuple, list)):
                _require = {p: None for p in _require}
            else:
                _require = _require.copy()
            for par in self.input_params:
                if par in _require:
                    direct_param_dependence[_component].add(par)
                    # requirements that are sampled parameters automatically satisfied
                    _require.pop(par, None)

            return [Requirement(p, v) for p, v in _require.items()]

        # Get the requirements and providers
        for component in components:
            if hasattr(component, "add_theory"):
                raise LoggedError(self.log,
                                  "Please remove add_theory from %r and return "
                                  "requirement dictionary from get_requirements() "
                                  "instead" % component)
            self._needs[component] = []
            component.initialize_with_params()
            requirements.append(
                _tidy_requirements(component, component.get_requirements()))

            methods = component.get_can_provide_methods()
            can_provide = list(component.get_can_provide()) + list(methods)
            # parameters that can be provided but not already explicitly assigned
            provide_params = [p for p in component.get_can_provide_params() if
                              p not in self.output_params and p not in
                              str_to_list(getattr(component, _requires, []))]

            for k in can_provide + component.output_params + provide_params:
                providers[k] = providers.get(k, []) + [component]

        requirement_providers = {}
        has_more_requirements = True
        while has_more_requirements:
            # list of dictionary of needs for each component
            needs = {c: [] for c in components}
            # Check supplier for each requirement, get dependency and needs
            for component, requires in zip(components, requirements):
                for requirement in requires:
                    suppliers = providers.get(requirement.name)
                    if suppliers:
                        supplier = None
                        if len(suppliers) > 1:
                            for sup in suppliers:
                                provide = str_to_list(getattr(sup, 'provides', []))
                                if requirement.name in provide:
                                    if supplier:
                                        raise LoggedError(self.log, "more than one "
                                                                    "component provides"
                                                                    " %s" %
                                                          requirement.name)
                                    supplier = sup
                            if not supplier:
                                raise LoggedError(self.log,
                                                  "requirement %s is provided by more "
                                                  "than one component: %s. Use 'provides'"
                                                  " keyword to specify which provides it "
                                                  % (requirement.name, suppliers))
                        else:
                            supplier = suppliers[0]
                        if supplier is component:
                            raise LoggedError(self.log,
                                              "Component %r cannot provide %s to "
                                              "itself!" % (component, requirement.name))
                        requirement_providers[requirement.name] = supplier.get_provider()
                        if requirement not in self._needs[supplier] + needs[supplier]:
                            needs[supplier] += [requirement]
                        dependencies[component] = \
                            dependencies.get(component, set()) | {supplier}
                    else:
                        raise LoggedError(self.log,
                                          "Requirement %s of %r is not provided by any "
                                          "component" % (requirement.name, component))

            # tell each component what it needs to calculate, and collect the
            # conditional requirements for those needs
            has_more_requirements = False
            for component, requires in zip(components, requirements):
                requires[:] = []
                if needs[component]:
                    for need in needs[component]:
                        conditional_requirements = \
                            _tidy_requirements(component,
                                               component.needs(
                                                   **{need.name: need.options}))
                        self._needs[component].append(need)
                        if conditional_requirements:
                            has_more_requirements = True
                            requires += conditional_requirements
            # set component compute order and raise error if circular dependence
            self._set_component_order(components, dependencies)

        # always call needs at least once (#TODO is this needed? e.g. for post)
        for component in components:
            if not self._needs[component]:
                component.needs()

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

        self._dependencies = {c: dependencies_of(c) for c in components}

        self._ordered_param_dependence = [set() for _ in components]
        for component, param_dep in zip(self._component_order,
                                        self._ordered_param_dependence):
            param_dep.update(direct_param_dependence.get(component))
            for dep in self._dependencies.get(component, []):
                param_dep.update(set(dep.input_params))
            param_dep -= set(component.input_params)
            if not len(component.input_params) and not param_dep \
                    and component.get_name() != 'one':
                raise LoggedError(self.log, "Component '%r' seems not to depend on any "
                                            "parameters (neither directly nor "
                                            "indirectly)", component)

        # Store the input params and components on which each sampled params depends.
        sampled_input_dependence = self.parameterization.sampled_input_dependence()
        sampled_dependence = odict((p, []) for p in sampled_input_dependence)
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

        if self.log.getEffectiveLevel() <= logging.DEBUG:
            self.log.debug("Requirements will be calculated by these components:")
            for requirement, provider in requirement_providers.items():
                self.log.debug("- %s: %s" % (requirement, provider))

        self.provider = Provider(self, requirement_providers)

        for component in components:
            component.initialize_with_provider(self.provider)

    def requested(self):
        """
        Get all the requested requirements that will be computed.

        :return: dictionary giving list of requirements calculated by
                each component name
        """
        return dict(("%r" % c, v) for c, v in self._needs.items() if v)

    def _assign_params(self, info_likelihood, info_theory=None):
        """
        Assign parameters to theories and likelihoods, following the algorithm explained
        in :doc:`DEVEL`.
        """
        self.input_params = list(self.parameterization.input_params())
        self.output_params = list(self.parameterization.output_params())
        params_assign = odict([
            ("input", odict((p, []) for p in self.input_params)),
            ("output", odict((p, []) for p in self.output_params))])
        agnostic_likes = {"input": [], "output": []}
        # All components, doing likelihoods first so unassigned can by default
        # go to theory
        components = self.components
        for kind, option, prefix, derived_param in (
                ["input", _input_params, _input_params_prefix, False],
                ["output", _output_params, _output_params_prefix, True]):
            for component in components:
                if isinstance(component, AbsorbUnusedParamsLikelihood):
                    # take leftover parameters
                    continue
                if component.get_allow_agnostic():
                    supports_params = None
                elif kind == 'output':
                    supports_params = set(component.get_can_provide_params())
                    provide = str_to_list(getattr(component, _provides, []))
                    supports_params |= set(provide)
                else:
                    supports_params = set(list(component.get_requirements())).union(
                        set(component.get_can_support_params()))

                # Identify parameters understood by this likelihood/theory
                # 1a. Does it have input/output params list?
                #     (takes into account that for callables, we can ignore elements)
                if getattr(component, option, None) is not None:
                    for p in getattr(component, option):
                        try:
                            params_assign[kind][p] += [component]
                        except KeyError:
                            if kind == "input":
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
                    for p in params_assign[kind]:
                        if p.startswith(getattr(component, prefix)):
                            params_assign[kind][p] += [component]
                # 3. Does it have a general (mixed) list of params? (set from default)
                elif getattr(component, _params, None):
                    for p, options in getattr(component, _params).items():
                        if p in params_assign[kind]:
                            if not hasattr(options, 'get') or \
                                    options.get('derived',
                                                derived_param) is derived_param:
                                params_assign[kind][p] += [component]
                # 4. otherwise explicitly supported?
                elif supports_params:
                    # outputs this parameter unless explicitly told
                    # another component provides it
                    for p in supports_params:
                        if p in params_assign[kind]:
                            if not any((c is not component and p in
                                        str_to_list(getattr(c, _provides, []))) for c in
                                       components):
                                params_assign[kind][p] += [component]
                # 5. No parameter knowledge: store as parameter agnostic
                elif supports_params is None:
                    agnostic_likes[kind] += [component]

            # Check that there is only one non-knowledgeable element, and assign
            # unused params
            if len(agnostic_likes[kind]) > 1 and not all(params_assign[kind].values()):
                raise LoggedError(
                    self.log, "More than one parameter-agnostic likelihood/theory "
                              "with respect to %s parameters: %r. Cannot decide "
                              "parameter assignments.", kind, agnostic_likes)
            elif agnostic_likes[kind]:  # if there is only one
                component = agnostic_likes[kind][0]
                for p, assigned in params_assign[kind].items():
                    if not assigned or not derived_param and \
                            p in getattr(component, _requires, []):
                        params_assign[kind][p] += [component]

        # If unit likelihood is present, assign all unassigned inputs to it
        for like in self.likelihood.values():
            if isinstance(like, AbsorbUnusedParamsLikelihood):
                for p, assigned in params_assign["input"].items():
                    if not assigned:
                        assigned.append(like)
                break
        # If there are unassigned input params, check later that they are used by
        # requirements of a component (and if not raise error)
        self._unassigned_input = set(p for p, assigned in params_assign["input"].items()
                                     if not assigned)

        # Assign the "chi2__" output parameters
        for p in params_assign["output"]:
            if p.startswith(_chi2 + _separator):
                like = p[len(_chi2 + _separator):]
                if like not in [l.replace(".", "_") for l in self.likelihood]:
                    raise LoggedError(
                        self.log, "Your derived parameters depend on an unknown "
                                  "likelihood: '%s'", like)
                # They may have been already assigned to an agnostic likelihood,
                # so purge first: no "=+"
                params_assign["output"][p] = [self.likelihood[like]]
        # Check that output parameters are assigned exactly once
        unassigned_output = [
            p for p, assigned in params_assign["output"].items() if not assigned]
        multiassigned_output = {
            p: assigned for p, assigned in params_assign["output"].items()
            if len(assigned) > 1}
        if unassigned_output:
            raise LoggedError(
                self.log, "Could not find whom to assign output parameters %r.",
                unassigned_output)
        if multiassigned_output:
            raise LoggedError(
                self.log,
                "Output params can only be computed by one likelihood/theory, "
                "but some were claimed by more than one: %r.",
                multiassigned_output)
        # Finished! Assign and update infos
        for kind, option, attr in (
                ["input", _input_params, "input_params"],
                ["output", _output_params, "output_params"]):
            for component in components:
                setattr(component, attr,
                        [p for p, assign in params_assign[kind].items() if
                         component in assign])
                # Update infos!
                inf = (info_theory, info_likelihood)[
                    component in self.likelihood.values()]
                inf = inf.get(component.get_name())
                if inf:
                    inf.pop(_params, None)
                    inf[option] = getattr(component, attr)
        if self.log.getEffectiveLevel() <= logging.DEBUG:
            self.log.debug("Parameters were assigned as follows:")
            for component in components:
                self.log.debug("- %r:", component)
                self.log.debug("     Input:  %r", component.input_params)
                self.log.debug("     Output: %r", component.output_params)

    @property
    def components(self):
        return list(chain(self.likelihood.values(), self.theory.values()))

    def _speeds_of_params(self, int_speeds=False):
        """
        Separates the sampled parameters in blocks according to the likelihood (or theory)
        re-evaluation that changing each one of them involves. Using the approximate speed
        (i.e. inverse evaluation time in seconds) of each likelihood, sorts the blocks in
        an optimal way, in ascending order of speed *per full block iteration*.

        Returns tuples of ``(speeds), (params_in_block)``, sorted by ascending speeds,
        where speeds are *per param* (though optimal blocking is chosen by speed
        *per full block*).

        If ``int_speeds=True``, returns integer speeds, instead of speeds in 1/s.

        """
        # Fill unknown speeds with the value of the slowest one`, and clip with overhead
        components = self.components
        speeds = np.array([component.get_speed() for component in components],
                          dtype=np.float64)
        # Add overhead to the defined ones, and clip to the slowest the undefined ones
        speeds[speeds > 0] = (speeds[speeds > 0] ** -1 + self.overhead) ** -1
        try:
            speeds = np.clip(speeds, min(speeds[speeds > 0]), None)
        except ValueError:
            # No speeds specified
            speeds = np.ones(len(speeds))
        for i, component in enumerate(components):
            component.speed = speeds[i]
        # Compute "footprint"
        # i.e. likelihoods (and theory) that we must recompute when each parameter changes
        footprints = np.zeros((len(self.sampled_dependence), len(components)), dtype=int)
        for i, deps in enumerate(self.sampled_dependence.values()):
            for j, component in enumerate(components):
                footprints[i, j] = component in deps
        # Group parameters by footprint
        different_footprints = list(set(tuple(row) for row in footprints.tolist()))
        blocks = [[p for ip, p in enumerate(self.sampled_dependence)
                   if all(footprints[ip] == fp)] for fp in different_footprints]
        # Find optimal ordering, such that one minimises the time it takes to vary every
        # parameter, one by one, in a basis in which they are mixed-down (i.e after a
        # Cholesky transformation)
        # To do that, compute that "total cost" for every permutation of the blocks order,
        # and find the minimum.
        n_params_per_block = np.array([len(b) for b in blocks])
        self._costs = 1 / np.array(speeds)
        self._footprints = np.array(different_footprints)
        self._lower = np.tri(len(n_params_per_block))

        def get_cost_per_param_per_block(ordering):
            """
            Computes cumulative cost per parameter for each block, given ordering.
            """
            footprints_chol = np.minimum(
                1, self._footprints[ordering].T.dot(self._lower).T)
            return footprints_chol.dot(self._costs)

        orderings = list(permutations(np.arange(len(n_params_per_block))))
        costs_per_param_per_block = np.array(
            [get_cost_per_param_per_block(list(o)) for o in orderings])
        total_costs = np.array(
            [n_params_per_block[list(o)].dot(costs_per_param_per_block[i])
             for i, o in enumerate(orderings)])
        i_optimal = np.argmin(total_costs)
        optimal_ordering = orderings[i_optimal]
        blocks = [blocks[i] for i in optimal_ordering]
        costs_per_param_per_block = costs_per_param_per_block[i_optimal]
        # This costs are *cumulative-down* (i.e. take into account the cost of varying the
        # parameters below the present one). Subtract that effect so that its inverse,
        # the speeds, are equivalent to oversampling factors
        costs_per_param_per_block[:-1] -= costs_per_param_per_block[1:]
        params_speeds = 1 / costs_per_param_per_block
        if int_speeds:
            # Oversampling precision: smallest difference in oversampling to be ignored.
            speed_precision = 1 / 10
            speeds = np.array(np.round(np.array(
                params_speeds) / min(params_speeds) / speed_precision), dtype=int)
            params_speeds = np.array(
                speeds / np.ufunc.reduce(np.frompyfunc(gcd, 2, 1), speeds), dtype=int)
        self.log.debug("Optimal ordering of parameter blocks: %r with speeds %r",
                       blocks, params_speeds)
        return params_speeds, blocks

    def _check_speeds_of_params(self, blocking):
        """
        Checks the correct formatting of the given parameter blocking.

        `blocking` is a list of tuples `(speed, (param1, param2, etc))`.

        Returns tuples of ``(speeds), (params_in_block)``.
        """
        try:
            speeds, blocks = zip(*list(blocking))
            speeds = np.array(speeds)
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
        if (speeds != np.sort(speeds)).all():
            self.log.warning(
                "Manual blocking: speed-blocking *apparently* non-optimal: "
                "sort by ascending speed when possible")
        return speeds, blocks

    def set_cache_size(self, n_states):
        """
        Sets the number of different parameter points to cache for all theories
        and likelihood.

        :param n_states: number of cached points
        """
        for theory in self.components:
            theory.set_cache_size(n_states)

    def get_auto_covmat(self, params_info):
        """
        Tries to get an automatic covariance matrix for the current model and data.

        ``params_info`` should contain preferably the slow parameters only.
        """
        likes_renames = {like.get_name(): {_aliases: getattr(like, _aliases, [])}
                         for like in self.likelihood.values()}
        try:
            for theory in self.theory.values():
                if hasattr(theory, 'get_auto_covmat'):
                    return theory.get_auto_covmat(params_info, likes_renames)
        except:
            return None

    def set_timing_on(self, on):
        self.timing = on
        for component in self.components:
            component.set_timing_on(on)

    def set_measured_speeds(self, test_point, speeds=None):
        if not speeds:
            timing_on = self.timing
            if not timing_on:
                self.set_timing_on(True)
            self.log.debug("measuring speeds")
            # call all components (at least) a second time
            test_point *= 1.00001
            self.loglikes(test_point, cached=False)
            times = [component.timer.get_time_avg() for component in self.components]
            if more_than_one_process():
                # average for different points
                times = np.average(get_mpi_comm().allgather(times), axis=0)
            self._measured_speeds = [1 / (1e-7 + time) for time in times]
            self.mpi_info('Setting measured speeds (per sec): %r',
                          {component: float("%.3g" % speed) for component, speed in
                           zip(self.components, self._measured_speeds)})
            if not timing_on:
                self.set_timing_on(False)
        else:
            self._measured_speeds = speeds
        for component, speed in zip(self.components, self._measured_speeds):
            component.set_measured_speed(speed)
        return self._measured_speeds
