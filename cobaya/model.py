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

# Local
from cobaya.conventions import kinds, _prior, _timing, _p_renames
from cobaya.conventions import _params, _overhead_per_param
from cobaya.conventions import _path_install, _debug, _debug_default, _debug_file
from cobaya.conventions import _input_params, _output_params, _chi2, _separator
from cobaya.conventions import _input_params_prefix, _output_params_prefix
from cobaya.input import update_info
from cobaya.parameterization import Parameterization
from cobaya.prior import Prior
from cobaya.likelihood import Likelihood, LikelihoodCollection, LikelihoodExternalFunction
from cobaya.theory import Theory, TheoryCollection
from cobaya.log import LoggedError, logger_setup, HasLogger
from cobaya.yaml import yaml_dump
from cobaya.tools import gcd, deepcopy_where_possible, are_different_params_lists

# Log-posterior namedtuple
logposterior = namedtuple("logposterior", ["logpost", "logpriors", "loglikes", "derived"])
logposterior.__new__.__defaults__ = (None, None, [], [])


def get_model(info):
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
    ignored_info = {}
    for k in list(info):
        if k not in [_params, kinds.likelihood, _prior, kinds.theory, _path_install,
                     _timing]:
            ignored_info.update({k: info.pop(k)})
    import logging
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
                 path_install=info.get(_path_install), timing=updated_info.get(_timing))


class Model(HasLogger):
    """
    Class containing all the information necessary to compute the unnormalized posterior.

    Allows for low-level interaction with the theory code, prior and likelihood.

    **NB:** do not initialize this class directly; use :func:`~model.get_model` instead,
    with some info as input.
    """

    def __init__(self, info_params, info_likelihood, info_prior=None, info_theory=None,
                 path_install=None, timing=None, allow_renames=True,
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

        # TODO: would be more logical called self.theories and self.likelihoods
        info_theory = self._updated_info.get(kinds.theory)
        self.theory = TheoryCollection(info_theory, path_install=path_install,
                                       timing=timing)

        info_likelihood = self._updated_info[kinds.likelihood]
        self.likelihood = LikelihoodCollection(info_likelihood, theory=self.theory,
                                               path_install=path_install, timing=timing)

        # Assign input/output parameters
        self._assign_params(info_likelihood, info_theory)
        # Do the user-defined post-initialisation, and assign the theory code
        for theory in self.theory.values():
            theory.initialize()
        for like in self.likelihood.values():
            like.initialize()
            like.theory = self.theory
            like.add_theory()
        for name, theory in self.theory.items():
            if post:
                # Make sure that theory.needs is called at least once, for adjustments
                theory.needs({})
            if getattr(theory, "_needs", None):
                self.log.info(
                    "The theory %s will compute the following products, "
                    "requested by the likelihoods: %r" % (name, list(theory._needs)))

        # Store the input params and components on which each sampled params depends.
        sampled_input_dependence = self.parameterization.sampled_input_dependence()
        sampled_dependence = odict()
        for p, i_s in sampled_input_dependence.items():
            sampled_dependence[p] = [component for component in self.theory.values()
                                     if any(
                    [(i in component.input_params) for i in (i_s or [p])])]
            # For the moment theory parameters "depend" on every likelihood, since
            # re-computing the theory code forces recomputation of the likelihoods
            if sampled_dependence[p]:
                sampled_dependence[p] += list(self.likelihood.values())
            else:
                sampled_dependence[p] = \
                    [component for component in self.likelihood.values() if any(
                        [(i in component.input_params) for i in (i_s or [p])])]

        self.sampled_dependence = sampled_dependence

        # Overhead per likelihood evaluation
        self.overhead = _overhead_per_param * len(self.parameterization.sampled_params())

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
            params_values = self.parameterization._check_sampled(**params_values)
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
        # Calculate required theory results and returns likelihoods
        theory_params_dict = {}
        theory_success = True
        derived_dict = {}
        this_derived_dict = {} if return_derived else None
        for theory in self.theory.values():
            theory_params = {p: input_params[p] for p in theory.input_params}
            theory_success = theory.compute(_derived=this_derived_dict, cached=cached,
                                            **theory_params)
            if not theory_success:
                self.log.debug(
                    "Theory code computation failed. Not computing likelihood.")
                break
            theory_params_dict.update(theory_params)
            if this_derived_dict:
                derived_dict.update(this_derived_dict)
                this_derived_dict.clear()

        if not theory_success:
            loglikes = np.array([-np.inf for _ in self.likelihood])
        else:
            loglikes = self.likelihood.logps(input_params, theory_params_dict,
                                             derived_dict=derived_dict, cached=cached)
        if make_finite:
            loglikes = np.nan_to_num(loglikes)

        if return_derived:
            # Turn the derived params dict into a list and return
            if not theory_success:
                derived_list = [np.nan] * len(self.output_params)
            else:
                derived_list = [derived_dict[p] for p in self.output_params]
            return loglikes, derived_list

        return loglikes

    def loglikes(self, params_values, return_derived=True, make_finite=False, cached=True,
                 _no_check=False):
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
        if hasattr(params_values, "keys") and not _no_check:
            params_values = self.parameterization._check_sampled(**params_values)

        input_params = self.parameterization._to_input(params_values)

        result = self.logps(input_params, return_derived=return_derived,
                            cached=cached, make_finite=make_finite)
        if return_derived:
            loglikes, derived_list = result
            derived_sampler = self.parameterization._to_derived(derived_list)
            if self.log.getEffectiveLevel() <= logging.DEBUG:
                self.log.debug(
                    "Computed derived parameters: %s",
                    dict(zip(self.parameterization.derived_params(), derived_sampler)))
            return loglikes, derived_sampler
        return result

    def loglike(self, params_values, return_derived=True, make_finite=False, cached=True):
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
            params_values = self.parameterization._check_sampled(**params_values)
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
        logpriors = self.logpriors(params_values_array, make_finite=False)
        logpost = sum(logpriors)
        if -np.inf not in logpriors:
            l = self.loglikes(params_values, return_derived=return_derived,
                              make_finite=make_finite, cached=cached, _no_check=True)
            loglikes, derived_sampler = l if return_derived else (l, [])
            logpost += sum(loglikes)
        else:
            loglikes = []
            derived_sampler = []
        if make_finite:
            logpriors = np.nan_to_num(logpriors)
            logpost = np.nan_to_num(logpost)
        return logposterior(logpost=logpost, logpriors=logpriors,
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

    def _assign_params(self, info_likelihood, info_theory=None):
        """
        Assign parameters to theories and likelihoods, following the algorithm explained
        in :doc:`DEVEL`.
        """
        self.input_params = list(self.parameterization.input_params())
        self.output_params = list(self.parameterization.output_params())
        params_assign = odict([
            ("input", odict([(p, []) for p in self.input_params])),
            ("output", odict([(p, []) for p in self.output_params]))])
        agnostic_likes = {"input": [], "output": []}
        # All components, doing likelihoods first so unassigned can by default
        # go to theory
        components = list(self.likelihood.values()) + list(self.theory.values())
        for kind, option, prefix in (
                ["input", _input_params, _input_params_prefix],
                ["output", _output_params, _output_params_prefix]):
            for component in components:
                # "one" only takes leftover parameters
                if component.get_name() == "one":
                    continue
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
                                if not isinstance(component, LikelihoodExternalFunction):
                                    raise LoggedError(
                                        self.log,
                                        "Parameter '%s' needed as input for '%s', "
                                        "but not provided.", p, component.name)
                # 2. Is there a params prefix?
                elif getattr(component, prefix, None) is not None:
                    for p in params_assign[kind]:
                        if p.startswith(getattr(component, prefix)):
                            params_assign[kind][p] += [component]
                # 3. Does it have a general (mixed) list of params?
                elif getattr(component, _params, None) is not None:
                    for p in getattr(component, _params):
                        if p in params_assign[kind]:
                            params_assign[kind][p] += [component]
                # 4. No parameter knowledge: store as parameter agnostic
                else:
                    agnostic_likes[kind] += [component]
                # Check that there is only one non-knowledgeable element, and assign
                # unused params
                if len(agnostic_likes[kind]) > 1:
                    raise LoggedError(
                        self.log, "More than once parameter-agnostic likelihood/theory "
                                  "with respect to %s parameters: %r. Cannot decide "
                                  "parameter assignments.", kind, agnostic_likes)
                elif agnostic_likes[kind]:  # if there is only one
                    for p, assigned in params_assign[kind].items():
                        if not assigned:
                            params_assign[kind][p] = [agnostic_likes[kind][0]]
        # Check that, if a theory code is present, it does not share input parameters with
        # any likelihood (because of the theory+experimental model separation)
        # TODO: could relax this?
        if self.theory:
            for p, assigned in params_assign["input"].items():
                if len(assigned) > 1 and [component for component in assigned if
                                          isinstance(component, Theory)] \
                        and [component for component in assigned if
                             isinstance(component, Likelihood)]:
                    raise LoggedError(
                        self.log, "Some parameter has been assigned to the theory code "
                                  "AND a likelihood, and that is not allowed: {%s: %r}",
                        p, [component.get_name() for component in assigned])
        # If unit likelihood is present, assign all *non-theory* inputs to it
        if "one" in self.likelihood:
            for p, assigned in params_assign["input"].items():
                if not [component for component in assigned if
                        isinstance(component, Theory)]:
                    assigned.append(self.likelihood["one"])
        # If there are unassigned input params, fail
        unassigned_input = [
            p for p, assigned in params_assign["input"].items() if not assigned]
        if unassigned_input:
            raise LoggedError(
                self.log, "Could not find anything to use input parameter(s) %r.",
                unassigned_input)
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
                self.log, "Output params can only be computed by one likelihood/theory, "
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
                if isinstance(component, Likelihood):
                    info_likelihood[component.get_name()].pop(_params, None)
                    info_likelihood[component.get_name()][option] = getattr(component,
                                                                            attr)
                else:
                    info_theory[component.get_name()].pop(_params, None)
                    info_theory[component.get_name()][option] = getattr(component, attr)
        if self.log.getEffectiveLevel() <= logging.DEBUG:
            self.log.debug("Parameters were assigned as follows:")
            for component in components:
                self.log.debug("- %r:", component)
                self.log.debug("     Input:  %r", component.input_params)
                self.log.debug("     Output: %r", component.output_params)

    def _speeds_of_params(self, int_speeds=False, fast_slow=False):
        """
        Separates the sampled parameters in blocks according to the likelihood (or theory)
        re-evaluation that changing each one of them involves. Using the approximate speed
        (i.e. inverse evaluation time in seconds) of each likelihood, sorts the blocks in
        an optimal way, in ascending order of speed *per full block iteration*.

        Returns tuples of ``(speeds), (params_in_block)``, sorted by ascending speeds,
        where speeds are *per param* (though optimal blocking is chosen by speed
        *per full block*).

        If ``int_speeds=True``, returns integer speeds, instead of speeds in 1/s.

        If ``fast_slow=True``, returns just 2 blocks: a fast and a slow one, each one
        assigned its slowest per-parameter speed.
        """
        # Fill unknown speeds with the value of the slowest one, and clip with overhead
        components = list(self.likelihood.values()) + list(self.theory.values())
        speeds = np.array([getattr(component, "speed", -1) for component in components],
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
        for i, ls in enumerate(self.sampled_dependence.values()):
            for j, like in enumerate(components):
                footprints[i, j] = like in ls
        # Group parameters by footprint
        different_footprints = list(set([tuple(row) for row in footprints.tolist()]))
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
        # Fast-slow separation: chooses separation that maximizes log-difference in speed
        # (speed per parameter in a combination of blocks is the slowest one)
        if fast_slow:
            if len(blocks) > 1:
                log_differences = np.zeros(len(blocks) - 1)
                for i in range(len(blocks) - 1):
                    log_differences[i] = (np.log(np.min(params_speeds[:i + 1])) -
                                          np.log(np.min(params_speeds[i + 1:])))
                i_max = np.argmin(log_differences)
                blocks = (
                    lambda l: [list(chain(*l[:i_max + 1])), list(chain(*l[i_max + 1:]))])(
                    blocks)
                # In this case, speeds must be *cumulative*, since I am squashing blocks
                cum_inv = lambda ss: 1 / (sum(1 / ss))
                params_speeds = (
                    lambda l: [cum_inv(l[:i_max + 1]), cum_inv(l[i_max + 1:])])(
                    params_speeds)
                self.log.debug("Fast-slow blocking: %r with speeds %r",
                               blocks, params_speeds)
            else:
                self.log.warning("Requested fast/slow separation, "
                                 "but all parameters have the same speed.")
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
                self.log, "Manual blocking: unkown parameters: %r", unknown)
        if (speeds != np.sort(speeds)).all():
            self.log.warning("Manual blocking: speed-blocking *apparently* non-optimal: "
                             "sort by ascending speed when possible")
        return speeds, blocks

    def _get_auto_covmat(self, params_info):
        """
        Tries to get an automatic covariance matrix for the current model and data.

        ``params_info`` should contain preferably the slow parameters only.
        """
        likes_renames = {like: {_p_renames: getattr(like, _p_renames, [])}
                         for like in self.likelihood}
        try:
            # TODO: get_auto_covmat has nothing to do with cosmology, move to model?
            return self.theory.values[0].get_auto_covmat(params_info, likes_renames)
        except:
            return None
