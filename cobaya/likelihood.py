"""
.. module:: likelihood

:Synopsis: Prototype likelihood and likelihood manager
:Author: Jesus Torrado

This module defines the main :class:`Likelihood` class, from which every likelihood
inherits, and the :class:`LikelihoodCollection` class, which groups and manages all the
individual likelihoods and is the actual instance passed to the sampler.

"""
# Python 2/3 compatibility
from __future__ import absolute_import, division

# Global
import sys
import traceback
from collections import OrderedDict as odict
from time import time, sleep
import numpy as np
from itertools import chain, permutations
import six
from copy import deepcopy

if six.PY3:
    from math import gcd
else:
    from fractions import gcd

# Local
from cobaya.conventions import _external, _theory, _params, _overhead_per_param
from cobaya.conventions import _timing, _p_renames, _chi2, _separator
from cobaya.conventions import _input_params, _output_params
from cobaya.conventions import _input_params_prefix, _output_params_prefix
from cobaya.tools import get_class, get_external_function, getargspec
from cobaya.tools import compare_params_lists
from cobaya.log import HandledException

# Logger
import logging

# Default options for all subclasses
class_options = {"speed": -1}


class Likelihood(object):
    """Likelihood class prototype."""

    # Generic initialization -- do not touch
    def __init__(self, info, modules=None, timing=None):
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        self.path_install = modules
        # Load info of the likelihood
        for k in info:
            setattr(self, k, info[k])
        # States, to avoid recomputing
        self.n_states = 3
        self.states = [{"params": None, "logp": None, "_derived": None,
                        "theory_params": None, "last": 0}
                       for _ in range(self.n_states)]
        # Timing
        self.timing = timing
        self.n = 0
        self.time_avg = 0

    # What you *must* implement to create your own likelihood:

    # Optional
    def initialize(self):
        """
        Initializes the specifics of this likelihood.
        Note that at this point we know `the `self.input_params``
        and the ``self.output_params``.
        """
        pass

    # Optional
    def add_theory(self):
        """Performs any necessary initialization on the theory side,
        e.g. requests observables"""
        pass

    # Mandatory
    def logp(self, **params_values):
        """
        Computes and returns the log likelihood value.
        Takes as keyword arguments the parameter values.
        To get the derived parameters, pass a `_derived` keyword with an empty dictionary.
        """
        return None

    # Optional
    def close(self):
        """Finalizes the likelihood, if something needs to be done."""
        pass

    # What you *can* implement to create your own likelihood:

    def marginal(self, directions=None, params_values=None):
        """
        (For analytic likelihoods only.)
        Computes the marginal likelihood.
        If nothing is specified, returns the total marginal likelihood.
        If some directions are specified (as a list, tuple or array), returns the marginal
        likelihood pdf over those directions evaluated at the given parameter values.
        """
        self.log.error("Exact marginal likelihood not defined.")
        raise HandledException

    # Other general methods

    def _logp_cached(self, theory_params=None, cached=True, _derived=None, **params_values):
        """
        Wrapper for the `logp` method that caches logp's and derived params.
        If the theory products have been re-computed, re-computes the likelihood anyway.
        """
        params_values = deepcopy(params_values)
        self.log.debug("Got parameters %r", params_values)
        lasts = [self.states[i]["last"] for i in range(self.n_states)]
        try:
            if not cached:
                raise StopIteration
            # Are the parameter values there already?
            i_state = next(i for i in range(self.n_states)
                           if self.states[i]["params"] == params_values)
            # StopIteration not raised, so state exists, but maybe the theory params have
            # changed? In that case, I would still have to re-compute the likelihood
            if self.states[i_state]["theory_params"] != theory_params:
                self.log.debug("Recomputing logp because theory params changed.")
                raise StopIteration
            if _derived is not None:
                _derived.update(self.states[i_state]["derived"] or {})
            self.log.debug("Re-using computed results.")
        except StopIteration:
            # update the (first) oldest one and compute
            i_state = lasts.index(min(lasts))
            self.states[i_state]["params"] = params_values
            self.states[i_state]["theory_params"] = deepcopy(theory_params)
            if self.timing:
                start = time()
            self.states[i_state]["logp"] = self.logp(_derived=_derived, **params_values)
            if self.timing:
                self.n += 1
                self.time_avg = (time() - start + self.time_avg * (self.n - 1)) / self.n
                self.log.debug("Average 'logp' evaluation time: %g s", self.time_avg)
            self.states[i_state]["derived"] = deepcopy(_derived)
        # make this one the current one by decreasing the antiquity of the rest
        for i in range(self.n_states):
            self.states[i]["last"] -= max(lasts)
        self.states[i_state]["last"] = 1
        self.log.debug("Evaluated to logp=%g with derived %r",
                       self.states[i_state]["logp"], self.states[i_state]["derived"])
        return self.states[i_state]["logp"]

    def wait(self):
        if self.delay:
            log.debug("Sleeping for %f seconds.", self.delay)
        sleep(self.delay)

    def d(self):
        return len(self.input_params)

    def __exit__(self, exception_type, exception_value, traceback):
        if self.timing:
            self.log.info("Average 'logp' evaluation time: %g s  (%d evaluations)" %
                          (self.time_avg, self.n))
        self.close()


class LikelihoodExternalFunction(Likelihood):
    def __init__(self, name, info, _theory=None, timing=None):
        self.name = name
        self.log = logging.getLogger(self.name)
        # Load info of the likelihood
        for k in info:
            setattr(self, k, info[k])
        # Store the external function and its arguments
        self.external_function = get_external_function(info[_external], name=self.name)
        argspec = getargspec(self.external_function)
        self.input_params = [p for p in argspec.args if p not in ["_derived", "_theory"]]
        self.has_derived = "_derived" in argspec.args
        if self.has_derived:
            derived_kw_index = argspec.args[-len(argspec.defaults):].index("_derived")
            self.output_params = argspec.defaults[derived_kw_index]
        else:
            self.output_params = []
        self.has_theory = "_theory" in argspec.args
        if self.has_theory:
            theory_kw_index = argspec.args[-len(argspec.defaults):].index("_theory")
            self.needs = argspec.defaults[theory_kw_index]
        # Timing
        self.timing = timing
        self.n = 0
        self.time_avg = 0
        # States, to avoid recomputing
        self.n_states = 3
        self.states = [{"params": None, "logp": None, "derived": None, "last": 0}
                       for _ in range(self.n_states)]

    def add_theory(self):
        if self.has_theory:
            self.theory.needs(**self.needs)

    def logp(self, **params_values):
        # if no derived params defined in external func, delete the "_derived" argument
        if not self.has_derived:
            params_values.pop("_derived")
        if self.has_theory:
            params_values["_theory"] = self.theory
        try:
            return self.external_function(**params_values)
        except:
            self.log.error("".join(
                ["-"] * 16 + ["\n\n"] + list(traceback.format_exception(*sys.exc_info())) +
                ["\n"] + ["-"] * 37))
            self.log.error("The external likelihood '%s' failed at evaluation. "
                      "See traceback on top of this message.", self.name)
            raise HandledException


class LikelihoodCollection(object):
    """
    Likelihood manager:
    Initializes the theory code and the experimental likelihoods.
    """

    def __init__(self, info_likelihood, parameterization, info_theory=None, modules=None,
                 timing=None):
        self.log = logging.getLogger("Likelihood")
        # *IF* there is a theory code, initialize it
        if info_theory:
            name = list(info_theory)[0]
            # If it has an "external" key, wrap it up. Else, load it up
            if _external in info_theory[name]:
                theory_class = info_theory[name][_external]
            else:
                theory_class = get_class(name, kind=_theory)
            self.theory = theory_class(info_theory[name], modules=modules, timing=timing)
        else:
            self.theory = None
        # Initialize individual Likelihoods
        self._likelihoods = odict()
        for name, info in info_likelihood.items():
            # If it has an "external" key, wrap it up. Else, load it up
            if _external in info:
                self._likelihoods[name] = LikelihoodExternalFunction(
                    name, info, _theory=getattr(self, _theory, None), timing=timing)
            else:
                like_class = get_class(name)
                self._likelihoods[name] = like_class(info, modules=modules, timing=timing)
        # Assign input/output parameters
        self._assign_params(parameterization, info_likelihood, info_theory)
        # Do the user-defined post-initialisation, and assign the theory code
        if self.theory:
            self.theory.initialize()
        for like in self:
            self[like].initialize()
            self[like].theory = self.theory
            self[like].add_theory()
        # Store the input params and likelihoods on which each sampled params depends.
        self.sampled_input_dependence = parameterization.sampled_input_dependence()
        self.sampled_like_dependence = odict(
            [[p, [like for like in list(self) + ([_theory] if self.theory else [])
                  if any([(i in self[like].input_params) for i in (i_s or [p])])]]
             for p, i_s in self.sampled_input_dependence.items()])
        # Theory parameters "depend" on every likelihood, since re-computing the theory
        # code forces recomputation of the likelihoods
        for p, ls in self.sampled_like_dependence.items():
            if _theory in ls:
                self.sampled_like_dependence[p] = ([_theory] + list(self._likelihoods))
        # Overhead per likelihood evaluation
        self.overhead = _overhead_per_param * len(parameterization.sampled_params())

    # "get" and iteration operate over the dictionary of likelihoods
    # notice that "get" can get "theory", but the iterator does not!
    def __getitem__(self, key):
        try:
            return self._likelihoods.__getitem__(key) if key != _theory else self.theory
        except KeyError:
            self.log.error("Likelihood '%r' not known", key)
            raise HandledException

    def __iter__(self):
        return self._likelihoods.__iter__()

    def logps(self, input_params, _derived=None, cached=True):
        """
        Computes observables and returns the (log) likelihoods *separately*.
        It takes a list of **input** parameter values, in the same order as they appear
        in the `OrderedDictionary` of the :class:`LikelihoodCollection`.
        To compute the derived parameters, it takes an optional keyword `_derived` as an
        empty list, which is then populated with the derived parameter values.
        """
        self.log.debug("Got input parameters: %r", input_params)
        # Prepare the likelihood-defined derived parameters (only computed if requested)
        # Notice that they are in general different from the sampler-defined ones.
        derived_dict = {}
        # If theory code present, compute the necessary products
        if self.theory:
            theory_params_dict = {p: input_params[p] for p in self.theory.input_params}
            theory_success = self.theory.compute(
                _derived=(derived_dict if _derived is not None else None), cached=cached,
                **theory_params_dict)
            if not theory_success:
                self.log.debug(
                    "Theory code computation failed. Not computing likelihood.")
                if _derived is not None:
                    _derived += [np.nan] * len(self.output_params)
                return np.array([-np.inf for _ in self])
        # Compute each log-likelihood, and optionally get the respective derived params
        logps = []
        for like in self:
            this_params_dict = {p: input_params[p] for p in self[like].input_params}
            this_derived_dict = {} if _derived is not None else None
            logps += [self[like]._logp_cached(
                theory_params=(theory_params_dict if self.theory else None),
                _derived=this_derived_dict, cached=cached, **this_params_dict)]
            derived_dict.update(this_derived_dict or {})
            if _derived is not None:
                derived_dict[_chi2 + _separator + like] = -2*logps[-1]
        # Turn the derived params dict into a list and return
        if _derived is not None:
            _derived += [derived_dict[p] for p in self.output_params]
        return np.array(logps)

    def logp(self, input_params, _derived=None, cached=True):
        """
        Computes observables and returns the (log) likelihood.
        It takes a list of **sampled** parameter values, in the same order as they appear
        in the `OrderedDictionary` of the :class:`LikelihoodCollection`.
        To compute the derived parameters, it takes an optional keyword `_derived` as an
        empty list, which is then populated with the derived parameter values.
        """
        return np.sum(self.logps(input_params, _derived=_derived))

    def marginal(self, *args):
        self.log.error("Marginal not implemented for >1 likelihoods. Sorry!")
        raise HandledException

    def _assign_params(self, parameterization, info_likelihood, info_theory=None):
        """
        Assign parameters to likelihoods, following the algorithm explained in
        :doc:`DEVEL`.
        """
        self.input_params = list(parameterization.input_params())
        self.output_params = list(parameterization.output_params())
        params_assign = odict([
            ["input", odict([[p, []] for p in self.input_params])],
            ["output", odict([[p, []] for p in self.output_params])]])
        agnostic_likes = {"input": [], "output": []}
        for kind, option, prefix in (
                ["input", _input_params, _input_params_prefix],
                ["output", _output_params, _output_params_prefix]):
            for like in (list(self) + ([_theory] if self.theory else [])):
                # "one" only takes leftover parameters
                if like == "one":
                    continue
                # Identify parameters understood by this likelihood/therory
                # 1a. Does it have input/output params list?
                #     (takes into account that for callables, we can ignore elements)
                if getattr(self[like], option, None) is not None:
                    for p in getattr(self[like], option):
                        try:
                            params_assign[kind][p] += [like]
                        except KeyError:
                            if kind == "input":
                                # If external function, no problem: it may have default value
                                if not isinstance(self[like], LikelihoodExternalFunction):
                                    self.log.error("Parameter '%s' needed as input for '%s', "
                                                   "but not provided.", p, like)
                                    raise HandledException
                # 2. Is there a params prefix?
                elif getattr(self[like], prefix, None) is not None:
                    for p in params_assign[kind]:
                        if p.startswith(getattr(self[like], prefix)):
                            params_assign[kind][p] += [like]
                # 3. Does it have a general (mixed) list of params?
                elif getattr(self[like], _params, None) is not None:
                    for p in getattr(self[like], _params):
                        if p in params_assign[kind]:
                            params_assign[kind][p] += [like]
                # 4. No parameter knowledge: store as parameter agnostic
                else:
                    agnostic_likes[kind] += [like]
                # Check that there is only one non-knowledgeable element, and assign unused params
                if len(agnostic_likes[kind]) > 1:
                    self.log.error("More than once parameter-agnostic likelihood/theory "
                                   "with respect to %s parameters: %r. Cannot decide "
                                   "parameter assignements.", kind, param_agnostic_likes)
                    raise HandledException
                elif agnostic_likes[kind]:  # if there is only one
                    for p, assigned in params_assign[kind].items():
                        if not assigned:
                            params_assign[kind][p] = [agnostic_likes[kind][0]]
        # Check that, if a theory code is present, it does not share input parameters with
        # any likelihood (because of the theory+experimental model separation)
        if self.theory:
            for p, assigned in params_assign["input"].items():
                 if _theory in assigned and len(assigned) > 1:
                     self.log.error("Some parameter has been asigned to the theory code "
                                    "AND a likelihood, and that is not allowed: {%s: %r}",
                                    p, assigned)
                     raise HandledException
        # If unit likelihood is present, assign all *non-theory* inputs to it
        if "one" in self:
            for p, assigned in params_assign["input"].items():
                if _theory not in assigned:
                    assigned += ["one"]
        # If there are unassigned input params, fail
        unassigned_input = [
            p for p, assigned in params_assign["input"].items() if not assigned]
        if unassigned_input:
            self.log.error("Could not find whom to assign input parameters %r.",
                           unassigned_input)
            raise HandledException
        # Assign the "chi2__" output parameters
        for p in params_assign["output"]:
            if p.startswith(_chi2 + _separator):
                like = p[len(_chi2 + _separator):]
                if like not in list(self):
                    self.log.error("Your derived parameters depend on an unknown "
                                   "likelihood: '%s'", like)
                    raise HandledException
                # They may have been already assigned to an agnostic likelihood,
                # so purge first: no "=+"
                params_assign["output"][p] = [like]
        # Check that output parameters are assigned exactly once
        unassigned_output = [
            p for p, assigned in params_assign["output"].items() if not assigned]
        multiassigned_output = {
            p: assigned for p, assigned in params_assign["output"].items()
            if len(assigned) > 1}
        if unassigned_output:
            self.log.error("Could not find whom to assign output parameters %r.",
                           unassigned_output)
            raise HandledException
        if multiassigned_output:
            self.log.error("Output params can only be computed by one likelihood/theory, "
                           "but some were claimed by more than one: %r.",
                           multiassigned_output)
            raise HandledException
        # Finished! Assign and update infos
        params_assign_inv = odict([["input", odict()], ["output", odict()]])
        for kind, option, attr in (
                ["input", _input_params, "input_params"],
                ["output", _output_params, "output_params"]):
            for like in list(self) + ([_theory] if self.theory else []):
                setattr(
                    self[like], attr,
                    [p for p, assign in params_assign[kind].items() if like in assign])
                # Update infos!
                if like != _theory:
                    info_likelihood[like].pop(_params, None)
                    info_likelihood[like][option] = getattr(self[like], attr)
                elif self.theory:
                    name = list(info_theory)[0]
                    info_theory[name].pop(_params, None)
                    info_theory[name][option] = getattr(self[like], attr)
        if self.log.getEffectiveLevel() <= logging.DEBUG:
            self.log.debug("Parameters were assigned as follows:")
            for like in list(self) + ([_theory] if self.theory else []):
                self.log.debug("- %r:", like)
                self.log.debug("     Input:  %r", self[like].input_params)
                self.log.debug("     Output: %r", self[like].output_params)

    def _speeds_of_params(self, int_speeds=False, fast_slow=False):
        """
        Separates the sampled parameters in blocks according to the likelihood (or theory)
        re-evaluation that changing each one of them involves. Using the appoximate speed
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
        speeds = np.array([getattr(self[like], "speed", -1) for like in self] +
                          ([getattr(self.theory, "speed", -1)] if self.theory else []),
                          dtype=float)
        # Add overhead to the defined ones, and clip to the slowest the undefined ones
        speeds[speeds > 0] = (speeds[speeds > 0] ** -1 + self.overhead) ** -1
        try:
            speeds = np.clip(speeds, min(speeds[speeds > 0]), None)
        except ValueError:
            # No speeds specified
            speeds = np.ones(len(speeds))
        likes = list(self) + ([_theory] if self.theory else [])
        for i, like in enumerate(likes):
            self[like].speed = speeds[i]
        # Compute "footprint"
        # i.e. likelihoods (and theory) that we must recompute when each parameter changes
        footprints = np.zeros((len(self.sampled_like_dependence), len(likes)), dtype=int)
        for i, ls in enumerate(self.sampled_like_dependence.values()):
            for j, like in enumerate(likes):
                footprints[i, j] = like in ls
        # Group parameters by footprint
        different_footprints = list(set([tuple(row) for row in footprints.tolist()]))
        blocks = [[p for ip, p in enumerate(self.sampled_like_dependence)
                   if all(footprints[ip] == fp)] for fp in different_footprints]
        # Find optimal ordering, such that one minimises the time it takes to vary every
        # parameter, one by one, in a basis in which they are mixed-down (i.e after a
        # Cholesky transformation)
        # To do that, compute that "total cost" for every permutation of the blocks order,
        # and find the minumum.
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
                    lambda l: [list(chain(*l[:i_max + 1])), list(chain(*l[i_max + 1:]))])(blocks)
                # In this case, speeds must be *cumulative*, since I am squashing blocks
                cum_inv = lambda ss: 1 / (sum(1 / ss))
                params_speeds = (
                    lambda l: [cum_inv(l[:i_max + 1]), cum_inv(l[i_max + 1:])])(params_speeds)
                self.log.debug("Fast-slow blocking: %r with speeds %r",
                               blocks, params_speeds)
            else:
                self.log.warning("Requested fast/slow separation, "
                                 "but all pararameters have the same speed.")
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
            raise HandledException(
                "Manual blocking not understood. Check documentation.")
        sampled_params = list(self.sampled_like_dependence)
        check = compare_params_lists(
            list(chain(*blocks)), sampled_params)
        duplicate = check.pop("duplicate_A", None)
        missing = check.pop("B_but_not_A", None)
        unknown = check.pop("A_but_not_B", None)
        if duplicate:
            self.log.error("Manual blocking: repeated parameters: %r", duplicate)
            raise HandledException
        if missing:
            self.log.error("Manual blocking: missing parameters: %r", missing)
            raise HandledException
        if unknown:
            self.log.error("Manual blocking: unkown parameters: %r", unknown)
            raise HandledException
        if (speeds != np.sort(speeds)).all():
            self.log.warn("Manual blocking: speed-blocking *apparently* non-optimal: "
                          "sort by ascending speed when possible")
        return speeds, blocks

    def _get_auto_covmat(self, params_info):
        """
        Tries to get an automatic covariance matrix for the current model and data.

        ``params_info`` should contain preferably the slow parameters only.
        """
        likes_renames = {like: {_p_renames: getattr(self[like], _p_renames, [])}
                         for like in self if like != _theory}
        try:
            return self.theory.get_auto_covmat(params_info, likes_renames)
        except:
            return None

    def dump_timing(self):
        avg_times_evals = odict([
            [like, {"t": self[like].time_avg, "n": self[like].n}] for like in
            (([_theory] if self.theory else []) + [like for like in self])
            if getattr(self[like], _timing)])
        if avg_times_evals:
            sep = "\n   "
            self.log.info(
                "Average computation time:" + sep + sep.join(
                ["%s : %g s (%d evaluations)" % (name, vals["t"], vals["n"])
                 for name, vals in avg_times_evals.items()]))

    # Python magic for the "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.theory:
            self.theory.__exit__(exception_type, exception_value, traceback)
        for like in self:
            self[like].__exit__(exception_type, exception_value, traceback)
