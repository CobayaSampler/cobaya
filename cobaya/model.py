"""
.. module:: model

:Synopsis: Wrapper for models: parameterization+prior+likelihood+theory
:Author: Jesus Torrado and Antony Lewis

"""

import dataclasses
import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from itertools import chain
from typing import Any, NamedTuple

import numpy as np

from cobaya import mpi
from cobaya.conventions import (
    get_chi2_name,
    overhead_time,
    packages_path_input,
    prior_1d_name,
)
from cobaya.input import load_info_overrides, update_info
from cobaya.likelihood import (
    AbsorbUnusedParamsLikelihood,
    LikelihoodCollection,
    is_LikelihoodInterface,
)
from cobaya.log import HasLogger, LoggedError, get_logger, is_debug, logger_setup
from cobaya.parameterization import Parameterization
from cobaya.prior import Prior
from cobaya.theory import Provider, Theory, TheoryCollection
from cobaya.tools import (
    are_different_params_lists,
    deepcopy_where_possible,
    recursive_update,
    sort_cosmetic,
    sort_parameter_blocks,
    str_to_list,
)
from cobaya.typing import (
    InfoDict,
    InputDict,
    LikesDict,
    ParamsDict,
    ParamValuesDict,
    PriorsDict,
    TheoriesDict,
    empty_dict,
    unset_params,
)
from cobaya.yaml import yaml_dump


@contextmanager
def timing_on(model: "Model"):
    was_on = model.timing
    if not was_on:
        model.set_timing_on(True)
    try:
        yield
    finally:
        if not was_on:
            model.set_timing_on(False)


@dataclasses.dataclass(frozen=True)
class LogPosterior:
    """
    Class holding the result of a log-posterior computation, including log-priors,
    log-likelihoods and derived parameters.

    A consistency check will be performed if initialized simultaneously with
    log-posterior, log-priors and log-likelihoods, so, for faster initialisation,
    you may prefer to pass log-priors and log-likelihoods only, and only pass all three
    (so that the test is performed) only when e.g. loading from an old sample.

    If ``finite=True`` (default: False), it will try to represent infinities as the
    largest real numbers allowed by machine precision.
    """

    # A note on typing:
    # Though None is allowed for some arguments, after initialisation everything should
    # be not None. So we can either (a) use Optional, and then get A LOT of typing errors
    # or (b) not use it (use dataclasses.field(default=None) instead) and get fewer errors
    # (only wherever LogPosterior is initialised).
    # Let's opt for (b) and suppress errors there.

    logpost: float = dataclasses.field(default=None)  # type: ignore
    logpriors: Sequence[float] = dataclasses.field(default=None)  # type: ignore
    loglikes: Sequence[float] = dataclasses.field(default=None)  # type: ignore
    derived: Sequence[float] = dataclasses.field(default=None)  # type: ignore
    finite: bool = dataclasses.field(default=False)
    logprior: float = dataclasses.field(init=False, repr=False)
    loglike: float = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        """Sets the value of logpost if necessary, and checks for consistency."""
        # Consistency: derived = None --> []
        if self.derived is None:
            object.__setattr__(self, "derived", [])
        object.__setattr__(
            self, "logprior", sum(self.logpriors) if self.logpriors is not None else None
        )
        object.__setattr__(
            self, "loglike", sum(self.loglikes) if self.loglikes is not None else None
        )
        if self.finite:
            self.make_finite()
        if self.logpost is None:
            if self.logpriors is None or self.loglikes is None:
                raise ValueError(
                    "If `logpost` not passed, both `logpriors` and "
                    "`loglikes` must be passed."
                )
            object.__setattr__(self, "logpost", self._logpost())
        elif self.logpriors is not None and self.loglikes is not None:
            if not self._logpost_is_consistent():
                raise ValueError(
                    "The given log-posterior is not equal to the "
                    "sum of given log-priors and log-likelihoods: "
                    "%g != sum(%r) + sum(%r)"
                    % (self.logpost, self.logpriors, self.loglikes)
                )

    def _logpost(self):
        """Computes logpost from prior and likelihood product."""
        return self.logprior + self.loglike

    def _logpost_is_consistent(self):
        """
        Checks that the sum of logpriors and loglikes (if present) add up to logpost, if
        passed.
        """
        if self.finite:
            return np.isclose(np.nan_to_num(self.logpost), np.nan_to_num(self._logpost()))
        else:
            return np.isclose(self.logpost, self._logpost())

    def make_finite(self):
        """
        Ensures that infinities are represented as the largest real numbers allowed by
        machine precision, instead of `+/- numpy.inf`.
        """
        object.__setattr__(self, "finite", True)
        if self.logpost is not None:
            object.__setattr__(self, "logpost", np.nan_to_num(self.logpost))
        if self.logpriors is not None:
            object.__setattr__(self, "logpriors", np.nan_to_num(self.logpriors))
            object.__setattr__(self, "logprior", np.nan_to_num(self.logprior))
        if self.loglikes is not None:
            object.__setattr__(self, "loglikes", np.nan_to_num(self.loglikes))
            object.__setattr__(self, "loglike", np.nan_to_num(self.loglike))

    def as_dict(self, model: "Model") -> dict[str, float | dict[str, float]]:
        """
        Given a :class:`~model.Model`, returns a more informative version of itself,
        containing the names of priors, likelihoods and derived parameters.
        """
        return {
            "logpost": self.logpost,
            "logpriors": dict(zip(model.prior, self.logpriors)),
            "loglikes": dict(zip(model.likelihood, self.loglikes)),
            "derived": dict(zip(model.parameterization.derived_params(), self.derived)),
        }


class Requirement(NamedTuple):
    name: str
    options: InfoDict | None

    def __eq__(self, other):
        return self.name == other.name and _dict_equal(self.options, other.options)

    def __repr__(self):
        return f"{{{self.name!r}:{self.options!r}}}"


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

    raise ValueError(
        "Requirements must be a dict of names and options, a list of names, "
        "or an iterable of requirement (name, option) pairs"
    )


def _dict_equal(d1, d2):
    # dict/None equality test accounting for numpy arrays not supporting standard eq
    if type(d1) is not type(d2):
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
    if hasattr(d1, "__len__"):
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

    def __init__(
        self,
        info_params: ParamsDict,
        info_likelihood: LikesDict,
        info_prior: PriorsDict | None = None,
        info_theory: TheoriesDict | None = None,
        packages_path=None,
        timing=None,
        allow_renames=True,
        stop_at_error=False,
        post=False,
        skip_unused_theories=False,
        dropped_theory_params: Iterable[str] | None = None,
    ):
        """
        Creates an instance of :class:`model.Model`.

        It is recommended to use the simpler function :func:`~model.get_model` instead.
        """
        self.set_logger()
        self._updated_info: InputDict = {
            "params": deepcopy_where_possible(info_params),
            "likelihood": deepcopy_where_possible(info_likelihood),
        }
        if not self._updated_info["likelihood"]:
            raise LoggedError(self.log, "No likelihood requested!")
        for k, v in (
            ("prior", info_prior),
            ("theory", info_theory),
            (packages_path_input, packages_path),
            ("timing", timing),
        ):
            if v not in (None, {}):
                self._updated_info[k] = deepcopy_where_possible(v)  # type: ignore
        self.parameterization = Parameterization(
            self._updated_info["params"],
            allow_renames=allow_renames,
            ignore_unused_sampled=post,
        )
        self.prior = Prior(self.parameterization, self._updated_info.get("prior"))
        self.timing = timing
        info_theory = self._updated_info.get("theory")
        self.theory = TheoryCollection(
            info_theory or {}, packages_path=packages_path, timing=timing
        )
        info_likelihood = self._updated_info["likelihood"]
        self.likelihood = LikelihoodCollection(
            info_likelihood,
            theory=self.theory,
            packages_path=packages_path,
            timing=timing,
        )
        if stop_at_error:
            for component in self.components:
                component.stop_at_error = stop_at_error
        # Assign input/output parameters
        self._assign_params(info_likelihood, info_theory, dropped_theory_params)
        self._set_dependencies_and_providers(skip_unused_theories=skip_unused_theories)
        # Add to the updated info some values that are only available after initialisation
        self._updated_info = recursive_update(
            self._updated_info, self.get_versions(add_version_field=True)
        )
        # Overhead per likelihood evaluation, approximately ind from # input params
        # Evaluation of non-uniform priors will add some overhead per parameter.
        self.overhead = overhead_time

    def info(self) -> InputDict:
        """
        Returns a copy of the information used to create the model, including defaults
        and some new values that are only available after initialisation.
        """
        return deepcopy_where_possible(self._updated_info)

    def _to_sampled_array(
        self, params_values: dict[str, float] | Sequence[float]
    ) -> np.ndarray:
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
                    self.log,
                    "Wrong dimensionality: it's %d, and it should be %d.",
                    len(params_values_array),
                    self.prior.d(),
                )
        if len(params_values_array.shape) >= 2:
            raise LoggedError(
                self.log, "Cannot take arrays of points as inputs, just single points."
            )
        return params_values_array

    def logpriors(
        self,
        params_values: dict[str, float] | Sequence[float],
        as_dict: bool = False,
        make_finite: bool = False,
    ) -> np.ndarray | dict[str, float]:
        """
        Takes an array or dictionary of sampled parameter values.
        If the argument is an array, parameters must have the same order as in the input.
        When in doubt, you can get the correct order as
        ``list([your_model].parameterization.sampled_params())``.

        Returns the log-values of the priors, in the same order as it is returned by
        ``list([your_model].prior)``. The first one, named ``0``, corresponds to the
        product of the 1-dimensional priors specified in the ``params`` block, and it's
        normalized (in general, the external prior densities aren't).

        If ``as_dict=True`` (default: False), returns a dictionary containing the prior
        names as keys, instead of an array.

        If ``make_finite=True``, it will try to represent infinities as the largest real
        numbers allowed by machine precision.
        """
        params_values = self.parameterization.check_sampled(params_values)
        params_values_array = self._to_sampled_array(params_values)
        logpriors = np.asarray(self.prior.logps(params_values_array))
        if make_finite:
            return np.nan_to_num(logpriors)
        if as_dict:
            return dict(zip(self.prior, logpriors))
        else:
            return logpriors

    def logprior(
        self,
        params_values: dict[str, float] | Sequence[float],
        make_finite: bool = False,
    ) -> float:
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

    def _loglikes_input_params(
        self,
        input_params: ParamValuesDict,
        return_derived: bool = True,
        return_output_params: bool = False,
        as_dict: bool = False,
        make_finite: bool = False,
        cached: bool = True,
    ) -> (
        np.ndarray
        | dict[str, float]
        | tuple[np.ndarray | dict[str, float], list[float] | dict[str, float]]
    ):
        """
        Takes a dict of input parameters, computes the likelihood pipeline, and returns
        the log-likelihoods and derived parameters.

        To return just the list of log-likelihood values, make ``return_derived=False``
        (default: True).

        To return raw output parameters (including chi2's) instead of derived parameters,
        make ``return_output_params=True`` (default=False). This overrides
        ``return_derived``.

        If ``as_dict=True`` (default: False), returns a dictionary containing the
        likelihood names (and derived parameters, if ``return_derived=True``) as keys,
        instead of an array.

        If ``make_finite=True``, it will try to represent infinities as the largest real
        numbers allowed by machine precision.

        If ``cached=False`` (default: True), it ignores previously computed results that
        could be reused.
        """
        # Calculate required results and returns likelihoods
        outpar_dict: ParamValuesDict = {}
        compute_success = True
        self.provider.set_current_input_params(input_params)
        self.param_dict_debug("Got input parameters: %r", input_params)
        loglikes = np.zeros(len(self.likelihood))
        need_derived = self.requires_derived or return_derived or return_output_params
        for (component, like_index), param_dep in zip(
            self._component_order.items(), self._params_of_dependencies
        ):
            depend_list = [input_params[p] for p in param_dep]
            params = {p: input_params[p] for p in component.input_params}
            compute_success = component.check_cache_and_compute(
                params,
                want_derived=need_derived,
                dependency_params=depend_list,
                cached=cached,
            )
            if not compute_success:
                loglikes[:] = -np.inf
                self.log.debug("Calculation failed, skipping rest of calculations ")
                break
            if return_derived or return_output_params:
                outpar_dict.update(component.current_derived)
            if like_index is not None:
                try:
                    loglikes[like_index] = component.current_logp
                except TypeError as type_excpt:
                    raise LoggedError(
                        self.log,
                        "Likelihood %s has not returned a valid log-likelihood, "
                        "but %s instead.",
                        component,
                        component.current_logp,
                    ) from type_excpt
        if make_finite:
            loglikes = np.nan_to_num(loglikes)
        return_likes = dict(zip(self.likelihood, loglikes)) if as_dict else loglikes
        if return_derived or return_output_params:
            if not compute_success:
                return_params_names = (
                    self.output_params if return_output_params else self.derived_params
                )
                if as_dict:
                    return_params = dict.fromkeys(return_params_names, np.nan)
                else:
                    return_params = [np.nan] * len(return_params_names)
            else:
                # Add chi2's to derived parameters
                for chi2_name, indices in self._chi2_names:
                    outpar_dict[chi2_name] = -2 * sum(loglikes[i] for i in indices)
                if return_output_params:
                    return_params = outpar_dict if as_dict else list(outpar_dict.values())
                else:  # explicitly derived, instead of output params
                    derived_dict = self.parameterization.to_derived(outpar_dict)
                    self.param_dict_debug("Computed derived parameters: %s", derived_dict)
                    return_params = (
                        derived_dict if as_dict else list(derived_dict.values())
                    )
            return return_likes, return_params
        return return_likes

    def loglikes(
        self,
        params_values: dict[str, float] | Sequence[float] | None = None,
        as_dict: bool = False,
        make_finite: bool = False,
        return_derived: bool = True,
        cached: bool = True,
    ) -> (
        np.ndarray
        | dict[str, float]
        | tuple[np.ndarray | dict[str, float], list[float] | dict[str, float]]
    ):
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

        To return just the list of log-likelihood values, make ``return_derived=False``
        (default: True).

        If ``as_dict=True`` (default: False), returns a dictionary containing the
        likelihood names (and derived parameters, if ``return_derived=True``) as keys,
        instead of an array.

        If ``make_finite=True``, it will try to represent infinities as the largest real
        numbers allowed by machine precision.

        If ``cached=False`` (default: True), it ignores previously computed results that
        could be reused.
        """
        params_values = self.parameterization.check_sampled(params_values)
        input_params = self.parameterization.to_input(params_values)
        return self._loglikes_input_params(
            input_params,
            return_derived=return_derived,
            cached=cached,
            make_finite=make_finite,
            as_dict=as_dict,
        )

    def loglike(
        self,
        params_values: dict[str, float] | Sequence[float] | None = None,
        make_finite: bool = False,
        return_derived: bool = True,
        cached: bool = True,
    ) -> float | tuple[float, np.ndarray]:
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

        To return just the list of log-likelihood values, make ``return_derived=False``,
        (default: True).

        If ``make_finite=True``, it will try to represent infinities as the largest real
        numbers allowed by machine precision.

        If ``cached=False`` (default: True), it ignores previously computed results that
        could be reused.
        """
        ret_value = self.loglikes(
            params_values,
            return_derived=return_derived,
            cached=cached,
            make_finite=make_finite,
        )
        if return_derived:
            return np.sum(ret_value[0]), ret_value[1]  # type: ignore
        else:
            return np.sum(ret_value)

    def logposterior(
        self,
        params_values: dict[str, float] | Sequence[float],
        as_dict: bool = False,
        make_finite: bool = False,
        return_derived: bool = True,
        cached: bool = True,
        _no_check: bool = False,
    ) -> LogPosterior | dict:
        """
        Takes an array or dictionary of sampled parameter values.
        If the argument is an array, parameters must have the same order as in the input.
        When in doubt, you can get the correct order as
        ``list([your_model].parameterization.sampled_params())``.

        Returns a :class:`~model.LogPosterior` object (except if ``as_dict=True``, see
        below), with the following fields:

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

        To ignore the derived parameters, make ``return_derived=False`` (default: True).

        If ``as_dict=True`` (default: False), returns a dictionary containing prior names,
        likelihoods names and, if applicable, derived parameters names as keys, instead of
        a :class:`~model.LogPosterior` object.

        If ``make_finite=True``, it will try to represent infinities as the largest real
        numbers allowed by machine precision.

        If ``cached=False`` (default: True), it ignores previously computed results that
        could be reused.
        """
        if _no_check:
            params_values_array = params_values
        else:
            params_values = self.parameterization.check_sampled(params_values)
            params_values_array = self._to_sampled_array(params_values)
            if self.is_debug():
                self.log.debug(
                    "Posterior to be computed for parameters %s",
                    dict(
                        zip(
                            self.parameterization.sampled_params(),
                            params_values_array.astype(float),
                        )
                    ),
                )
            if not np.all(np.isfinite(params_values_array)):
                raise LoggedError(
                    self.log,
                    "Got non-finite parameter values: %r",
                    dict(
                        zip(
                            self.parameterization.sampled_params(),
                            params_values_array.astype(float),
                        )
                    ),
                )
        # Notice that we don't use the make_finite in the prior call,
        # to correctly check if we have to compute the likelihood
        logpriors_1d = self.prior.logps_internal(params_values_array)
        if logpriors_1d == -np.inf:
            logpriors = [-np.inf] * (1 + len(self.prior.external))
        else:
            input_params = self.parameterization.to_input(params_values_array)
            logpriors = [logpriors_1d]
            if self.prior.external:
                logpriors.extend(self.prior.logps_external(input_params))
        if -np.inf not in logpriors:
            like = self._loglikes_input_params(
                input_params,
                return_derived=return_derived,
                cached=cached,
                make_finite=make_finite,
            )
            loglikes, derived_sampler = like if return_derived else (like, [])
        else:
            loglikes = []
            derived_sampler = []
        logposterior = LogPosterior(
            logpriors=logpriors,
            loglikes=loglikes,
            derived=derived_sampler,
            finite=make_finite,
        )
        if as_dict:
            return logposterior.as_dict(self)
        else:
            return logposterior

    def logpost(
        self,
        params_values: dict[str, float] | Sequence[float],
        make_finite: bool = False,
        cached: bool = True,
    ) -> float:
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
        return self.logposterior(
            params_values,
            make_finite=make_finite,
            return_derived=False,
            cached=cached,
        ).logpost

    def get_valid_point(
        self,
        max_tries: int,
        ignore_fixed_ref: bool = False,
        logposterior_as_dict: bool = False,
        random_state=None,
    ) -> tuple[np.ndarray, LogPosterior | dict]:
        """
        Finds a point with finite posterior, sampled from the reference pdf.

        It will fail if no valid point is found after `max_tries`.

        If `ignored_fixed_ref=True` (default: `False`), fixed reference values will be
        ignored in favor of the full prior, ensuring some randomness for all parameters
        (useful e.g. to prevent caching when measuring speeds).

        Returns (point, LogPosterior(logpost, logpriors, loglikes, derived))

        If ``logposterior_as_dict=True`` (default: False), returns for the log-posterior
        a dictionary containing prior names, likelihoods names and, if applicable, derived
        parameters names as keys, instead of a :class:`~model.LogPosterior` object.
        """
        for loop in range(max(1, max_tries // self.prior.d())):
            initial_point = self.prior.reference(
                max_tries=max_tries,
                ignore_fixed=ignore_fixed_ref,
                warn_if_no_ref=not loop,
                random_state=random_state,
            )
            results = self.logposterior(initial_point)
            if results.logpost != -np.inf:
                break
        else:
            if self.prior.reference_is_pointlike:
                raise LoggedError(
                    self.log,
                    "The reference point provided has null "
                    "likelihood. Set 'ref' to a different point "
                    "or a pdf.",
                )
            raise LoggedError(
                self.log,
                "Could not find random point giving finite posterior after %g tries",
                max_tries,
            )
        if logposterior_as_dict:
            results = results.as_dict(self)
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
        return {
            "theory": self.theory.get_versions(add_version_field=add_version_field),
            "likelihood": self.likelihood.get_versions(
                add_version_field=add_version_field
            ),
        }

    def get_speeds(self, ignore_sub=False):
        return {
            "theory": self.theory.get_speeds(ignore_sub=ignore_sub),
            "likelihood": self.likelihood.get_speeds(ignore_sub=ignore_sub),
        }

    def _set_component_order(self, components, dependencies):
        dependence_order: list[Theory] = []
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
                raise LoggedError(
                    self.log, "Circular dependency, cannot calculate %r" % comps
                )
            _last = len(dependence_order)
        likes = list(self.likelihood.values())
        self._component_order = {
            c: likes.index(c) if c in likes else None for c in dependence_order
        }

    def _set_dependencies_and_providers(
        self, manual_requirements=empty_dict, skip_unused_theories=False
    ):
        components: list[Theory] = self.components
        direct_param_dependence: dict[Theory, set[str]] = {c: set() for c in components}

        def _tidy_requirements(_require, _component=None):
            # take input requirement and return tuples of
            # requirement names and requirement options
            if not _require:
                return []
            _require = as_requirement_list(_require)
            requirements_in_input_params = {req.name for req in _require}.intersection(
                self.input_params
            )
            if requirements_in_input_params and _component is not None:
                # Save parameters dependence
                direct_param_dependence[_component].update(requirements_in_input_params)
                # requirements that are sampled parameters automatically satisfied
                return [
                    _req
                    for _req in _require
                    if _req.name not in requirements_in_input_params
                ]
            else:
                return _require

        # ## 1. Get the requirements and providers ##
        # requirements of each component
        requirements: dict[Theory, list[Requirement]] = {}
        # providers of each *available* requirement (requested or not)
        providers: dict[str, list[Theory]] = {}
        # set of requirement names that may be required derived parameters
        requirements_are_params: set[str] = set()
        for component in components:
            component.initialize_with_params()
            requirements[component] = _tidy_requirements(
                component.get_requirements(), component
            )
            # Component params converted to requirements if not explicitly sampled
            requirements[component] += [
                Requirement(p, None)
                for p in (getattr(component, "params", {}) or [])
                if p not in self.input_params and p not in component.output_params
            ]
            # Gather what this component can provide
            can_provide = list(component.get_can_provide()) + list(
                component.get_can_provide_methods()
            )
            # Parameters that can be provided
            # Corner case: some components can either take some parameters as input OR
            # provide their own calculation of them. Pop those if required as input.
            provide_params = {
                p
                for p in component.get_can_provide_params()
                if all(p != req.name for req in requirements[component])
            }
            provide_params.update(component.output_params)
            requirements_are_params.update(provide_params)
            # Invert to get the provider(s) of each available product/parameter
            for k in chain(can_provide, provide_params):
                providers[k] = providers.get(k, [])
                if component not in providers[k]:
                    providers[k].append(component)
        # Add requirements requested by hand
        manual_theory = Theory(name="_manual")
        if manual_requirements:
            self._manual_requirements = getattr(
                self, "_manual_requirements", []
            ) + _tidy_requirements(manual_requirements)
            requirements[manual_theory] = deepcopy(self._manual_requirements)

        # ## 2. Assign each requirement to a provider ##
        # store requirements assigned to each provider:
        self._must_provide: dict[Theory, list[Requirement]] = {c: [] for c in components}
        # inverse of the one above, *minus conditional requirements* --
        # this is the one we need to initialise the provider
        requirement_providers = {}
        dependencies: dict[
            Theory, set[Theory]
        ] = {}  # set of components of which each component depends
        used_suppliers = {c for c in components if c.output_params}
        there_are_more_requirements = True
        # temp list of dictionary of requirements for each component
        must_provide = {
            c: [Requirement(p, None) for p in c.output_params] for c in components
        }
        while there_are_more_requirements:
            # Check supplier for each requirement, get dependency and must_provide
            for component, requires in requirements.items():
                for requirement in requires:
                    suppliers = providers.get(requirement.name)
                    if not suppliers:
                        # If failed requirement was manually added,
                        # remove from list, or it will still fail in the next call too
                        requirements[manual_theory] = [
                            req
                            for req in requirements.get(manual_theory, [])
                            if req.name != requirement.name
                        ]
                        raise LoggedError(
                            self.log,
                            "Requirement %s of %r is not provided by any "
                            "component, nor sampled directly",
                            requirement.name,
                            component,
                        )
                    supplier: Theory | None
                    if len(suppliers) == 1:
                        supplier = suppliers[0]
                    else:
                        supplier = None
                        for sup in suppliers:
                            provide = str_to_list(getattr(sup, "provides", []))
                            if requirement.name in provide:
                                if supplier:
                                    raise LoggedError(
                                        self.log,
                                        "more than one component provides %s",
                                        requirement.name,
                                    )
                                supplier = sup
                        if not supplier:
                            raise LoggedError(
                                self.log,
                                "requirement %s is provided by more than one "
                                "component: %s. Use 'provides' keyword to "
                                "specify which provides it",
                                requirement.name,
                                suppliers,
                            )
                    if supplier is component:
                        raise LoggedError(
                            self.log,
                            "Component %r cannot provide %s to itself!",
                            component,
                            requirement.name,
                        )
                    requirement_providers[requirement.name] = supplier.get_provider()
                    used_suppliers.add(supplier)
                    declared_requirements_for_this_supplier = (
                        self._must_provide[supplier] + must_provide[supplier]
                    )
                    if requirement not in declared_requirements_for_this_supplier:
                        must_provide[supplier] += [requirement]
                    dependencies[component] = dependencies.get(component, set()) | {
                        supplier
                    }
                    # Requirements per component excluding input params
                    # -- saves some overhead in theory.check_cache_and_compute
                    if (
                        component is not manual_theory
                        and requirement.name not in component.input_params
                        and requirement.name in requirements_are_params
                    ):
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
                    conditional_requirements = _tidy_requirements(
                        component.must_provide(**{request.name: request.options}),
                        component,
                    )
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
                        self.log,
                        "Could not find anything to use input parameter(s) %r.",
                        unassigned,
                    )
                else:
                    self.mpi_warning(
                        "Parameter(s) %s are only used by the prior",
                        self._unassigned_input,
                    )

        unused_theories = set(self.theory.values()) - used_suppliers
        if unused_theories:
            if skip_unused_theories:
                self.mpi_debug(
                    "Theories %s do not need to be computed and will be skipped",
                    unused_theories,
                )
                for theory in unused_theories:
                    self._component_order.pop(theory, None)
                    components.remove(theory)
            else:
                self.mpi_warning(
                    "Theories %s do not appear to be actually used for anything",
                    unused_theories,
                )

        self.mpi_debug("Components will be computed in the order:")
        self.mpi_debug(" - %r" % list(self._component_order))

        def dependencies_of(_component):
            deps = set()
            for c in dependencies.get(_component, []):
                deps.add(c)
                deps.update(dependencies_of(c))
            return deps

        # ## 3. Save dependencies on components and their parameters ##
        self._dependencies = {c: dependencies_of(c) for c in components}
        # this next one is not a dict to save a lookup per iteration
        self._params_of_dependencies: list[set[str]] = [
            set() for _ in self._component_order
        ]
        for component, param_dep in zip(
            self._component_order, self._params_of_dependencies
        ):
            param_dep.update(direct_param_dependence.get(component) or [])
            for dep in self._dependencies.get(component, []):
                param_dep.update(
                    set(dep.input_params).union(direct_param_dependence.get(dep) or [])
                )
            param_dep -= set(component.input_params)
            if (
                not len(component.input_params)
                and not param_dep
                and component.get_name() != "one"
            ):
                raise LoggedError(
                    self.log,
                    "Component '%r' seems not to depend on any parameters "
                    "(neither directly nor indirectly)",
                    component,
                )
        # Store the input params and components on which each sampled params depends.
        sampled_input_dependence = self.parameterization.sampled_input_dependence()
        sampled_dependence: dict[str, list[Theory]] = {
            p: [] for p in sampled_input_dependence
        }
        for p, i_s in sampled_input_dependence.items():
            for component in components:
                if (
                    p in component.input_params
                    or i_s
                    and any(p_i in component.input_params for p_i in i_s)
                ):
                    sampled_dependence[p].append(component)
                    for comp in components:
                        if comp is not component and component in self._dependencies.get(
                            comp, []
                        ):
                            sampled_dependence[p].append(comp)
        self.sampled_dependence = sampled_dependence
        self.requires_derived: set[str] = requirements_are_params.intersection(
            requirement_providers
        )

        # ## 4. Initialize the provider and pass it to each component ##
        if self.is_debug_and_mpi_root():
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

    def _assign_params(
        self, info_likelihood, info_theory=None, dropped_theory_params=None
    ):
        """
        Assign input and output parameters to theories and likelihoods, following the
        algorithm explained in :doc:`DEVEL`.
        """
        self.input_params = [
            p
            for p in self.parameterization.input_params()
            if p not in self.parameterization.dropped_param_set()
        ]
        self.output_params = list(self.parameterization.output_params())
        self.derived_params = list(self.parameterization.derived_params())
        input_assign: dict[str, list[Theory]] = {p: [] for p in self.input_params}
        output_assign: dict[str, list[Theory]] = {p: [] for p in self.output_params}
        # Go through all components.
        # NB: self.components iterates over likelihoods first, and then theories
        # so unassigned can by default go to theories
        assign_components = [
            c for c in self.components if not isinstance(c, AbsorbUnusedParamsLikelihood)
        ]
        for assign, option, prefix, derived_param in (
            (input_assign, "input_params", "input_params_prefix", False),
            (output_assign, "output_params", "output_params_prefix", True),
        ):
            agnostic_likes = []
            for component in assign_components:
                if derived_param:
                    required_params = set(str_to_list(getattr(component, "provides", [])))
                else:
                    required_params = {
                        p
                        for p, v in as_requirement_list(component.get_requirements())
                        # ignore non-params; it's ok if some non-param goes through
                        if v is None
                    }
                # Identify parameters understood by this likelihood/theory
                # 1a. Does it have input/output params list?
                #     (takes into account that for callables, we can ignore elements)
                if getattr(component, option) is not unset_params:
                    for p in getattr(component, option):
                        try:
                            assign[p] += [component]
                        except KeyError as excpt:
                            if not derived_param:
                                raise LoggedError(
                                    self.log,
                                    "Parameter '%s' needed as input for '%s', "
                                    "but not provided.",
                                    p,
                                    component.get_name(),
                                ) from excpt
                # 2. Is there a params prefix?
                elif (_prefix := getattr(component, prefix, None)) is not None:
                    for p in assign:
                        if p.startswith(_prefix):
                            assign[p] += [component]
                # 3. Does it have a general (mixed) list of params? (set from default)
                # 4. or otherwise required
                elif (_params := getattr(component, "params", {})) or required_params:
                    if _params:
                        for p, options in _params.items():
                            if (
                                not isinstance(options, Mapping)
                                and not derived_param
                                or isinstance(options, Mapping)
                                and options.get("derived", False) is derived_param
                            ):
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
                for p in unassigned if derived_param else assign:
                    if p in pars_to_assign and component not in assign[p]:
                        assign[p] += [component]
            # Check that there is only one non-knowledgeable element, and assign
            # unused params
            if len(agnostic_likes) > 1 and not all(assign.values()):
                raise LoggedError(
                    self.log,
                    "More than one parameter-agnostic likelihood/theory "
                    "with respect to %s: %r. Cannot decide "
                    "parameter assignments.",
                    option,
                    agnostic_likes,
                )
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
        self._unassigned_input = {
            p for p, assigned in input_assign.items() if not assigned
        }.difference(
            chain(
                *(
                    self.parameterization.input_dependencies.get(p, [])
                    for p, assigned in input_assign.items()
                    if assigned
                )
            )
        )

        chi2_names: dict[str, list[int]] = {}
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
                like = p[len(get_chi2_name("")) :]
                index = list(self.likelihood).index(like)
                if index is None:
                    raise LoggedError(
                        self.log,
                        "Your derived parameters depend on an unknown likelihood: '%s'",
                        like,
                    )
                if p in chi2_names:
                    raise LoggedError(
                        self.log,
                        "Your have likelihoods with type labels that are the "
                        "same as a likelihood",
                        like,
                    )
                chi2_names[p] = [index]
                # They may have been already assigned to an agnostic likelihood,
                # so purge first: no "=+"
                output_assign[p] = [self.likelihood[like]]
        self._chi2_names = tuple(chi2_names.items())
        # Check that there are no unassigned parameters (with the exception of aggr chi2)
        if unassigned_output := [
            p for p, assigned in output_assign.items() if not assigned
        ]:
            raise LoggedError(
                self.log,
                "Could not find whom to assign output parameters %r.",
                unassigned_output,
            )
        # Check that output parameters are assigned exactly once
        if multi_assigned_output := {
            p: assigned for p, assigned in output_assign.items() if len(assigned) > 1
        }:
            raise LoggedError(
                self.log,
                "Output params can only be computed by one likelihood/theory, "
                "but some were claimed by more than one: %r.",
                multi_assigned_output,
            )
        # Finished! Assign and update infos
        for assign, option, output in (
            (input_assign, "input_params", False),
            (output_assign, "output_params", True),
        ):
            for component in self.components:
                assign_params = [p for p, assign in assign.items() if component in assign]
                current_assign = getattr(component, option)
                if output or current_assign is unset_params:
                    setattr(component, option, assign_params)
                elif set(assign_params) != set(current_assign):
                    raise LoggedError(
                        self.log,
                        "exising %s %r do not match assigned parameters %r",
                        option,
                        assign_params,
                        current_assign,
                    )

                # Update infos! (helper theory parameters stored in yaml with host)
                inf = (
                    info_likelihood
                    if component in self.likelihood.values()
                    else info_theory
                )
                if inf := inf.get(component.get_name()):
                    inf.pop("params", None)
                    inf[option] = component.get_attr_list_with_helpers(option)
        if self.is_debug_and_mpi_root():
            self.log.debug("Parameters were assigned as follows:")
            for component in self.components:
                self.log.debug("- %r:", component)
                self.log.debug("     Input:  %r", component.input_params)
                self.log.debug("     Output: %r", component.output_params)

    @property
    def components(self) -> list[Theory]:
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
        sampled_dependence_names = {
            k: [c.get_name() for c in v] for k, v in self.sampled_dependence.items()
        }
        for i, ls in enumerate(sampled_dependence_names.values()):
            for j, comp in enumerate(speeds):
                footprints[i, j] = comp in ls
        # Group parameters by footprint
        different_footprints = list({tuple(row) for row in footprints})
        blocks = [
            [
                p
                for ip, p in enumerate(self.sampled_dependence)
                if all(footprints[ip] == fp)
            ]
            for fp in different_footprints
        ]
        # a) Multiple blocks
        if not split_fast_slow:
            i_optimal_ordering, costs, oversample_factors = sort_parameter_blocks(
                blocks,
                np.array(list(speeds.values()), dtype=float),
                different_footprints,
                oversample_power=oversample_power,
            )
            blocks_sorted = [blocks[i] for i in i_optimal_ordering]
        # b) 2-block slow-fast separation
        else:
            if len(blocks) == 1:
                raise LoggedError(
                    self.log,
                    "Requested fast/slow separation, "
                    "but all parameters have the same speed.",
                )
            # First sort them optimally (w/o oversampling)
            i_optimal_ordering, costs, oversample_factors = sort_parameter_blocks(
                blocks,
                np.array(list(speeds.values()), dtype=float),
                different_footprints,
                oversample_power=0,
            )
            blocks_sorted = [blocks[i] for i in i_optimal_ordering]
            footprints_sorted = np.array(different_footprints)[list(i_optimal_ordering)]
            # Then, find the split that maxes cost LOG-differences.
            # Since costs are already "accumulated down",
            # we need to subtract those below each one
            costs_per_block = costs - np.concatenate([costs[1:], [0]])
            # Split them so that "adding the next block to the slow ones" has max cost
            log_differences = np.log(costs_per_block[:-1]) - np.log(costs_per_block[1:])
            i_last_slow: int = np.argmax(log_differences)  # type: ignore
            blocks_split = (
                lambda L: [
                    list(chain(*L[: i_last_slow + 1])),
                    list(chain(*L[i_last_slow + 1 :])),
                ]
            )(blocks_sorted)
            footprints_split = [
                np.array(footprints_sorted[: i_last_slow + 1]).sum(axis=0)
            ] + [np.array(footprints_sorted[i_last_slow + 1 :]).sum(axis=0)]
            footprints_split = np.clip(np.array(footprints_split), 0, 1)  # type: ignore
            # Recalculate oversampling factor with 2 blocks
            _, _, oversample_factors = sort_parameter_blocks(
                blocks_split,
                np.array(list(speeds.values()), dtype=float),
                footprints_split,
                oversample_power=oversample_power,
            )
            # If no oversampling, slow-fast separation makes no sense: warn and set to 2
            if oversample_factors[1] == 1:
                min_factor = 2
                self.mpi_warning(
                    "Oversampling would be trivial due to small speed difference or "
                    "small `oversample_power`. Set to %d.",
                    min_factor,
                )
            # Finally, unfold `oversampling_factors` to have the right number of elements,
            # taking into account that that of the fast blocks should be interpreted as a
            # global one for all of them.
            # NB: the int() below forces the copy of the factors.
            #     Otherwise the yaml_representer prints references to a single object.
            oversample_factors = [int(oversample_factors[0])] * (1 + i_last_slow) + [
                int(oversample_factors[1])
            ] * (len(blocks) - (1 + i_last_slow))
            self.mpi_debug(
                "Doing slow/fast split. The oversampling factors for "
                "the fast blocks should be interpreted as a global one "
                "for all of them"
            )
        if self.is_debug_and_mpi_root():
            self.log.debug(
                "Cost, oversampling factor and parameters per block, in optimal order:"
            )
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
        except (TypeError, ValueError) as excpt:
            raise LoggedError(
                self.log, "Manual blocking not understood. Check documentation."
            ) from excpt
        sampled_params = list(self.sampled_dependence)
        check = are_different_params_lists(list(chain(*blocks)), sampled_params)
        duplicate = check.pop("duplicate_A", None)
        missing = check.pop("B_but_not_A", None)
        unknown = check.pop("A_but_not_B", None)
        if duplicate:
            raise LoggedError(
                self.log, "Manual blocking: repeated parameters: %r", duplicate
            )
        if missing:
            raise LoggedError(
                self.log, "Manual blocking: missing parameters: %r", missing
            )
        if unknown:
            raise LoggedError(
                self.log, "Manual blocking: unknown parameters: %r", unknown
            )
        oversampling_factors = np.array(oversampling_factors)
        if np.all(oversampling_factors != np.sort(oversampling_factors)):
            self.log.warning(
                "Manual blocking: speed-blocking *apparently* non-optimal: "
                "oversampling factors must go from small (slow) to large (fast)."
            )
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
                if hasattr(theory, "get_auto_covmat"):
                    return theory.get_auto_covmat(params_info, self.info()["likelihood"])
        except Exception as e:
            self.log.warning("Something went wrong when looking for a covmat: %r", str(e))
            return None

    def set_timing_on(self, on):
        self.timing = on
        for component in self.components:
            component.set_timing_on(on)

    def measure_and_set_speeds(
        self, n=None, discard=1, max_tries=np.inf, random_state=None
    ):
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

        # Get proposal values for parameters to enable better perturbation from fixed refs
        proposal_scale = self.parameterization.get_sampled_params_proposals()

        n_done = 0
        with timing_on(self):
            while n_done < int(n) + int(discard):
                point = self.prior.reference(
                    random_state=random_state,
                    max_tries=max_tries,
                    ignore_fixed=True,
                    warn_if_no_ref=False,
                    override_std=proposal_scale,
                )
                if self.loglike(point, cached=False)[0] != -np.inf:  # type: ignore
                    n_done += 1
            self.mpi_debug("Computed %d points to measure speeds.", n_done)
            times = [
                component.timer.get_time_avg() or 0  # type: ignore
                for component in self.components
            ]
        if mpi.more_than_one_process():
            # average for different points
            times = np.average(mpi.allgather(times), axis=0)
        measured_speeds = [1 / (1e-7 + time) for time in times]
        self.mpi_info(
            "Setting measured speeds (per sec): %r",
            {
                component: float("%.3g" % speed)
                for component, speed in zip(self.components, measured_speeds)
            },
        )

        for component, speed in zip(self.components, measured_speeds):
            component.set_measured_speed(speed)


class DummyModel:
    """Dummy class for loading chains (e.g. for post processing)."""

    def __init__(self, info_params, info_likelihood, info_prior=None):
        self.parameterization = Parameterization(info_params, ignore_unused_sampled=True)
        self.prior = [prior_1d_name] + list(info_prior or [])
        self.likelihood = list(info_likelihood)


def get_model(
    info_or_yaml_or_file: InputDict | str | os.PathLike,
    debug: bool | None = None,
    stop_at_error: bool | None = None,
    packages_path: str | None = None,
    override: InputDict | None = None,
) -> Model:
    """
    Creates a :class:`model.Model`, from Cobaya's input (either as a dictionary, yaml file
    or yaml string). Input fields/options not needed (e.g. ``sampler``, ``output``,
    ``force``, ...) will simply be ignored.

    :param info_or_yaml_or_file: input options dictionary, yaml file, or yaml text
    :param debug: true for verbose debug output, or a specific logging level
    :param packages_path: path where external packages were installed
       (if external dependencies are present).
    :param stop_at_error: stop if an error is raised
    :param override: option dictionary to merge into the input one, overriding settings
       (but with lower precedence than the explicit keyword arguments)
    :return: a :class:`model.Model` instance.

    """
    flags = {
        packages_path_input: packages_path,
        "debug": debug,
        "stop_at_error": stop_at_error,
    }
    info = load_info_overrides(info_or_yaml_or_file, override or {}, **flags)
    logger_setup(info.get("debug"))
    # Inform about ignored info keys
    ignored_info = []
    for k in list(info):
        if k not in {
            "params",
            "likelihood",
            "prior",
            "theory",
            "packages_path",
            "timing",
            "stop_at_error",
            "auto_params",
            "debug",
        }:
            value = info.pop(k)  # type: ignore
            if value is not None and (not isinstance(value, Mapping) or value):
                ignored_info.append(k)
    # Create the updated input information, including defaults for each component.
    updated_info = update_info(info)
    if ignored_info:
        get_logger(__name__).warning("Ignored blocks/options: %r", ignored_info)
    if is_debug():
        get_logger(__name__).debug(
            "Input info updated with defaults (dumped to YAML):\n%s",
            yaml_dump(sort_cosmetic(updated_info)),
        )
    # Initialize the parameters and posterior
    return Model(
        updated_info["params"],
        updated_info["likelihood"],
        updated_info.get("prior"),
        updated_info.get("theory"),
        packages_path=info.get("packages_path"),
        timing=updated_info.get("timing"),
        stop_at_error=info.get("stop_at_error", False),
    )
