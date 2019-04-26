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
from copy import deepcopy
from collections import namedtuple

# Local
from cobaya.conventions import _likelihood, _prior, _params, _theory, _timing
from cobaya.conventions import _path_install, _debug, _debug_default, _debug_file
from cobaya.input import get_full_info
from cobaya.parameterization import Parameterization
from cobaya.prior import Prior
from cobaya.likelihood import LikelihoodCollection as Likelihood
from cobaya.log import HandledException, logger_setup
from cobaya.yaml import yaml_dump

# Logger
import logging

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
    # Just a dummy import before configuring the logger, until I fix root/individual level
    import getdist
    logger_setup(info.pop(_debug, _debug_default), info.pop(_debug_file, None))
    # Create the full input information, including defaults for each module.
    info = deepcopy(info)
    ignored_info = {}
    for k in list(info):
        if k not in [_params, _likelihood, _prior, _theory, _path_install, _timing]:
            ignored_info.update({k: info.pop(k)})
    import logging
    if ignored_info:
        logging.getLogger(__name__.split(".")[-1]).warn(
            "Ignored blocks/options: %r", list(ignored_info))
    full_info = get_full_info(info)
    if logging.root.getEffectiveLevel() <= logging.DEBUG:
        logging.getLogger(__name__.split(".")[-1]).debug(
            "Input info updated with defaults (dumped to YAML):\n%s",
            yaml_dump(full_info))
    # Initialize the posterior and the sampler
    return Model(full_info[_params], full_info[_likelihood], full_info.get(_prior),
                 full_info.get(_theory), modules=info.get(_path_install),
                 timing=full_info.get(_timing))


class Model(object):
    """
    Class containing all the information necessary to compute the unnormalized posterior.

    Allows for low-level interaction with the theory code, prior and likelihood.

    **NB:** do not initialize this class directly; use :func:`~model.get_model` instead,
    with some info as input.
    """

    def __init__(self, info_params, info_likelihood, info_prior=None, info_theory=None,
                 modules=None, timing=None, allow_renames=True):
        self.log = logging.getLogger(self.__class__.__name__)
        self._full_info = {
            _params: deepcopy(info_params),
            _likelihood: deepcopy(info_likelihood)}
        if not self._full_info[_likelihood]:
            self.log.error("No likelihood requested!")
            raise HandledException
        for like in self._full_info[_likelihood].values():
            like.pop(_params)
        for k, v in ((_prior, info_prior), (_theory, info_theory),
                     (_path_install, modules), (_timing, timing)):
            if v not in (None, {}):
                self._full_info[k] = deepcopy(v)
        self.parameterization = Parameterization(info_params, allow_renames=allow_renames)
        self.prior = Prior(self.parameterization, info_prior)
        self.likelihood = Likelihood(info_likelihood, self.parameterization, info_theory,
                                     modules=modules, timing=timing)

    def info(self):
        """
        Returns a copy of the information used to create the model, including defaults.
        """
        return deepcopy(self._full_info)

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
                self.log.error("Wrong dimensionality: it's %d and it should be %d.",
                               len(params_values_array), self.prior.d())
                raise HandledException
        if len(params_values_array.shape) >= 2:
            self.log.error("Cannot take arrays of points as inputs, just single points.")
            raise HandledException
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
        _derived = [] if return_derived else None
        loglikes = self.likelihood.logps(
            self.parameterization._to_input(params_values),
            _derived=_derived, cached=cached)
        if make_finite:
            loglikes = np.nan_to_num(loglikes)
        if return_derived:
            derived_sampler = self.parameterization._to_derived(_derived)
            if self.log.getEffectiveLevel() <= logging.DEBUG:
                self.log.debug(
                    "Computed derived parameters: %s",
                    dict(zip(self.parameterization.derived_params(), derived_sampler)))
            return loglikes, derived_sampler
        return loglikes

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
            self.log.error(
                "Got non-finite parameter values: %r",
                dict(zip(self.parameterization.sampled_params(), params_values_array)))
            raise HandledException
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

    # Python magic for the "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type=None, exception_value=None, traceback=None):
        self.likelihood.__exit__(exception_type, exception_value, traceback)

    def close(self):
        self.__exit__()
