"""
.. module:: likelihood

:Synopsis: Likelihood class and likelihood manager
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
from time import sleep
import numpy as np
import six
from copy import deepcopy

# Local
from cobaya.conventions import _external
from cobaya.conventions import _chi2, _separator, _likelihood, _self_name
from cobaya.conventions import _module_path
from cobaya.tools import get_class, get_external_function, getfullargspec
from cobaya.log import LoggedError
from cobaya.component import CobayaComponent, ComponentCollection


class Likelihood(CobayaComponent):
    """Likelihood base class."""

    class_options = {"speed": -1, "stop_at_error": False}

    # Generic initialization -- do not touch
    def __init__(self, info={}, name=None, timing=None, path_install=None, standalone=True):
        name = name or self.get_qualified_class_name()
        if standalone:
            # TODO: would probably be more natural if defaults were already read here
            default_info = self.get_defaults()
            if _likelihood in default_info:
                default_info = default_info[_likelihood][
                    name if _self_name not in default_info[_likelihood] else _self_name]
            default_info.update(info)
            info = default_info
        super(Likelihood, self).__init__(info, name=name, timing=timing, path_install=path_install)
        # States, to avoid recomputing
        self._n_states = 3
        self._states = [{"params": None, "logp": None, "_derived": None,
                         "theory_params": None, "last": 0}
                        for _ in range(self._n_states)]
        if standalone:
            self.initialize()

    # Optional
    def initialize(self):
        """
        Initializes the specifics of this likelihood.
        Note that at this point we know `the `self.input_params``
        and the ``self.output_params`` if run from Cobaya.
        """
        pass

    # Optional
    def get_requirements(self):
        """
        Get a dictionary of requirements to request from the theory
        :return: dictionary of requirements
        """
        return {}

    # Optional
    def add_theory(self):
        """Performs any necessary initialization on the theory side,
        e.g. requests observables. By default just call get_requirements and pass to theory"""
        needs = self.get_requirements()
        if needs:
            self.theory.needs(**needs)

    # Mandatory
    def logp(self, **params_values):
        """
        Computes and returns the log likelihood value.
        Takes as keyword arguments the parameter values.
        To get the derived parameters, pass a `_derived` keyword with an empty dictionary.
        """
        return None

    # What you *can* implement to create your own likelihood:

    def marginal(self, directions=None, params_values=None):
        """
        (For analytic likelihoods only.)
        Computes the marginal likelihood.
        If nothing is specified, returns the total marginal likelihood.
        If some directions are specified (as a list, tuple or array), returns the marginal
        likelihood pdf over those directions evaluated at the given parameter values.
        """
        raise LoggedError(self.log, "Exact marginal likelihood not defined.")

    # Other general methods

    def _logp_cached(self, theory_params=None, cached=True, _derived=None, **params_values):
        """
        Wrapper for the `logp` method that caches logp's and derived params.
        If the theory products have been re-computed, re-computes the likelihood anyway.
        """
        params_values = deepcopy(params_values)
        self.log.debug("Got parameters %r", params_values)
        lasts = [self._states[i]["last"] for i in range(self._n_states)]
        try:
            if not cached:
                raise StopIteration
            # Are the parameter values there already?
            i_state = next(i for i in range(self._n_states)
                           if self._states[i]["params"] == params_values)
            # StopIteration not raised, so state exists, but maybe the theory params have
            # changed? In that case, I would still have to re-compute the likelihood
            if self._states[i_state]["theory_params"] != theory_params:
                self.log.debug("Recomputing logp because theory params changed.")
                raise StopIteration
            if _derived is not None:
                _derived.update(self._states[i_state]["derived"] or {})
            self.log.debug("Re-using computed results.")
        except StopIteration:
            # update the (first) oldest one and compute
            i_state = lasts.index(min(lasts))
            self._states[i_state]["params"] = params_values
            self._states[i_state]["theory_params"] = deepcopy(theory_params)
            if self.timer:
                self.timer.start()
            try:
                self._states[i_state]["logp"] = self.logp(_derived=_derived, **params_values)
                if self.timer:
                    self.timer.increment()
            except Exception as e:
                if self.stop_at_error:
                    raise LoggedError(self.log, "Error at evaluation: %r", e)
                else:
                    self.log.debug(
                        "Ignored error at evaluation and assigned 0 likelihood "
                        "(set 'stop_at_error: True' as an option for this likelihood to stop here). "
                        "Error message: %r", e)
                    self._states[i_state]["logp"] = -np.inf
            self._states[i_state]["derived"] = deepcopy(_derived)
        # make this one the current one by decreasing the antiquity of the rest
        for i in range(self._n_states):
            self._states[i]["last"] -= max(lasts)
        self._states[i_state]["last"] = 1
        self.log.debug("Evaluated to logp=%g with derived %r",
                       self._states[i_state]["logp"], self._states[i_state]["derived"])
        return self._states[i_state]["logp"]

    def wait(self):
        if self.delay:
            self.log.debug("Sleeping for %f seconds.", self.delay)
        sleep(self.delay)

    def d(self):
        """
        Dimension of the input vector.

        NB: Different from dimensionality of the sampling problem, e.g. this may include
        fixed input parameters.
        """
        return len(self.input_params)


class LikelihoodExternalFunction(Likelihood):
    def __init__(self, info, name, timing=None):
        CobayaComponent.__init__(self, info, name=name, timing=timing)
        # Store the external function and its arguments
        self.external_function = get_external_function(info[_external], name=name)
        argspec = getfullargspec(self.external_function)
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
        # States, to avoid recomputing
        self._n_states = 3
        self._states = [{"params": None, "logp": None, "derived": None, "last": 0}
                        for _ in range(self._n_states)]
        self.log.info("Initialized external likelihood.")

    def get_requirements(self):
        return self.needs if self.has_theory else {}

    def logp(self, **params_values):
        # if no derived params defined in external func, delete the "_derived" argument
        if not self.has_derived:
            params_values.pop("_derived")
        if self.has_theory:
            params_values["_theory"] = self.theory
        try:
            return self.external_function(**params_values)
        except Exception as ex:
            if isinstance(ex, LoggedError):
                # Assume proper error info was written before raising LoggedError
                pass
            else:
                # Print traceback
                self.log.error("".join(
                    ["-"] * 16 + ["\n\n"] +
                    list(traceback.format_exception(*sys.exc_info())) | +
                    ["\n"] + ["-"] * 37))
            raise LoggedError(
                self.log, "The external likelihood '%s' failed at evaluation. "
                          "See error info on top of this message.", self.get_name())


class LikelihoodCollection(ComponentCollection):
    """
    Manages list of experimental likelihoods.
    """

    def __init__(self, info_likelihood, path_install=None, timing=None, theory=None):
        super(LikelihoodCollection, self).__init__()
        self.set_logger("likelihood")
        self.theory = theory
        # Get the individual likelihood classes
        for name, info in info_likelihood.items():
            # If it has an "external" key, wrap it up. Else, load it up
            if _external in info:
                self[name] = LikelihoodExternalFunction(info, name, timing=timing)
            else:
                like_class = get_class(name, kind=_likelihood, module_path=info.pop(_module_path, None))
                self[name] = like_class(info, path_install=path_install, timing=timing, standalone=False, name=name)

    def logps(self, input_params, theory_params_dict=None, derived_dict=None, cached=True):
        """
        Computes observables and returns the (log) likelihoods *separately*.
        It takes a list of **input** parameter values, in the same order as they appear
        in the `OrderedDictionary` of the :class:`LikelihoodCollection`.
        To compute the derived parameters, it takes an optional keyword `derived_dict` as an
        empty list, which is then populated with the derived parameter values.
        """
        self.log.debug("Got input parameters: %r", input_params)
        # Prepare the likelihood-defined derived parameters (only computed if requested)
        # Notice that they are in general different from the sampler-defined ones.
        # Compute each log-likelihood, and optionally get the respective derived params
        logps = []
        want_derived = derived_dict is not None
        this_derived_dict = {} if want_derived else None
        for like in self.values():
            this_params_dict = {p: input_params[p] for p in like.input_params}
            logps += [like._logp_cached(theory_params=theory_params_dict,
                                        _derived=this_derived_dict, cached=cached, **this_params_dict)]
            if want_derived:
                if this_derived_dict:
                    derived_dict.update(this_derived_dict)
                    this_derived_dict.clear()
                derived_dict[_chi2 + _separator + like.get_name().replace(".", "_")] = -2 * logps[-1]
        return np.array(logps)
