"""
.. module:: likelihood

:Synopsis: Likelihood class and likelihood collection
:Author: Jesus Torrado and Antony Lewis

This module defines the main :class:`Likelihood` class, from which every likelihood
usually inherits, and the :class:`LikelihoodCollection` class, which groups and manages
all the individual likelihoods.

Likelihoods inherit from :class:`.theory.Theory`, adding an additional method
to return the likelihood. As with all theories, likelihoods cache results, and the
function :meth:`LikelihoodInterface.get_current_logp` is used by :class:`model.Model` to
calculate the total likelihood. The default Likelihood implementation does the actual
calculation of the log likelihood in the `logp` function, which is then called
by :meth:`Likelihood.calculate` to save the result into the current state.

Subclasses typically just provide the `logp` function to define their likelihood result,
and use :meth:`theory.Theory.get_requirements` to specify which inputs are needed from
other theory codes (or likelihoods). Other methods of the :class:`.theory.Theory` base
class can be used as and when needed.

"""
# Python 2/3 compatibility
from __future__ import absolute_import, division

# Global
import sys
import traceback
from time import sleep
import numpy as np
import inspect
import six

# Local
from cobaya.conventions import kinds, _external, _module_path
from cobaya.tools import get_class, get_external_function, getfullargspec
from cobaya.log import LoggedError, always_stop_exceptions
from cobaya.component import ComponentCollection
from cobaya.theory import Theory


class LikelihoodInterface(object):
    """
    Interface function for likelihoods. Can descend from a :class:`theory.Theory` class
    and this to make a likelihood (where the calculate() method stores state['logp'] for
    the current parameters), or likelihoods can directly inherit from :class:`Likelihood`
    instead.

    The get_current_logp function returns the current state's logp, and does not normally
    need to be changed.
    """

    def get_current_logp(self):
        """
        Gets log likelihood for the current point

        :return:  log likelihood from the current state
        """
        return self._current_state["logp"]


class Likelihood(Theory, LikelihoodInterface):
    """Likelihood base class. Extends from :class:`LikelihoodInterface` and the general
    :class:`theory.Theory` class by adding functions to return likelihoods functions
    (logp function for a given point)."""

    def __init__(self, info={}, name=None, timing=None, path_install=None,
                 initialize=True, standalone=True):
        super(Likelihood, self).__init__(info, name=name, timing=timing,
                                         path_install=path_install, initialize=initialize,
                                         standalone=standalone)
        # Make sure `types` is a list of data types, for aggregated chi2
        self.type = (lambda x: [x] if isinstance(x, six.string_types) else x)(
            getattr(self, "type", []))

    @property
    def theory(self):
        # for backwards compatibility
        return self.provider

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
        derived = {} if want_derived else None
        state["logp"] = -np.inf  # in case of exception
        state["logp"] = self.logp(_derived=derived, **params_values_dict)
        if want_derived:
            state["derived"] = derived.copy()

    def wait(self):
        if self.delay:
            self.log.debug("Sleeping for %f seconds.", self.delay)
        sleep(self.delay)


class AbsorbUnusedParamsLikelihood(Likelihood):
    pass


class LikelihoodExternalFunction(Likelihood):
    def __init__(self, info, name, timing=None):
        Theory.__init__(self, info, name=name, timing=timing, standalone=False)

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
            self._needs = argspec.defaults[theory_kw_index]

        self.log.info("Initialized external likelihood.")

    def get_requirements(self):
        return self._needs if self.has_theory else {}

    def logp(self, **params_values):
        # if no derived params defined in external func, delete the "_derived" argument
        if not self.has_derived:
            params_values.pop("_derived")
        if self.has_theory:
            params_values["_theory"] = self.provider
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
    A dictionary storing experimental likelihood :class:`Likelihood` instances index
    by their names.
    """

    def __init__(self, info_likelihood, path_install=None, timing=None, theory=None):
        super(LikelihoodCollection, self).__init__()
        self.set_logger("likelihood")
        self.theory = theory
        # Get the individual likelihood classes
        for name, info in info_likelihood.items():
            if isinstance(name, Theory):
                name, info = name.get_name(), info
            if isinstance(info, Theory):
                self[name] = info
            elif _external in info:
                if isinstance(info[_external], Theory):
                    self[name] = info[_external]
                elif inspect.isclass(info[_external]):
                    if not hasattr(info[_external], "get_current_logp") or \
                            not issubclass(info[_external], Theory):
                        raise LoggedError(self.log, "external class likelihoods must be "
                                                    "a subclass of Theory and have"
                                                    "logp, get_current_logp functions")
                    self[name] = info[_external](info, path_install=path_install,
                                                 timing=timing,
                                                 standalone=False, name=name)
                else:
                    # If it has an "external" key, wrap it up. Else, load it up
                    self[name] = LikelihoodExternalFunction(info, name, timing=timing)
            else:
                like_class = get_class(name, kind=kinds.likelihood,
                                       module_path=info.pop(_module_path, None))
                self[name] = like_class(info, path_install=path_install, timing=timing,
                                        standalone=False, name=name)

            if not hasattr(self[name], "get_current_logp"):
                raise LoggedError(self.log, "'Likelihood' %s is not actually a "
                                            "likelihood (no get_current_logp function)",
                                  name)
