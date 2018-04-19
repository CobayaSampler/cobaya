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
from time import sleep
import numpy as np
import inspect
from itertools import chain
from fractions import gcd

# Local
from cobaya.conventions import _external, _theory, _params, _overhead
from cobaya.tools import get_class, get_external_function
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__.split(".")[-1])

# Default options for all subclasses
class_options = {"speed": -1}


class Likelihood(object):
    """Likelihood class prototype."""

    # Generic initialization -- do not touch
    def __init__(self, info, parametrization):
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        # Load info of the likelihood
        for k in info:
            setattr(self, k, info[k])
        # *Mock* likelihoods: gather all parameters starting with `prefix`
        if self.is_mock():
            all_params = (list(parametrization.input_params()) +
                          list(parametrization.output_params()))
            info[_params] = [p for p in all_params
                             if p.startswith(self.prefix or "")]
        # Load parameters
        self.input_params = odict(
            [(p,p_info) for p,p_info in parametrization.input_params().items()
             if p in info[_params]])
        self.output_params = odict(
            [(p,p_info) for p,p_info in parametrization.output_params().items()
             if p in info[_params]])
        # Initialise
        self.initialise()

    # What you *must* implement to create your own likelihood:

    # Optional
    def initialise(self):
        """Initialises the specifics of this likelihood."""
        pass

    # Optional
    def add_theory(self):
        """Performs any necessary initialisation on the theory side,
        e.g. requests observables"""
        pass

    # Mandatory
    def logp(self, **params_values):
        """
        Computes and returns the log likelihood value.
        Takes as keyword arguments the parameters values.
        To get the derived parameters, pass a `derived` keyword with an empty dictionary.
        """
        return None

    # Optional
    def close(self):
        """Finalises the likelihood, if something needs to be done."""
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
        log.error("Exact marginal likelihood not defined.")
        raise HandledException

    # Other general methods

    def wait(self):
        if self.delay:
            log.debug("Sleeping for %f seconds.", self.delay)
        sleep(self.delay)

    def d(self):
        return len(self.input_params)

    def is_mock(self):
        return hasattr(self, "prefix")


class LikelihoodExternalFunction(Likelihood):
    def __init__(self, name, info, parametrization, theory=None):
        self.name = name
        # Load info of the likelihood
        for k in info:
            setattr(self, k, info[k])
        # Store the external function and its arguments
        self.external_function = get_external_function(info[_external], name=self.name)
        argspec = inspect.getargspec(self.external_function)
        self.input_params = odict(
            [(p, None) for p in argspec.args
             if p not in ["derived", "theory"] and p in parametrization.input_params()])
        self.has_derived = "derived" in argspec.args
        if self.has_derived:
            derived_kw_index = argspec.args[-len(argspec.defaults):].index("derived")
            self.output_params = argspec.defaults[derived_kw_index]
        else:
            self.output_params = []
        self.has_theory = "theory" in argspec.args
        if self.has_theory:
            theory_kw_index = argspec.args[-len(argspec.defaults):].index("theory")
            self.needs = argspec.defaults[theory_kw_index]

    def add_theory(self):
        if self.has_theory:
            self.theory.needs(self.needs)

    def logp(self, **params_values):
        # if no derived params defined in the external call, delete the "derived" argument
        if not self.has_derived:
            params_values.pop("derived")
        if self.has_theory:
            params_values["theory"] = self.theory
        try:
            return self.external_function(**params_values)
        except:
            log.error("".join(
                ["-"]*16 + ["\n\n"] + list(traceback.format_exception(*sys.exc_info())) +
                ["\n"] + ["-"]*37))
            log.error("The external likelihood '%s' failed at evaluation. "
                      "See traceback on top of this message.", self.name)
            raise HandledException


class LikelihoodCollection(object):
    """
    Likelihood manager:
    Initialises the theory code and the experimental likelihoods.
    """

    def __init__(self, info_likelihood, parametrization, info_theory=None):
        # Store the input/output parameters
        self.input_params = parametrization.input_params()
        self.output_params = parametrization.output_params()
        # Initialise individual Likelihoods
        self._likelihoods = odict()
        for name, info in info_likelihood.items():
            # If it does "external" key, wrap it up. Else, load it up
            if _external in info:
                self._likelihoods[name] = LikelihoodExternalFunction(
                    name, info, parametrization, theory=getattr(self, _theory, None))
            else:
                lik_class = get_class(name)
                self._likelihoods[name] = lik_class(info, parametrization)
        # Check that all are recognized
        requested_not_known = {}
        for params in ("input_params", "output_params"):
            info = getattr(parametrization, params)()
            setattr(self, params, info)
            requested = set(info)
            known = set(chain(*[getattr(self[lik], params) for lik in self]))
            requested_not_known[params] = requested.difference(known)
            if requested_not_known[params]:
                if info_theory:
                    log.debug("The following %s parameters are not recognised by any "
                              "likelihood, and will be passed to the theory code: %r",
                              params.split("_")[0], requested_not_known[params])
                else:
                    log.error("Some of the requested %s parameters were not recognized "
                              "by any likelihood: %r",
                              params.split("_")[0], requested_not_known[params])
                    raise HandledException
        # *IF* there is a theory code, initialise it
        if info_theory:
            input_params_theory = self.input_params.fromkeys(
                requested_not_known["input_params"])
            output_params_theory = self.output_params.fromkeys(
                requested_not_known["output_params"])
            name, fields = list(info_theory.items())[0]
            theory_class = get_class(name, kind=_theory)
            self.theory = theory_class(input_params_theory, output_params_theory, fields)
        else:
            self.theory = None
        for lik in self:
            self[lik].theory = self.theory
            self[lik].add_theory()
        # Store the input params and likelihods on which each sampled params depends.
        # Theory parameters "depend" on every likelihood, since re-computing the theory
        # code forces recomputation of the likelihoods
        self.sampled_input_dependence = parametrization.sampled_input_dependence()
        self.sampled_lik_dependence = odict(
            [[p,[lik for lik in list(self)+([_theory] if self.theory else [])
                 if any([(i in self[lik].input_params) for i in (i_s or [p])])]]
             for p,i_s in self.sampled_input_dependence.items()])
        # Theory parameters "depend" on every likelihood, since re-computing the theory
        # code forces recomputation of the likelihoods
        for p, ls in self.sampled_lik_dependence.items():
            if _theory in ls:
                self.sampled_lik_dependence[p] = ([_theory] + list(self._likelihoods))
        # Pop the per-likelihood parameters info, that was auxiliary
        for lik in info_likelihood:
            info_likelihood[lik].pop(_params)

    # "get" and iteration operate over the dictionary of likelihoods
    # notice that "get" can get "theory", but the iterator does not!
    def __getitem__(self, key):
        return self._likelihoods.__getitem__(key) if key != _theory else self.theory

    def __iter__(self):
        return self._likelihoods.__iter__()

    def logps(self, input_params, derived=None):
        """
        Computes observables and returns the (log) likelihoods *separately*.
        It takes a list of **input** parameter values, in the same order as they appear
        in the `OrderedDictionary` of the :class:LikelihoodCollection.
        To compute the derived parameters, it takes an optional keyword `derived` as an
        empty list, which is then populated with the derived parameters values.
        """
        log.debug("Got input parameters: %r", input_params)
        # Prepare the likelihood-defined derived parameters (only computed if requested)
        # Notice that they are in general different from the sampler-defined ones.
        derived_dict = {}
        # If theory code present, compute the necessary products
        if self.theory:
            this_params_dict = {p: input_params[p] for p in self.theory.input_params}
            success = self.theory.compute(
                derived=(derived_dict if derived is not None else None),
                **this_params_dict)
            if not success:
                if derived is not None:
                    derived += [np.nan]*len(self.output_params)
                return np.array([-np.inf for _ in self])
        # Compute each log-likelihood, and optionally get the respective derived params
        logps = []
        for lik in self:
            this_params_dict = {p: input_params[p] for p in self[lik].input_params}
            this_derived_dict = {} if derived is not None else None
            logps += [self[lik].logp(derived=this_derived_dict, **this_params_dict)]
            derived_dict.update(this_derived_dict or {})
            log.debug("'%s' evaluated to logp=%g with params %r, and got derived %r",
                      lik, logps[-1], this_params_dict, this_derived_dict)
        # Turn the derived params dict into a list and return
        if derived is not None:
            derived += [derived_dict[p] for p in self.output_params]
            log.debug("Produced derived parameters: %r", derived_dict)
        return np.array(logps)

    def logp(self, input_params, derived=None):
        """
        Computes observables and returns the (log) likelihood.
        It takes a list of **sampled** parameter values, in the same order as they appear
        in the `OrderedDictionary` of the :class:LikelihoodCollection.
        To compute the derived parameters, it takes an optional keyword `derived` as an
        empty list, which is then populated with the derived parameters values.
        """
        return np.sum(self.logps(input_params, derived=derived))

    def marginal(self, *args):
        log.error("Marginal not implemented for >1 likelihoods. Sorry!")
        raise HandledException

    def speeds_of_params(self, int_speeds=False, fast_slow=False):
        """
        Separates the sampled parameters in blocks according to the likelihood (or theory)
        re-evaluation that changing each one of them involves. Using the appoximate speed
        (i.e. inverse evaluation time in seconds) of each likelihood, sorts the blocks in
        a nearly optimal way, in ascending order of speed *per step/parameter*.

        Returns tuples of ``(speeds), (params_inblock)``, sorted by ascending speeds.

        If ``int_speeds=True``, returns integer speeds (i.e. oversampling factors
        *per param* -- not per block) instead of speeds in 1/s.

        If ``fast_slow=True``, returns just 2 blocks: a fast and a slow one, each one
        assigned its slowest speed.

        TODO: take into account "mixing towards fastest" introduced by the Cholesky
        transformation, and sort fully optimally.
        """
        # Fill unknown speeds with the value of the slowest one, and clip with overhead
        speeds = np.array([getattr(self[lik], "speed", -1) for lik in self] +
                          ([getattr(self.theory, "speed", -1)] if self.theory else []),
                          dtype=float)
        # Add overhead to the defined ones, and clip to the slowest the undefined ones
        speeds[speeds > 0] = (speeds[speeds > 0]**-1 + _overhead)**-1
        try:
            speeds = np.clip(speeds, min(speeds[speeds > 0]), None)
        except ValueError:
            # No speeds specified
            speeds = np.ones(len(speeds))
        liks = list(self) + ([_theory] if self.theory else [])
        for i, lik in enumerate(liks):
            self[lik].speed = speeds[i]
        # Group params by "footprint"
        footprints = np.zeros((len(self.sampled_lik_dependence), len(liks)), dtype=int)
        for i, ls in enumerate(self.sampled_lik_dependence.values()):
            for j, lik in enumerate(liks):
                footprints[i,j] = lik in ls
        different_footprints = list(set([tuple(row) for row in footprints.tolist()]))
        blocks = [[p for ip, p in enumerate(self.sampled_lik_dependence)
                   if all(footprints[ip] == fp)] for fp in different_footprints]
        # Compute "intrinsic" time cost of block (i.e. before mixing towards fastest)
        blocks_costs = [sum([i*s**-1 for i,s in zip(fpblock,speeds)])
                        for fpblock in different_footprints]
        blocks_speeds = np.array([1/c for c in blocks_costs])
        # Separate fast-slow
        # WIP: just theory vs non-theory yet
        if fast_slow and self.theory:
            blocks_fs = [[], []]
            speeds_fs = [[], []]
            for i in range(len(blocks)):
                if different_footprints[i][-1]:  # is theory param
                    blocks_fs[0] += [blocks[i]]
                    speeds_fs[0] += [blocks_speeds[i]]
                else:
                    blocks_fs[1] += [blocks[i]]
                    speeds_fs[1] += [blocks_speeds[i]]
            blocks = blocks_fs[0] + blocks_fs[1]
            blocks_speeds = speeds_fs[0] + speeds_fs[1]
        # Sort *naively*: less cost per paramter first
        isort = np.argsort(blocks_speeds)
        # WIP: ideally, when likelihoods have states, one would sort even *inside* each of
        # the fast-slow blocks!
        if int_speeds:
            # Oversampling precision: smallest difference in oversampling to be ignored.
            speed_precision = 1/20
            speeds = np.array(np.round(np.array(
                blocks_speeds)/min(blocks_speeds)/speed_precision), dtype=int)
            speeds = np.array(
                speeds/np.ufunc.reduce(np.frompyfunc(gcd, 2, 1), speeds), dtype=int)
            blocks_speeds = speeds
        return zip(*[[blocks_speeds[i], blocks[i]] for i in isort])

    # Python magic for the "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for lik in self:
            self[lik].close()


# class LikelihoodCollection_MPI(LikelihoodCollection):
#     """
#     MPI wrapper around the LikelihoodCollection class.
#     """
#     def __init__(self, info_likelihood, info_params, info_theory=None):
#         from mpi4py import MPI
#         comm = MPI.COMM_WORLD
#         rank = comm.Get_rank()
#         to_broadcast = ("theory", "likelihoods", "fixed", "sampled", "derived",
#                         "_updated_info_params_liks")
#         if rank == 0:
#             LikelihoodCollection.__init__(
#                 self, info_likelihood, info_params, info_theory=info_theory)
#         else:
#             for var in to_broadcast:
#                 setattr(self, var, None)
#         for var in to_broadcast:
#             setattr(self, var, comm.bcast(getattr(self, var), root=0))
