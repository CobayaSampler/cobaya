"""
.. module:: likelihood

:Synopsis: Prototype likelihood and likelihood manager
:Author: Jesus Torrado

This module defines the main :class:`Likelihood` class, from which every likelihood
inherits, and the :class:`LikelihoodCollection` class, which groups and manages all the
individual likelihoods and is the actual instance passed to the sampler. Additionally, it
defines an MPI wrapper for the last one, that initialises the likelihoods once and passes
a copy to each MPI slave (rank>1) process.

Input: specifying likelihoods to explore
----------------------------------------

Likelihoods are specified under the `likelihood` block, together with their options:

.. code-block:: yaml

   likelihood:
     [likelihood 1]:
        [likelihood 1 option 1]: [value 1]
        [...]
     [likelihood 2]:
        [likelihood 2 option 2]: [value 2]
        [...]

Likelihood parameters are specified within the ``params`` block, as explained in
:doc:`params_prior`.

Code conventions and defauls
----------------------------

Each likelihood *lives* in a folder with its name under the ``likelihoods`` folder of the
source tree. In that folder, there must be at least *three* files:

- A trivial ``__init__.py`` file containing a single line: ``from [name] import [name]``,
  where ``name`` is the name of the likelihood, and it's folder.
- A ``name.py`` file, containing the particular class definition of the likelihood, 
  inheriting from the :class:`Likelihood` class (see below).
- A ``defaults.yaml`` containing a block:

  .. code-block:: yaml

     likelihood:
       [name]:
         [option 1]: [value 1]
         [...]
       params:
         [param 1]:
           prior:
             [prior info]
           [label, ref, etc.]

  The options and parameters defined in this file are the only ones recognised by the
  likelihood, and they are loaded automatically with their default values (options) and
  priors (parameters) by simply mentioning the likelihood in the input file, where one can
  re-define any of those options with a different value. The same parameter may be
  defined by different likelihoods -- in those cases, it needs to have the same default 
  information (prior, label, etc.) in the defaults file of those likelihoods.

.. note::

   Some *mock* likelihoods can have any number of non-predefined parameters, as long as
   they start with a certain prefix specified by the user with the option ``mock_prefix``
   of said likelihood.

.. note::

   Actually, there are some user-defined options that are common to all likelihoods and
   do not need to be specified in the ``defaults.yaml`` file, such as the computational
   ``speed`` of the likelihood (see :ref:`mcmc_speed_hierarchy`).


Creating your own likelihood
----------------------------

Since cobaya was created to be flexible, creating your own likelihood is very easy: simply
create a folder with its name under ``likelihoods`` in the source tree and follow the 
conventions explained above. Inside the class definition of your function, you can use any
of the attributes defined in the ``defaults.yaml`` file directly, and you only need to 
specify one, or at most three, functions
(see the :class:`Likelihood` class documentation below):

- A ``logp`` function taking a dictionary of (sampled) parameter values and returning a
  log-likelihood.
- An (optional) ``initialise`` function preparing any computation, importing any necessary
  code, etc.
- An (optional) ``close`` function doing whatever needs to be done at the end of the
  sampling (e.g. releasing memory).

You can use the :doc:`Gaussian likelihood <likelihood_gaussian>` as a guide.
If your likelihood needs a cosmological code, just define one in the input file and you
can automatically access it as an attribute of your class: ``[your_likelihood].theory``.
Use the :doc:`Planck likelihood <likelihood_planck>` as a guide to create your own
cosmological likelihood.

.. note:: ``theory`` and ``derived`` are reserved parameter names: you cannot use them! 

"""
# Python 2/3 compatibility
from __future__ import absolute_import, division

# Global
import sys
import os
from importlib import import_module
from collections import OrderedDict as odict
from time import sleep
import numpy as np
from scipy import stats
import inspect
from itertools import chain

# Local
from cobaya.conventions import _external, _theory, _params
from cobaya.tools import get_class, get_external_function
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)


class Likelihood():
    """Likelihood class prototype."""

    # Generic initialisation -- do not touch
    def __init__(self, info, parametrisation, theory=None):
        self.input_params = odict(
            [(p,p_info) for p,p_info in parametrisation.input_params().iteritems() if p in info[_params]])
        self.output_params = odict(
            [(p,p_info) for p,p_info in parametrisation.output_params().iteritems() if p in info[_params]])
        # Load info of the likelihood
        for k in info:
            setattr(self, k, info[k])
        # Initialise
        self.theory = theory
        self.initialise()
    
    # What you *must* implement to create your own likelihood:

    # Optional
    def initialise(self):
        """Initialises the specifics of this likelihood."""
        pass

    # Mandatory
    def logp(self, **params_values):
        """
        Computes and returns the likelihood value.
        Takes as keyword arguments the parameters values.
        To get the derived parameters, pass a `derived` keyword with an empty dictionary.
        """
        return None

    # Optional
    def close(self):
        """Finalises the likelihood, if something needs to be done."""
        pass

    # What you *can* implement to create your own *mock* likelihood:

    def marginal(self, directions=None, params_values=None):
        """
        (For mock likelihoods only.)
        Computes the marginal likelihood.
        If nothing is specified, returns the total marginal likelihood.
        If some directions are specified (as a list, tuple or array), returns the marginal
        likelihood pdf over those directions evaluated at the given parameter values.
        """
        log.error("Exact marginal likelihood not defined.")
        raise HandledException

    # Other general methods

    def params_defaults(self):
        return self._params_defaults

    def wait(self):
        if self.delay:
            log.debug("Sleeping for %f seconds.", self.delay)
        sleep(self.delay)
    
    # def data_folder(self):
    #     """
    #     Default data folder: "data" under the likelihood folder.
    #     Creates it if it does not exist.
    #     Can be overriden in your likelihood if a different, external folder is needed.
    #     """
    #     data_folder = os.path.join(
    #         os.path.dirname(sys.modules[self.__module__].__file__), "data")
    #     if not os.path.exists(data_folder):
    #         os.makedirs(data_folder)
    #     return data_folder

    def is_mock(self):
        return hasattr(self, "mock_prefix")
    
    # def mock_set_params(self, fixed=None, sampled=None, derived=None):
    #     """
    #     Set parameters of a mock likelihood after initialisation (does nothing for non-mock
    #     likelihoods).

    #     Takes a list of parameter names.
    #     """
    #     self.fixed, self.sampled, self.derived = fixed, sampled, derived


class LikelihoodExternalFunction(Likelihood):
    def __init__(self, name, info, theory=None):
        self.theory = theory
        # Store the external function and its arguments
        self.external_function = get_external_function(info[_external])
        argspec = inspect.getargspec(self.external_function)
        self.input_params = odict([(p, None) for p in argspec.args if p!="derived"])
        self.has_derived = "derived" in argspec.args
        if self.has_derived:
            derived_kw_index = argspec.args[-len(argspec.defaults):].index("derived")
            self.output_params = argspec.defaults[derived_kw_index]
        else:
            self.output_params = []

    def logp(self, **params_values):
        # if not derived params defined in the external call, delete the "derived" argument
        if not self.has_derived:
            params_values.pop("derived")
        return self.external_function(**params_values)


class LikelihoodCollection():
    """
    Likelihood manager:
    Initialises the theory code and the experimental likelihoods.
    """

    def __init__(self, info_likelihood, parametrisation, info_theory=None):
        # Store the input/output parameters
        self.input_params = parametrisation.input_params()
        self.output_params = parametrisation.output_params()
        # *IF* there is a theory code, initialise it
        if info_theory:
            input_params_theory = self.input_params.fromkeys(
                [k for k in self.input_params if k in parametrisation.theory_params()])
            output_params_theory = self.output_params.fromkeys(
                [k for k in self.output_params if k in parametrisation.theory_params()])
            name, fields = info_theory.items()[0]
            theory_class = get_class(name, kind=_theory)
            self.theory = theory_class(input_params_theory, output_params_theory, fields)
        else:
            self.theory = None
        # Initialise individual Likelihoods
        self._likelihoods = odict()
        for name, info in info_likelihood.iteritems():
            # If it does "external" key, wrap it up. Else, load it up
            if _external in info:
                self._likelihoods[name] = LikelihoodExternalFunction(
                    name, info, theory=getattr(self, _theory, None))
            else:
                lik_class = get_class(name)
                self._likelihoods[name] = lik_class(info, parametrisation, theory=self.theory)

        print "TODO!!!!! Make it work with MOCK likelihoods!!!!!"

## TO TOOLS!!!!
#        # Parameters: first check consistency through likelihoods. Then load.
#        mock_prefixes = [
#            p for p in [getattr(lik, "mock_prefix", None)
#                        for lik in self.likelihoods.values()] if p != None]
#        load_params(self, params_info=self._params_defaults, allow_unknown_prefixes=[""])
#        load_params(self, params_info=self._params_input,    allow_unknown_prefixes=mock_prefixes)
##        self._updated_info_params_liks = get_updated_params_info(self)
#        # Tell the the mock likelihoods their respective parameters read
#        for lik in self.likelihoods.values():
#            if lik.is_mock():
#                lik.mock_set_params(
#                    [p for p in self.fixed   if p.startswith(lik.mock_prefix)],
#                    [p for p in self.sampled if p.startswith(lik.mock_prefix)],
#                    [p for p in self.derived if p.startswith(lik.mock_prefix)])

# PASS ALL NEXT BLOCK TO THE NEW TOOL!
#        # "Externally" defined priors
#        self._info_prior = {}
#        for lik in self.likelihoods.values():
#            new = getattr(lik, _prior, {})
#            if any((k in self._info_prior) for k in new):
#                log.error("There are default priors sharing a name.")
#                raise HandledException
#            self._info_prior.update(new)
#        if info_prior:
#            self._info_prior.update(info_prior)

        # Check that all are recognised
        for params in ("input_params", "output_params"):
            info = getattr(parametrisation, params)()
            setattr(self, params, info)
            requested = set(info)
            known = set(chain(getattr(self.theory, params, []),
                              *[getattr(self[lik], params) for lik in self]))
            r_not_k = requested.difference(known)
            if r_not_k:
                log.error("Some of the requested %s parameters were not recognised "
                          "by any likelihood: %r.", params.split("_")[0], r_not_k)
                raise HandledException

    # "get" and iteration operate over the dictionary of likelihoods
    def __getitem__(self, key):
        return self._likelihoods.__getitem__(key)
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
        # Prepare the likelihood-defined derived parameters (only computed if requested)
        # Notice that they are in general different from the sampler-defined ones.
        derived_dict = {}
        # If theory code present, compute the necessary products
        if self.theory:
            this_params_dict = {p: input_params[p] for p in self.theory.input_params}
            self.theory.compute(derived=(derived_dict if derived != None else None),
                                **this_params_dict)
        # Compute each log-likelihood, and optionally get the respective derived params
        logps = []
        for lik in self:
            this_params_dict = {p: input_params[p] for p in self[lik].input_params}
            #,v in .iteritems() if
            #    ((p in self[lik].params_defaults()) or
            #     (p.startswith(self[lik].mock_prefix) if hasattr(self[lik], "mock_prefix") else False))])
            if derived != None:
                this_derived_dict = {}
            logps += [self[lik].logp(derived=this_derived_dict, **this_params_dict)]
            derived_dict.update(this_derived_dict)
        # Turn the derived params dict into a list and return
        if derived != None:
            derived += [derived_dict[p] for p in self.output_params]
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

    # def sampled_params(self):
    #     params = odict()
    #     if self.theory:
    #         params.update(self.theory.sampled_params())
    #     params.update(self.sampled)
    #     return params

    # def derived_all(self):
    #     """Returns all derived params (theory ones always go first)."""
    #     return (list(self.theory.derived) if self.theory else []) + list(self.derived)

    # def sampled_params_names(self):
    #     return self.sampled_params().keys()

    # def d(self):
    #     return len(self.sampled_params())

    # def sampled_params_by_likelihood(self):
    #     blocks = odict()
    #     for lik in self:
    #         params = [p for p in self[lik].params_defaults() if p in self.sampled]
    #         # mock likelihoods -- identify params by prefix
    #         if not params and hasattr(self[lik], "mock_prefix"):
    #             params = [p for p in self.sampled if p.startswith(self[lik].mock_prefix)]
    #         blocks[lik] = params
    #     if self.theory:
    #         blocks[_theory] = [p for p in self.theory.sampled]
    #     return blocks
            
    def speed_blocked_params(self, as_indices=False):
        """
        Blocks the parameters by likelihood, and sorts the blocks by speed.
        Parameters recognised by more than one likelihood are blocked in the slowest one.
        """
        print "TODO: UPDATE THIS ONE!!!"
        params_blocks = self.sampled_params_by_likelihood()
        speeds = odict([[lik,getattr(self[lik], "speed", 1)] for lik in self])
        if self.theory:
            speeds[_theory] = self.theory.speed
        speed_blocked = [[speed,params_blocks[name]] for name,speed in speeds.iteritems()]
        speed_blocked.sort()
        speed_blocked.reverse() # easier to look for slower ones
        # remove duplicates (take lowest speed)
        for i,(speed,params) in enumerate(speed_blocked):
            slower_params = [params2 for speed2,params2 in speed_blocked[i+1:]]
            slower_params = [p for ps in slower_params for p in ps] # flatten!
            speed_blocked[i][1] = [p for p in params if p not in slower_params]
        # remove empty blocks
        speed_blocked = [[speed,block] for speed,block in speed_blocked if block]
        speed_blocked.reverse()
# TODO: add the derived at the end with speed 0!!!
#        speed_blocked += [0, [derived!]]
        if not as_indices:
            return speed_blocked
        else:
            names = self.sampled_params().keys()
            return [[speed,[names.index(p) for p in block]]
                    for speed,block in speed_blocked]

#    def updated_info(self):
#        updated_info = odict()
#        for name, lik in self.likelihoods.iteritems():
#            updated_info[name] = dict([(k,v) for k,v in lik._updated_info[name].iteritems()
#                                       if k != _params])
#        return updated_info

    # def updated_info_theory(self):
    #     if self.theory:
    #         return self.theory.updated_info()
    #     else:
    #         return odict()
    
#    def updated_info_params(self):
#        if self.theory:
#            return odict([(k,v) for k,v in zip(
#                [_theory, _likelihood],
#                [self.theory.updated_info_params(), self._updated_info_params_liks])])
#        else:
#            return self._updated_info_params_liks

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
