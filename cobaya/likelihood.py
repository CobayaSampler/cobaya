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

# Default options for all subclasses
class_options = {"speed": 1}


class Likelihood():
    """Likelihood class prototype."""

    # Generic initialisation -- do not touch
    def __init__(self, info, parametrisation, theory=None):
        # Load info of the likelihood
        for k in info:
            setattr(self, k, info[k])
        # Mock likelihoods: gather all parameters starting with `mock_prefix`
        if self.is_mock():
            all_params = list(parametrisation.input_params())+list(parametrisation.output_params())
            info[_params] = [p for p in all_params if p.startswith(self.mock_prefix or "")]
        # Load parameters
        self.input_params = odict(
            [(p,p_info) for p,p_info in parametrisation.input_params().iteritems() if p in info[_params]])
        self.output_params = odict(
            [(p,p_info) for p,p_info in parametrisation.output_params().iteritems() if p in info[_params]])
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

    def wait(self):
        if self.delay:
            log.debug("Sleeping for %f seconds.", self.delay)
        sleep(self.delay)

    def d(self):
        return len(self.input_params)

    def is_mock(self):
        return hasattr(self, "mock_prefix")


class LikelihoodExternalFunction(Likelihood):
    def __init__(self, name, info, theory=None):
        self.theory = theory
        # Load info of the likelihood
        for k in info:
            setattr(self, k, info[k])
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
        # Store the input params and likelihods on which each sampled params depends
        self.sampled_input_dependence = parametrisation.sampled_input_dependence()
        self.sampled_lik_dependence = odict(
            [[p,[lik for lik in list(self)+([_theory] if self.theory else [])
                 if any([(i in self[lik].input_params) for i in (i_s or [p])])]]
             for p,i_s in self.sampled_input_dependence.items()])
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
        # Prepare the likelihood-defined derived parameters (only computed if requested)
        # Notice that they are in general different from the sampler-defined ones.
        derived_dict = {}
        # If theory code present, compute the necessary products
        if self.theory:
            this_params_dict = {p: input_params[p] for p in self.theory.input_params}
            success = self.theory.compute(
                derived=(derived_dict if derived != None else None), **this_params_dict)
            if not success:
                if derived != None:
                    derived += [np.nan]*len(self.output_params)
                return np.array([-np.inf for _ in self])
        # Compute each log-likelihood, and optionally get the respective derived params
        logps = []
        for lik in self:
            this_params_dict = {p: input_params[p] for p in self[lik].input_params}
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

    def d(self):
        return sum([self[lik].d() for lik in self])

    def speeds_of_params(self):
        """
        Blocks the sampled parameters by likelihood, and sorts the blocks by speed.
        Returns an ``OrderedDict`` ``{speed: [params]}``, sorted by ascending speeds.
        Parameters recognised by more than one likelihood are blocked in the slowest one.
        """
        param_with_speed = odict([[p,min([self[lik].speed for lik in liks])]
                                  for p,liks in self.sampled_lik_dependence.items()])
        # Invert it!
        return odict([[speed,[p for p,speed2 in param_with_speed.items() if speed == speed2]]
                      for speed in sorted(list(set(param_with_speed.values())))])

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
