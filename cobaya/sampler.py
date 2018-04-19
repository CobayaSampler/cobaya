"""
.. module:: sampler

:Synopsis: Prototype sampler class and sampler loader
:Author: Jesus Torrado

cobaya includes by default an
:doc:`advanced Monte Carlo Markov Chain (MCMC) sampler <sampler_mcmc>`
(a direct translation from `CosmoMC <http://cosmologist.info/cosmomc/>`_ and a dummy
:doc:`evaluate <sampler_evaluate>` sampler, that simply evaluates the posterior at a given
(or sampled) reference point. It also includes an interface to the
:doc:`PolyChord sampler <sampler_polychord>` (needs to be installed separately).

The sampler to use is specified by a `sampler` block in the input file, whose only member
is the sampler used, containing some options, if necessary.

.. code-block:: yaml

   sampler:
     mcmc:
       max_samples: 1000

or

.. code-block:: yaml

   sampler:
     polychord:
       path: /path/to/cosmo/PolyChord

Samplers can in general be swapped in the input file without needing to modify any other
block of the input.

In the cobaya code tree, each sampler is placed in its own folder, containing a file
defining the sampler's class, which inherits from the :class:`cobaya.Sampler`, and a
``[sampler_name].yaml`` file, containing all possible user-specified options for the
sampler and their default values. Whatever option is defined in this file becomes
automatically an attibute of the sampler's instance.

To implement your own sampler, or an interface to an external one, simply create a folder
under the ``cobaya/cobaya/samplers/`` folder and include the two files described above.
Your class needs to inherit from the :class:`cobaya.Sampler` class below, and needs to
implement only the methods ``initialise``, ``run``, ``close``, and ``products``.

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Global
from importlib import import_module
import numpy as np
from copy import deepcopy
import logging

# Local
from cobaya.conventions import _sampler, package
from cobaya.tools import get_class
from cobaya.log import HandledException

# Logger
log = logging.getLogger(__name__.split(".")[-1])


class Sampler(object):
    """Prototype of the sampler class."""

    # What you *must* implement to create your own sampler:

    def initialise(self):
        """
        Initialises the sampler: prepares the samples' collection,
        prepares the output, deals with MPI scheduling, imports an external sampler, etc.

        Options defined in the ``defaults.yaml`` file in the sampler's folder are
        automatically recognized as attributes, with the value given in the input file,
        if redefined there.

        The prior and likelihood are also accesible through the attributes with the same
        names.
        """
        pass

    def run(self):
        """
        Runs the main part of the algorithm of the sampler.
        Normally, it looks somewhat like

        .. code-block:: python

           while not [convergence criterion]:
               [do one more step]
               [update the collection of samples]
        """
        pass

    def close(self):
        """
        Finalises the sampler, if something needs to be done
        (e.g. generating additional output).
        """
        pass

    def products(self):
        """
        Returns the products expected in a scripted call of cobaya,
        (e.g. a collection of smaples or a list of them).
        """
        return None

    # Private methods: just ignore them:
    def __init__(self, info_sampler, parametrization, prior, likelihood, output):
        """
        Actual initialization of the class. Loads the default and input information and
        call the custom ``initialise`` method.

        [Do not modify this one.]
        """
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        # Load default and input info
        self.parametrization = parametrization
        self.prior = prior
        self.likelihood = likelihood
        self.output = output
        # Load info of the sampler
        for k in info_sampler:
            setattr(self, k, info_sampler[k])
        # Number of times the posterior has been evaluated
        self.n_eval = 0
        self.initialise()

    def logposterior(self, params_values, ignore_prior=False, make_finite=False):
        """
        Returns (logposterior,logprior,[loglikelihoods]) for an array of parameter values.
        If passes an empty list through ``derived``,
        it gets populated it with the derived parameters' values.
        """
        if not np.all(np.isfinite(params_values)):
            self.log.error("Got non-finite parameter values: %r", params_values)
            raise HandledException
        if not ignore_prior:
            logprior = self.prior.logp(params_values)
        else:
            logprior = 0
        logpost = deepcopy(logprior)
        logliks = []
        if logprior > -np.inf:
            derived = []
            logliks = self.likelihood.logps(
                self.parametrization.to_input(params_values), derived=derived)
            logpost += sum(logliks)
            derived_sampler = self.parametrization.to_derived(derived)
        else:
            derived_sampler = []
        self.n_eval += 1
        if make_finite:
            logpost = np.nan_to_num(logpost)
        return logpost, logprior, logliks, derived_sampler

    # Python magic for the "with" statement

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


def get_Sampler(info_sampler, parametrization, prior, likelihood, output_file):
    """
    Auxiliary function to retrieve and initialise the requested sampler.
    """
    try:
        name = list(info_sampler.keys())[0]
    except AttributeError:
        log.error("The sampler block must be a dictionary 'sampler: {options}'.")
        raise HandledException
    sampler_class = get_class(name, kind=_sampler)
    return sampler_class(info_sampler[name], parametrization, prior, likelihood, output_file)
