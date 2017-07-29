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
``defaults.yaml`` file, containing all possible user-specified options for the sampler and
their default value. Whatever option is defined in this file becomes automatically an
attibute of the sampler's instance.

To implement your own sampler, or an interface to an external one, simply create a folder
under the ``cobaya/cobaya/samplers/`` folder and include the two files described above.
Your class needs to inherit from the :class:`cobaya.Sampler` class below, and needs to
implement only the methods ``initialise``, ``run``, ``close``, and ``products``.

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
from importlib import import_module

# Local
from cobaya.conventions import *
from cobaya.tools import get_folder
from cobaya.input import load_input_and_defaults
from cobaya.log import HandledException

class Sampler(object):
    """Prototype of the sampler class."""

    # What you *must* implement to create your own sampler:
    
    def initialise(self):
        """
        Initialises the sampler: prepares the samples' collection,
        prepares the output, deals with MPI scheduling, imports an external sampler, etc.

        Options defined in the ``defaults.yaml`` file in the sampler's folder are
        automatically recognised as attributes, with the value given in the input file, 
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
    def __init__(self, info_sampler, prior, likelihood, output):
        """
        Actual initialisation of the class. Loads the default and input information and
        call the custom ``initialise`` method.
        
        [Do not modify this one.]
        """
        self.name = None
        # Load default and input info
        self._updated_info = load_input_and_defaults(self, info_sampler, kind="sampler")
        self.output = output
        self.prior = prior
        self.likelihood = likelihood
        self.initialise()

    # Get updated info, to be dumped into the *full* ``yaml`` file.
    def updated_info(self):
        return self._updated_info
        
    # Python magic for the "with" statement
    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


def get_Sampler(info_sampler, prior, likelihood, output_file):
    """
    Auxiliary function to retrieve and initialise the requested sampler.
    """
    try:
        name = info_sampler.keys()[0]
        class_folder = get_folder(name, input_sampler, sep=".", absolute=False)
        sampler_class = getattr(
            import_module(class_folder, package=package), name)
    except ImportError:
        log.error("Could not import sampler '%s'.", name.lower())
        raise HandledException
    return sampler_class(info_sampler[name], prior, likelihood, output_file)
