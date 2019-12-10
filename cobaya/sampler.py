"""
.. module:: sampler

:Synopsis: Prototype sampler class and sampler loader
:Author: Jesus Torrado

cobaya includes by default a
:doc:`Monte Carlo Markov Chain (MCMC) sampler <sampler_mcmc>`
(a direct translation from `CosmoMC <https://cosmologist.info/cosmomc/>`_) and a dummy
:doc:`evaluate <sampler_evaluate>` sampler that simply evaluates the posterior at a given
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
sampler and their default values. Whatever option is defined in this file automatically
becomes an attribute of the sampler's instance.

To implement your own sampler, or an interface to an external one, simply create a folder
under the ``cobaya/cobaya/samplers/`` folder and include the two files described above.
Your class needs to inherit from the :class:`cobaya.Sampler` class below, and needs to
implement only the methods ``initialize``, ``run``, ``close``, and ``products``.

"""
# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

# Global
import os
import logging
import numpy as np

# Local
from cobaya.conventions import kinds, _resume_default, _checkpoint_extension
from cobaya.conventions import _covmat_extension, _progress_extension, _module_path
from cobaya.tools import get_class
from cobaya.log import LoggedError
from cobaya.yaml import yaml_load_file
from cobaya.mpi import is_main_process
from cobaya.component import CobayaComponent


class Sampler(CobayaComponent):
    """Prototype of the sampler class."""

    # What you *must* implement to create your own sampler:

    def initialize(self):
        """
        Initializes the sampler: prepares the samples' collection,
        prepares the output, deals with MPI scheduling, imports an external sampler, etc.

        Options defined in the ``defaults.yaml`` file in the sampler's folder are
        automatically recognized as attributes, with the value given in the input file,
        if redefined there.

        The prior and likelihood are also accessible through the attributes with the same
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

    def products(self):
        """
        Returns the products expected in a scripted call of cobaya,
        (e.g. a collection of samples or a list of them).
        """
        return None

    # Private methods: just ignore them:
    def __init__(self, info_sampler, model, output, resume=_resume_default,
                 path_install=None, name=None):
        """
        Actual initialization of the class. Loads the default and input information and
        call the custom ``initialize`` method.

        [Do not modify this one.]
        """

        self.model = model
        self.output = output

        super(Sampler, self).__init__(info_sampler, path_install=path_install,
                                      name=name, initialize=False, standalone=False)

        # Seed, if requested
        if getattr(self, "seed", None) is not None:
            self.log.warning("This run has been SEEDED with seed %d", self.seed)
            try:
                np.random.seed(self.seed)
            except TypeError:
                raise LoggedError(
                    self.log, "Seeds must be *integer*, but got %r with type %r",
                    self.seed, type(self.seed))
        # Load checkpoint info, if resuming
        self.resuming = resume
        if self.resuming and not isinstance(self, Minimizer):
            try:
                checkpoint_info = yaml_load_file(self.checkpoint_filename())
                try:
                    for k, v in checkpoint_info[kinds.sampler][self.get_name()].items():
                        setattr(self, k, v)
                    self.resuming = True
                    if is_main_process():
                        self.log.info("Resuming from previous sample!")
                except KeyError:
                    if is_main_process():
                        raise LoggedError(
                            self.log, "Checkpoint file found at '%s' "
                                      "but it corresponds to a different sampler.",
                            self.checkpoint_filename())
            except (IOError, TypeError):
                pass
        else:
            try:
                os.remove(self.checkpoint_filename())
                os.remove(self.progress_filename())
            except (OSError, TypeError):
                pass
        self.initialize()
        self.model.set_cache_size(self._get_requested_cache_size())

    def checkpoint_filename(self):
        if self.output:
            return os.path.join(
                self.output.folder, self.output.prefix + _checkpoint_extension)
        return None

    def progress_filename(self):
        if self.output:
            return os.path.join(
                self.output.folder, self.output.prefix + _progress_extension)
        return None

    def close(self, exception_type, exception_value, traceback):
        """
        Finalizes the sampler, if something needs to be done
        (e.g. generating additional output).
        """
        pass

    def _get_requested_cache_size(self):
        """
        Override this for samplers than need more than 3 states cached
        per theory/likelihood.

        :return: number of points to cache
        """
        return 3

    # Python magic for the "with" statement

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # Reset seed
        if getattr(self, "seed", None) is not None:
            np.random.seed(self.seed)
        self.close(exception_type, exception_value, traceback)


class Minimizer(Sampler):
    """
    base class for minimizers
    """
    pass


def get_sampler_class(info):
    info_sampler = info.get(kinds.sampler)
    if info_sampler:
        name = list(info_sampler)[0]
        if name:
            module_path = (info_sampler[name] or {}).get(_module_path)
            return get_class(name, kinds.sampler, None_if_not_found=True,
                             module_path=module_path)


def get_sampler(info_sampler, model, output_file,
                resume=_resume_default, modules=None):
    """
    Auxiliary function to retrieve and initialize the requested sampler.
    """
    log = logging.getLogger(__name__.split(".")[-1])
    if not info_sampler:
        raise LoggedError(log, "No sampler given!")
    try:
        name = list(info_sampler)[0]
    except AttributeError:
        raise LoggedError(
            log, "The sampler block must be a dictionary 'sampler: {options}'.")
    if len(info_sampler) > 1:
        raise LoggedError(log, "nly one sampler currently supported at a time.")

    sampler_class = get_class(name, kind=kinds.sampler)
    assert issubclass(sampler_class, Sampler)
    return sampler_class(info_sampler[name], model, output_file, resume=resume,
                         path_install=modules, name=name)
