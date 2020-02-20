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
# Global
import os
import logging
import numpy as np
from typing import Optional, Sequence
from itertools import chain

# Local
from cobaya.conventions import kinds, _resume_default, _checkpoint_extension, _version
from cobaya.conventions import _progress_extension, _module_path, _covmat_extension
from cobaya.conventions import partag, _path_install
from cobaya.tools import get_class, deepcopy_where_possible, find_with_regexp
from cobaya.log import LoggedError
from cobaya.yaml import yaml_load_file
from cobaya.mpi import is_main_process, share_mpi, sync_processes
from cobaya.component import CobayaComponent


class Sampler(CobayaComponent):
    """Prototype of the sampler class."""

    # What you *must* implement to create your own sampler:

    seed: Optional[int]

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
    def __init__(self, info_sampler, model, output=None, path_install=None, name=None):
        """
        Actual initialization of the class. Loads the default and input information and
        call the custom ``initialize`` method.

        [Do not modify this one.]
        """
        self.model = model
        self.output = output
        self._updated_info = deepcopy_where_possible(info_sampler)
        super().__init__(info_sampler, path_install=path_install,
                         name=name, initialize=False, standalone=False)
        # Seed, if requested
        if getattr(self, "seed", None) is not None:
            self.log.warning("This run has been SEEDED with seed %d", self.seed)
            try:
                # TODO, says this is deprecated
                np.random.seed(self.seed)
            except TypeError:
                raise LoggedError(
                    self.log, "Seeds must be *integer*, but got %r with type %r",
                    self.seed, type(self.seed))
        # Load checkpoint info, if resuming
        if self.output.is_resuming() and not isinstance(self, Minimizer):
            try:
                checkpoint_info = yaml_load_file(self.checkpoint_filename())
                try:
                    for k, v in checkpoint_info[kinds.sampler][self.get_name()].items():
                        setattr(self, k, v)
                    self.mpi_info("Resuming from previous sample!")
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
        # Add to the updated info some values which are only available after initialisation
        self._updated_info[_version] = self.get_version()

    def info(self):
        """
        Returns a copy of the information used to initialise the sampler,
        including defaults and some new values that are only available after
        initialisation.
        """
        return deepcopy_where_possible(self._updated_info)

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

    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        """
        Returns a list of regexp's of output files potentially produced.

        If `minimal=True`, returns regexp's for the files that should really not be there
        when we are not resuming.
        """
        return []

    @classmethod
    def delete_output_files(cls, output, info=None):
        if output and is_main_process():
            for regexp in cls.output_files_regexps(output, info=info):
                # Special case: CovmatSampler's may have been given a covmat with the same
                # name that the output one. In that case, don't delete it!
                if issubclass(cls, CovmatSampler) and info:
                    if regexp.pattern.rstrip("$").endswith(_covmat_extension):
                        covmat_file = info.get("covmat", "")
                        if (isinstance(covmat_file, str) and covmat_file ==
                            getattr(regexp.match(covmat_file), "group", lambda: None)()):
                            continue
                output.delete_with_regexp(regexp)

    @classmethod
    def check_force_resume(cls, output, info=None):
        """
        Performs the necessary checks on existing files if resuming or forcing
        (including deleting some output files when forcing).
        """
        if not output:
            return
        if is_main_process():
            if output.is_forcing():
                cls.delete_output_files(output, info=info)
            elif any(find_with_regexp(regexp)
                     for regexp in cls.output_files_regexps(
                             output=output, info=info, minimal=True)):
                if output.is_resuming():
                    output.log.info("Found and old sample. Resuming.")
                else:
                    raise LoggedError(
                        output.log, "Delete the previous output manually, automatically "
                                    "('-%s', '--%s', '%s: True')" % (
                                    _force[0], _force, _force) +
                                    " or request resuming ('-%s', '--%s', '%s: True')" % (
                                   _resume[0], _resume, _resume))
            else:
                if output.is_resuming():
                    output.log.info("Did not find an old sample. Cleaning up and starting anew.")
                # Clean up old files, and set resuming=False, regardless of requested value
                cls.delete_output_files(output, info=info)
                output.set_resuming(False)

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


def get_sampler_class_OLD(info):
    info_sampler = info.get(kinds.sampler)
    if info_sampler:
        name = list(info_sampler)[0]
        if name:
            module_path = (info_sampler[name] or {}).get(_module_path)
            return get_class(name, kinds.sampler, None_if_not_found=True,
                             module_path=module_path)


def get_sampler_class(info_sampler):
    """
    Auxiliary function to retrieve the class of the required sampler.
    """
    check_sane_info_sampler(info_sampler)
    name = list(info_sampler)[0]
    return get_class(name, kind=kinds.sampler)


def check_sane_info_sampler(info_sampler):
    log = logging.getLogger(__name__.split(".")[-1])
    if not info_sampler:
        raise LoggedError(log, "No sampler given!")
    try:
        list(info_sampler)[0]
    except AttributeError:
        raise LoggedError(
            log, "The sampler block must be a dictionary 'sampler: {options}'.")
    if len(info_sampler) > 1:
        raise LoggedError(log, "Only one sampler currently supported at a time.")


class CovmatSampler(Sampler):
    """
    Parent class for samplers that are initialised with a covariance matrix.
    """
    covmat_params: Sequence[str]

    def _load_covmat(self, from_old_chain, slow_params=None):
        if from_old_chain and os.path.exists(self.covmat_filename()):
            covmat = np.atleast_2d(share_mpi(np.loadtxt(
                self.covmat_filename()) if is_main_process() else None))
            self.mpi_info("Covariance matrix from checkpoint.")
            return covmat, []
        else:
            if slow_params is None:
                slow_params = list(self.model.parameterization.sampled_params())
            return share_mpi(self.initial_proposal_covmat(slow_params=slow_params) if
                             is_main_process() else None)

    def initial_proposal_covmat(self, slow_params=None):
        """
        Build the initial covariance matrix, using the data provided, in descending order
        of priority:
        1. "covmat" field in the "mcmc" sampler block.
        2. "proposal" field for each parameter.
        3. variance of the reference pdf.
        4. variance of the prior pdf.

        The covariances between parameters when both are present in a covariance matrix
        provided through option 1 are preserved. All other covariances are assumed 0.
        """
        params_infos = self.model.parameterization.sampled_params_info()
        covmat = np.diag([np.nan] * len(params_infos))
        # Try to generate it automatically
        self.covmat = getattr(self, 'covmat', None)
        if isinstance(self.covmat, str) and self.covmat.lower() == "auto":
            slow_params_info = {
                p: info for p, info in params_infos.items() if p in slow_params}
            auto_covmat = self.model.get_auto_covmat(slow_params_info)
            if auto_covmat:
                self.covmat = os.path.join(auto_covmat["folder"], auto_covmat["name"])
                self.log.info("Covariance matrix selected automatically: %s", self.covmat)
            else:
                self.covmat = None
                self.log.info("Could not automatically find a good covmat. "
                              "Will generate from parameter info (proposal and prior).")
        # If given, load and test the covariance matrix
        if isinstance(self.covmat, str):
            covmat_pre = "{%s}" % _path_install
            if self.covmat.startswith(covmat_pre):
                self.covmat = self.covmat.format(
                    **{_path_install: self.path_install}).replace("/", os.sep)
            try:
                with open(self.covmat, "r", encoding="utf-8-sig") as file_covmat:
                    header = file_covmat.readline()
                loaded_covmat = np.loadtxt(self.covmat)
            except TypeError:
                raise LoggedError(self.log, "The property 'covmat' must be a file name,"
                                            "but it's '%s'.", str(self.covmat))
            except IOError:
                raise LoggedError(self.log, "Can't open covmat file '%s'.", self.covmat)
            if header[0] != "#":
                raise LoggedError(
                    self.log, "The first line of the covmat file '%s' "
                              "must be one list of parameter names separated by spaces "
                              "and staring with '#', and the rest must be a square "
                              "matrix, with one row per line.", self.covmat)
            loaded_params = header.strip("#").strip().split()
        elif hasattr(self.covmat, "__getitem__"):
            if not self.covmat_params:
                raise LoggedError(
                    self.log, "If a covariance matrix is passed as a numpy array, "
                              "you also need to pass the parameters it corresponds to "
                              "via 'covmat_params: [name1, name2, ...]'.")
            loaded_params = self.covmat_params
            loaded_covmat = np.array(self.covmat)
        elif self.covmat:
            raise LoggedError(self.log, "Invalid covmat")
        if self.covmat is not None:
            if len(loaded_params) != len(set(loaded_params)):
                raise LoggedError(
                    self.log, "There are duplicated parameters in the header of the "
                              "covmat file '%s' ", self.covmat)
            if len(loaded_params) != loaded_covmat.shape[0]:
                raise LoggedError(
                    self.log, "The number of parameters in the header of '%s' and the "
                              "dimensions of the matrix do not coincide.", self.covmat)
            if not (np.allclose(loaded_covmat.T, loaded_covmat) and
                    np.all(np.linalg.eigvals(loaded_covmat) > 0)):
                raise LoggedError(
                    self.log, "The covmat loaded from '%s' is not a positive-definite, "
                              "symmetric square matrix.", self.covmat)
            # Fill with parameters in the loaded covmat
            renames = [[p] + np.atleast_1d(v.get(partag.renames, [])).tolist()
                       for p, v in params_infos.items()]
            renames = {a[0]: a for a in renames}
            indices_used, indices_sampler = zip(*[
                [loaded_params.index(p),
                 [list(params_infos).index(q) for q, a in renames.items() if p in a]]
                for p in loaded_params])
            if not any(indices_sampler):
                raise LoggedError(
                    self.log,
                    "A proposal covariance matrix has been loaded, but none of its "
                    "parameters are actually sampled here. Maybe a mismatch between"
                    " parameter names in the covariance matrix and the input file?")
            indices_used, indices_sampler = zip(*[
                [i, j] for i, j in zip(indices_used, indices_sampler) if j])
            if any(len(j) - 1 for j in indices_sampler):
                first = next(j for j in indices_sampler if len(j) > 1)
                raise LoggedError(
                    self.log,
                    "The parameters %s have duplicated aliases. Can't assign them an "
                    "element of the covariance matrix unambiguously.",
                    ", ".join([list(params_infos)[i] for i in first]))
            indices_sampler = list(chain(*indices_sampler))
            covmat[np.ix_(indices_sampler, indices_sampler)] = (
                loaded_covmat[np.ix_(indices_used, indices_used)])
            self.log.info(
                "Covariance matrix loaded for params %r",
                [list(params_infos)[i] for i in indices_sampler])
            missing_params = set(params_infos).difference(
                set(list(params_infos)[i] for i in indices_sampler))
            if missing_params:
                self.log.info(
                    "Missing proposal covariance for params %r",
                    [p for p in self.model.parameterization.sampled_params()
                     if p in missing_params])
            else:
                self.log.info("All parameters' covariance loaded from given covmat.")
        # Fill gaps with "proposal" property, if present, otherwise ref (or prior)
        where_nan = np.isnan(covmat.diagonal())
        if np.any(where_nan):
            covmat[where_nan, where_nan] = np.array(
                [info.get(partag.proposal, np.nan) ** 2
                 for info in params_infos.values()])[where_nan]
        where_nan2 = np.isnan(covmat.diagonal())
        if np.any(where_nan2):
            covmat[where_nan2, where_nan2] = (
                self.model.prior.reference_covmat().diagonal()[where_nan2])
        assert not np.any(np.isnan(covmat))
        return covmat, where_nan

    def covmat_filename(self):
        if self.output:
            return os.path.join(
                self.output.folder, self.output.prefix + _covmat_extension)
        return None
