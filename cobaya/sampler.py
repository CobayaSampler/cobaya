"""
.. module:: sampler

:Synopsis: Base class for samplers and other parameter-space explorers
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
implement only the methods ``initialize``, ``_run``, and ``products``.

"""
# Global
import os
import logging
import numpy as np
from typing import Optional, Sequence, Mapping, Union, Any
from itertools import chain

# Local
from cobaya.conventions import kinds, _checkpoint_extension, _version
from cobaya.conventions import _progress_extension, _covmat_extension
from cobaya.conventions import partag, _packages_path, _force, _resume, _output_prefix
from cobaya.tools import get_class, deepcopy_where_possible, find_with_regexp
from cobaya.tools import recursive_update
from cobaya.log import LoggedError
from cobaya.yaml import yaml_load_file, yaml_dump
from cobaya.mpi import is_main_process, share_mpi, get_mpi_rank, more_than_one_process
from cobaya.component import CobayaComponent
from cobaya.input import update_info, is_equal_info, get_preferred_old_values
from cobaya.output import OutputDummy


def get_sampler_name_and_class(info_sampler):
    """
    Auxiliary function to retrieve the class of the required sampler.
    """
    check_sane_info_sampler(info_sampler)
    name = list(info_sampler)[0]
    return name, get_class(name, kind=kinds.sampler)


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


def check_sampler_info(info_old=None, info_new=None, is_resuming=False):
    """
    Checks compatibility between the new sampler info and that of a pre-existing run.

    Done separately from `Output.check_compatible_and_dump` because there may be
    multiple samplers mentioned in an `updated.yaml` file, e.g. `MCMC` + `Minimize`.
    """
    logger_sampler = logging.getLogger(__name__.split(".")[-1])
    if not info_old:
        return
    # TODO: restore this at some point: just append minimize info to the old one
    # There is old info, but the new one is Minimizer and the old one is not
    # if (len(info_old) == 1 and list(info_old) != ["minimize"] and
    #      list(info_new) == ["minimize"]):
    #     # In-place append of old+new --> new
    #     aux = info_new.pop("minimize")
    #     info_new.update(info_old)
    #     info_new.update({"minimize": aux})
    #     info_old = {}
    #     keep_old = {}
    if list(info_old) != list(info_new) and list(info_new) == ["minimize"]:
        return
    if list(info_old) == list(info_new):
        # Restore some selected old values for some classes
        keep_old = get_preferred_old_values({kinds.sampler: info_old})
        info_new = recursive_update(info_new, keep_old.get(kinds.sampler, {}))
    if not is_equal_info(
            {kinds.sampler: info_old}, {kinds.sampler: info_new}, strict=False):
        if is_resuming:
            raise LoggedError(
                logger_sampler, "Old and new Sampler information not compatible! "
                                "Resuming not possible!")
        else:
            raise LoggedError(
                logger_sampler, "Found old Sampler information which is not compatible "
                                "with the new one. Delete the previous output manually, "
                                "or automatically with either "
                                "'-%s', '--%s', '%s: True'" % (_force[0], _force, _force))


def get_sampler(info_sampler, model, output=None, packages_path=None):
    assert isinstance(info_sampler, Mapping), (
        "The first argument must be a dictionary with the info needed for the sampler. "
        "If you were trying to pass the name of an input file instead, "
        "load it first with 'cobaya.input.load_input', "
        "or, if you were passing a yaml string, load it with 'cobaya.yaml.yaml_load'.")
    logger_sampler = logging.getLogger(__name__.split(".")[-1])
    info_sampler = deepcopy_where_possible(info_sampler)
    if output is None:
        output = OutputDummy()
    # Check and update info
    check_sane_info_sampler(info_sampler)
    updated_info_sampler = update_info({kinds.sampler: info_sampler})[kinds.sampler]
    if logging.root.getEffectiveLevel() <= logging.DEBUG:
        logger_sampler.debug(
            "Input info updated with defaults (dumped to YAML):\n%s",
            yaml_dump(updated_info_sampler))
    # Get sampler class & check resume/force compatibility
    sampler_name, sampler_class = get_sampler_name_and_class(updated_info_sampler)
    check_sampler_info(
        (output.reload_updated_info(use_cache=True) or {}).get(kinds.sampler),
        updated_info_sampler, is_resuming=output.is_resuming())
    # Check if resumable run
    sampler_class.check_force_resume(output, info=updated_info_sampler[sampler_name])
    # Instantiate the sampler
    sampler_instance = sampler_class(updated_info_sampler[sampler_name], model,
                                     output, packages_path=packages_path)
    # If output, dump updated
    if output:
        to_dump = model.info()
        to_dump[kinds.sampler] = {sampler_name: sampler_instance.info()}
        to_dump[_output_prefix] = os.path.join(output.folder, output.prefix)
        output.check_and_dump_info(None, to_dump, check_compatible=False)
    return sampler_instance


class Sampler(CobayaComponent):
    """Base class for samplers."""

    # What you *must* implement to create your own sampler:

    seed: Optional[int]
    version: Optional[Union[dict, str]] = None

    _old_rng_state: Any
    _old_ext_rng_state: Any

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

    def _run(self):
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
    def __init__(self, info_sampler, model, output=None, packages_path=None, name=None):
        """
        Actual initialization of the class. Loads the default and input information and
        call the custom ``initialize`` method.

        [Do not modify this one.]
        """
        self.model = model
        self.output = output
        self._updated_info = deepcopy_where_possible(info_sampler)
        super().__init__(info_sampler, packages_path=packages_path,
                         name=name, initialize=False, standalone=False)
        # Seed, if requested
        if getattr(self, "seed", None) is not None:
            if not isinstance(self.seed, int) or not (0 <= self.seed <= 2 ** 32 - 1):
                raise LoggedError(
                    self.log, "Seeds must be a *positive integer* < 2**32 - 1, "
                              "but got %r with type %r",
                    self.seed, type(self.seed))
            # MPI-awareness: sum the rank to the seed
            if more_than_one_process():
                self.seed += get_mpi_rank()
            self.log.warning("This run has been SEEDED with seed %d", self.seed)
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
        self._set_rng()
        self.initialize()
        self._release_rng()
        self.model.set_cache_size(self._get_requested_cache_size())
        # Add to the updated info some values which are
        # only available after initialisation
        self._updated_info[_version] = self.get_version()

    def run(self):
        """
        Wrapper for `Sampler._run`, that takes care of seeding the
        random number generator.
        """
        self._set_rng()
        self._run()
        self._release_rng()

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

    def _get_requested_cache_size(self):
        """
        Override this for samplers than need more than 3 states cached
        per theory/likelihood.

        :return: number of points to cache
        """
        return 3

    def _set_rng(self):
        """
        For seeded runs, sets the internal state of the RNG.#
        """
        if getattr(self, "seed", None) is None:
            return
        # Store external state
        self._old_ext_rng_state = np.random.get_state()
        # Set our seed/state
        if not hasattr(self, "_old_rng_state"):
            np.random.seed(self.seed)
        else:
            np.random.set_state(self._old_rng_state)

    def _release_rng(self):
        """
        For seeded runs, releases the state of the RNG, restoring the old one.
        """
        if getattr(self, "seed", None) is None:
            return
        # Store our state
        self._old_rng_state = np.random.get_state()
        # Restore external state
        np.random.set_state(self._old_ext_rng_state)

    # TO BE DEPRECATED IN NEXT SUBVERSION
    def __getitem__(self, k):
        self.log.warning(
            "NB: the variables returned by `cobaya.run` have changed since the last "
            "version: they were `(updated_info, sampler_products)` and they are now "
            "`(updated_info, sampler)`. You can access the sampler products (the old "
            "return value) as `sampler.products()` and the `Model` used as "
            "`sampler.model`.")
        return self.products()[k]

    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        """
        Returns a list of tuples `(regexp, root)` of output files potentially produced.
        If `root` in the tuple is `None`, `output.folder` is used.

        If `minimal=True`, returns regexp's for the files that should really not be there
        when we are not resuming.
        """
        return []

    @classmethod
    def delete_output_files(cls, output, info=None):
        if output and is_main_process():
            for (regexp, root) in cls.output_files_regexps(output, info=info):
                # Special case: CovmatSampler's may have been given a covmat with the same
                # name that the output one. In that case, don't delete it!
                if issubclass(cls, CovmatSampler) and info:
                    if regexp.pattern.rstrip("$").endswith(_covmat_extension):
                        covmat_file = info.get("covmat", "")
                        if (isinstance(covmat_file, str) and covmat_file ==
                                getattr(regexp.match(covmat_file), "group",
                                        lambda: None)()):
                            continue
                output.delete_with_regexp(regexp, root)

    @classmethod
    def check_force_resume(cls, output, info=None):
        """
        Performs the necessary checks on existing files if resuming or forcing
        (including deleting some output files when forcing).
        """
        if not output:
            return
        if is_main_process():
            resuming = False
            if output.is_forcing():
                cls.delete_output_files(output, info=info)
            elif any(find_with_regexp(regexp, root or output.folder) for (regexp, root)
                     in cls.output_files_regexps(output=output, info=info, minimal=True)):
                if output.is_resuming():
                    output.log.info("Found an old sample. Resuming.")
                    resuming = True
                else:
                    raise LoggedError(
                        output.log, "Delete the previous output manually, automatically "
                                    "('-%s', '--%s', '%s: True')" % (
                                        _force[0], _force, _force) +
                                    " or request resuming ('-%s', '--%s', '%s: True')" % (
                                        _resume[0], _resume, _resume))
            else:
                if output.is_resuming():
                    output.log.info(
                        "Did not find an old sample. Cleaning up and starting anew.")
                # Clean up old files, and set resuming=False,
                # regardless of requested value
                cls.delete_output_files(output, info=info)
        else:
            resuming = None
        output.set_resuming(resuming)


class Minimizer(Sampler):
    """
    base class for minimizers
    """


class CovmatSampler(Sampler):
    """
    Parent class for samplers that are initialised with a covariance matrix.
    """
    covmat_params: Sequence[str]

    def _load_covmat(self, prefer_load_old, auto_params=None):
        if prefer_load_old and os.path.exists(self.covmat_filename()):
            if is_main_process():
                covmat = np.atleast_2d(np.loadtxt(self.covmat_filename()))
            else:
                covmat = None
            covmat = share_mpi(covmat)
            self.mpi_info("Covariance matrix from previous sample.")
            return covmat, []
        else:
            return share_mpi(self.initial_proposal_covmat(auto_params=auto_params) if
                             is_main_process() else None)

    def initial_proposal_covmat(self, auto_params=None):
        """
        Build the initial covariance matrix, using the data provided, in descending order
        of priority:
        1. "covmat" field in the sampler block (including `auto` search).
        2. "proposal" field for each parameter.
        3. variance of the reference pdf.
        4. variance of the prior pdf.

        The covariances between parameters when both are present in a covariance matrix
        provided through option 1 are preserved. All other covariances are assumed 0.

        If `covmat: auto`, use the keyword `auto_params` to restrict the parameters for
        which a covariance matrix is searched (default: None, meaning all sampled params).
        """
        params_infos = self.model.parameterization.sampled_params_info()
        covmat = np.diag([np.nan] * len(params_infos))
        # Try to generate it automatically
        self.covmat = getattr(self, 'covmat', None)
        if isinstance(self.covmat, str) and self.covmat.lower() == "auto":
            params_infos_covmat = deepcopy_where_possible(params_infos)
            for p in list(params_infos_covmat):
                if p not in (auto_params or []):
                    params_infos_covmat.pop(p, None)
            auto_covmat = self.model.get_auto_covmat(params_infos_covmat)
            if auto_covmat:
                self.covmat = os.path.join(auto_covmat["folder"], auto_covmat["name"])
                self.log.info("Covariance matrix selected automatically: %s", self.covmat)
            else:
                self.covmat = None
                self.log.info("Could not automatically find a good covmat. "
                              "Will generate from parameter info (proposal and prior).")
        # If given, load and test the covariance matrix
        if isinstance(self.covmat, str):
            covmat_pre = "{%s}" % _packages_path
            if self.covmat.startswith(covmat_pre):
                self.covmat = self.covmat.format(
                    **{_packages_path: self.packages_path}).replace("/", os.sep)
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
            str_msg = "the `covmat_params` list"
            if isinstance(self.covmat, str):
                str_msg = "the header of the covmat file %r" % self.covmat
            if len(loaded_params) != len(set(loaded_params)):
                duplicated = list(set(
                    p for p in loaded_params if list(loaded_params).count(p) > 1))
                raise LoggedError(
                    self.log,
                    "Parameter(s) %r appear more than once in %s", duplicated, str_msg)
            if len(loaded_params) != loaded_covmat.shape[0]:
                raise LoggedError(
                    self.log, "The number of parameters in %s and the "
                              "dimensions of the matrix do not agree: %d vs %r",
                    str_msg, len(loaded_params), loaded_covmat.shape)
            if not (np.allclose(loaded_covmat.T, loaded_covmat) and
                    np.all(np.linalg.eigvals(loaded_covmat) > 0)):
                str_msg = "passed"
                if isinstance(self.covmat, str):
                    str_msg = "loaded from %r" % self.covmat
                raise LoggedError(
                    self.log, "The covariance matrix %s is not a positive-definite, "
                              "symmetric square matrix.", str_msg)
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

    def dump_covmat(self, covmat=None):
        if covmat is None:
            covmat = self.covmat
        np.savetxt(self.covmat_filename(), covmat, header=" ".join(
            list(self.model.parameterization.sampled_params())))
