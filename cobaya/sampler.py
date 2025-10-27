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
defining the sampler's class, which inherits from the :class:`cobaya.sampler.Sampler`, and a
``[sampler_name].yaml`` file, containing all possible user-specified options for the
sampler and their default values. Whatever option is defined in this file automatically
becomes an attribute of the sampler's instance.

To implement your own sampler, or an interface to an external one, simply create a folder
under the ``cobaya/samplers/`` folder and include the two files described above.
Your class needs to inherit from the :class:`cobaya.sampler.Sampler` class below, and needs to
implement only the methods ``initialize``, ``run``, and ``products``.

"""

import os
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.random import SeedSequence, default_rng

from cobaya import mpi
from cobaya.component import CobayaComponent, get_component_class
from cobaya.conventions import Extension, packages_path_input
from cobaya.input import get_preferred_old_values, is_equal_info, update_info
from cobaya.log import LoggedError, get_logger, is_debug
from cobaya.model import Model
from cobaya.output import Output, OutputDummy
from cobaya.tools import (
    deepcopy_where_possible,
    find_with_regexp,
    recursive_update,
    str_to_list,
)
from cobaya.typing import SamplerDict, SamplersDict
from cobaya.yaml import yaml_dump, yaml_load_file

# Avoid importing GetDist if not necessary
if TYPE_CHECKING:
    from getdist import MCSamples

    from cobaya.collection import SampleCollection


def get_sampler_name_and_class(info_sampler: SamplersDict, logger=None):
    """
    Auxiliary function to retrieve the class of the required sampler.
    """
    check_sane_info_sampler(info_sampler)
    name = list(info_sampler)[0]
    sampler_class = get_component_class(name, kind="sampler", logger=logger)
    assert issubclass(sampler_class, Sampler)
    return name, sampler_class


def check_sane_info_sampler(info_sampler: SamplersDict):
    if not info_sampler:
        raise LoggedError(__name__, "No sampler given!")
    try:
        list(info_sampler)[0]
    except AttributeError as excpt:
        raise LoggedError(
            __name__, "The sampler block must be a dictionary 'sampler: {options}'."
        ) from excpt
    if len(info_sampler) > 1:
        raise LoggedError(__name__, "Only one sampler currently supported at a time.")


def check_sampler_info(
    info_old: SamplersDict | None, info_new: SamplersDict, is_resuming=False
):
    """
    Checks compatibility between the new sampler info and that of a pre-existing run.

    Done separately from `Output.check_compatible_and_dump` because there may be
    multiple samplers mentioned in an `updated.yaml` file, e.g. `MCMC` + `Minimize`.
    """
    logger_sampler = get_logger(__name__)
    if not info_old:
        return info_new
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
        keep_old = get_preferred_old_values({"sampler": info_old})
        info_new = recursive_update(info_new, keep_old.get("sampler", {}))
    if not is_equal_info({"sampler": info_old}, {"sampler": info_new}, strict=False):
        if is_resuming:
            raise LoggedError(
                logger_sampler,
                "Old and new Sampler information not compatible! Resuming not possible!",
            )
        else:
            raise LoggedError(
                logger_sampler,
                "Found old Sampler information which is not compatible "
                "with the new one. Delete the previous output manually, "
                "or automatically with either "
                "'-f', '--force', 'force: True'",
            )
    return info_new


def get_sampler(
    info_sampler: SamplersDict,
    model: Model,
    output: Output | None = None,
    packages_path: str | None = None,
) -> "Sampler":
    assert isinstance(info_sampler, Mapping), (
        "The first argument must be a dictionary with the info needed for the sampler. "
        "If you were trying to pass the name of an input file instead, "
        "load it first with 'cobaya.input.load_input', "
        "or, if you were passing a yaml string, load it with 'cobaya.yaml.yaml_load'."
    )
    logger_sampler = get_logger(__name__)
    info_sampler = deepcopy_where_possible(info_sampler)
    if output is None:
        output = OutputDummy()
    # Check and update info
    check_sane_info_sampler(info_sampler)
    updated_info_sampler = update_info({"sampler": info_sampler})["sampler"]  # type: ignore
    if is_debug(logger_sampler):
        logger_sampler.debug(
            "Input info updated with defaults (dumped to YAML):\n%s",
            yaml_dump(updated_info_sampler),
        )
    # Get sampler class & check resume/force compatibility
    sampler_name, sampler_class = get_sampler_name_and_class(
        updated_info_sampler, logger=logger_sampler
    )
    updated_info_sampler = check_sampler_info(
        (output.get_updated_info(use_cache=True) or {}).get("sampler"),
        updated_info_sampler,
        is_resuming=output.is_resuming(),
    )
    # Check if resumable run
    sampler_class.check_force_resume(output, info=updated_info_sampler[sampler_name])
    # Instantiate the sampler
    sampler_instance = sampler_class(
        updated_info_sampler[sampler_name], model, output, packages_path=packages_path
    )
    # If output, dump updated
    if output:
        to_dump = model.info()
        to_dump["sampler"] = {sampler_name: sampler_instance.info()}
        to_dump["output"] = os.path.join(output.folder, output.prefix)
        output.check_and_dump_info(None, to_dump, check_compatible=False)
    return sampler_instance


class Sampler(CobayaComponent):
    """Base class for samplers."""

    # What you *must* implement to create your own sampler:

    seed: None | int | Sequence[int]
    version: dict | str | None = None
    # Set to True if sampler is guaranteed to never periodic parameter values outside
    # the prior definition range.
    supports_periodic_params: bool = False

    _rng: np.random.Generator

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

    def run(self):
        """
        Runs the main part of the algorithm of the sampler.
        Normally, it looks somewhat like

        .. code-block:: python

           while not [convergence criterion]:
               [do one more step]
               [update the collection of samples]
        """

    def samples(self, **kwargs) -> Union["SampleCollection", "MCSamples", None]:
        """
        Returns the products expected in a scripted call of cobaya,
        (e.g. a collection of samples or a list of them).
        """

    def products(self, **kwargs) -> dict:
        """
        Returns the products expected in a scripted call of cobaya,
        (e.g. a collection of samples or a list of them).
        """
        return {}

    @property
    def random_state(self) -> np.random.Generator:
        return self._rng

    @property
    def model(self) -> Model:
        return self._model

    @property
    def output(self) -> Output | None:
        return self._output

    # Private methods: just ignore them:
    def __init__(
        self,
        info_sampler: SamplerDict,
        model: Model,
        output: Output | None = None,
        packages_path: str | None = None,
        name: str | None = None,
    ):
        """
        Actual initialization of the class. Loads the default and input information and
        call the custom ``initialize`` method.

        [Do not modify this one.]
        """
        self._model = model
        self._output = output
        self._updated_info = deepcopy_where_possible(info_sampler)
        super().__init__(
            info_sampler,
            packages_path=packages_path,
            name=name,
            initialize=False,
            standalone=False,
        )
        if not model.parameterization.sampled_params():
            self.mpi_warning(
                "No sampled parameters requested! This will fail for non-mock samplers."
            )
        if self.model.prior._periodic_bounds and not self.supports_periodic_params:
            self.log.warning(
                "There are periodic sampled parameters, but this sampler does not support"
                " them. Treating their prior definition range as hard boundaries. This "
                "may have unexpected effects."
            )
        # Load checkpoint info, if resuming
        if self.output.is_resuming() and not isinstance(self, Minimizer):
            checkpoint_info = None
            if mpi.is_main_process():
                try:
                    checkpoint_info = yaml_load_file(self.checkpoint_filename())

                    if self.get_name() not in checkpoint_info["sampler"]:
                        raise LoggedError(
                            self.log,
                            "Checkpoint file found at '%s' "
                            "but it corresponds to a different sampler.",
                            self.checkpoint_filename(),
                        )
                except (OSError, TypeError):
                    pass
            checkpoint_info = mpi.share_mpi(checkpoint_info)
            if checkpoint_info:
                self.set_checkpoint_info(checkpoint_info)
                self.mpi_info("Resuming from previous sample!")
        elif not isinstance(self, Minimizer) and mpi.is_main_process():
            try:
                output.delete_file_or_folder(self.checkpoint_filename())
                output.delete_file_or_folder(self.progress_filename())
            except (OSError, TypeError):
                pass
        self._set_rng()
        self.initialize()
        model.set_cache_size(self._get_requested_cache_size())
        # Add to the updated info some values which are
        # only available after initialisation
        self._updated_info["version"] = self.get_version()

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
                self.output.folder, self.output.prefix + Extension.checkpoint
            )
        return None

    def progress_filename(self):
        if self.output:
            return os.path.join(
                self.output.folder, self.output.prefix + Extension.progress
            )
        return None

    def set_checkpoint_info(self, checkpoint_info):
        for k, v in checkpoint_info["sampler"][self.get_name()].items():
            setattr(self, k, v)
        # check if convergence parameters changed, and if so converged=False
        old_info = self.output.get_updated_info(use_cache=True)
        assert old_info
        if self.converge_info_changed(
            old_info["sampler"][self.get_name()], self._updated_info
        ):
            self.converged = False

    def converge_info_changed(self, old_info, new_info):
        return old_info != new_info

    def _get_requested_cache_size(self):
        """
        Override this for samplers than need more than 3 states cached
        per theory/likelihood.

        :return: number of points to cache
        """
        return 3

    def _set_rng(self):
        """
        Initialize random generator stream. For seeded runs, sets the state reproducibly.
        """
        # TODO: checkpointing save of self._rng.bit_generator.state per process
        if mpi.is_main_process():
            seed = getattr(self, "seed", None)
            if seed is not None:
                self.mpi_warning("This run has been SEEDED with seed %s", seed)
            ss = SeedSequence(seed)
            child_seeds = ss.spawn(mpi.size())
        else:
            child_seeds = None
        ss = mpi.scatter(child_seeds)
        self._entropy = ss.entropy  # for debugging store for reproducibility
        self._rng = default_rng(ss)

    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        """
        Returns a list of tuples ``(regexp, root)`` of output files potentially produced.
        If ``root`` in the tuple is ``None``, ``output.folder`` is used. If ``regexp``
        part of the tuple is ``None``, all files inside ``root`` will be considered as
        output files of the sampler.

        If `minimal=True`, returns regexp's for the files that should really not be there
        when we are not resuming.
        """
        return []

    @classmethod
    @mpi.root_only
    def delete_output_files(cls, output, info=None):
        if output:
            for regexp, root in cls.output_files_regexps(output, info=info):
                # Special case: CovmatSampler's may have been given a covmat with the same
                # name that the output one. In that case, don't delete it!
                if issubclass(cls, CovmatSampler) and info:
                    if regexp.pattern.rstrip("$").endswith(Extension.covmat):
                        covmat_file = info.get("covmat", "")
                        if (
                            isinstance(covmat_file, str)
                            and covmat_file
                            == getattr(regexp.match(covmat_file), "group", lambda: None)()
                        ):
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
        resuming: bool | None
        if mpi.is_main_process():
            resuming = False
            if output.force:
                cls.delete_output_files(output, info=info)
            elif any(
                find_with_regexp(regexp, root or output.folder)
                for (regexp, root) in cls.output_files_regexps(
                    output=output, info=info, minimal=True
                )
            ):
                if output.is_resuming():
                    output.log.info("Found an old sample. Resuming.")
                    resuming = True
                else:
                    raise LoggedError(
                        output.log,
                        "Delete the previous output manually, automatically "
                        "('-%s', '--%s', '%s: True')"
                        % ("force"[0], "force", "force")
                        + " or request resuming ('-{}', '--{}', '{}: True')".format(
                            "resume"[0], "resume", "resume"
                        ),
                    )
            else:
                if output.is_resuming():
                    output.log.info(
                        "Did not find an old sample. Cleaning up and starting anew."
                    )
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
    # Amount by which to shrink covmat diagonals when set from priors or reference.
    fallback_covmat_scale: float = 4

    @mpi.from_root
    def _load_covmat(self, prefer_load_old, auto_params=None):
        if prefer_load_old and os.path.exists(self.covmat_filename()):
            covmat = np.atleast_2d(np.loadtxt(self.covmat_filename()))
            self.mpi_info("Covariance matrix from previous sample.")
            return covmat, []
        else:
            return self.initial_proposal_covmat(auto_params=auto_params)

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
        self.covmat = getattr(self, "covmat", None)
        if isinstance(self.covmat, str) and self.covmat.lower() == "auto":
            params_infos_covmat = deepcopy_where_possible(params_infos)
            if auto_params is not None:
                for p in list(params_infos_covmat):
                    if p not in auto_params:
                        params_infos_covmat.pop(p, None)
            auto_covmat = self.model.get_auto_covmat(params_infos_covmat)
            if auto_covmat:
                self.covmat = os.path.join(auto_covmat["folder"], auto_covmat["name"])
                self.log.info("Covariance matrix selected automatically: %s", self.covmat)
            else:
                self.covmat = None
                self.log.info(
                    "Could not automatically find a good covmat. "
                    "Will generate from parameter info (proposal and prior)."
                )
        # If given, load and test the covariance matrix
        loaded_params: Sequence[str]
        if isinstance(self.covmat, str):
            covmat_pre = "{%s}" % packages_path_input
            if self.covmat.startswith(covmat_pre):
                self.covmat = self.covmat.format(
                    **{packages_path_input: self.packages_path}
                ).replace("/", os.sep)
            try:
                with open(self.covmat, encoding="utf-8-sig") as file_covmat:
                    header = file_covmat.readline()
                loaded_covmat = np.loadtxt(self.covmat)
                self.log.debug("Loaded a covariance matrix from '%r'", self.covmat)
            except TypeError as texcpt:
                raise LoggedError(
                    self.log,
                    "The property 'covmat' must be a file name, but it's '%s'.",
                    str(self.covmat),
                ) from texcpt
            except OSError as ioexcpt:
                raise LoggedError(
                    self.log,
                    "Can't open covmat file '%s'.",
                    self.covmat,
                ) from ioexcpt
            if header[0] != "#":
                raise LoggedError(
                    self.log,
                    "The first line of the covmat file '%s' "
                    "must be one list of parameter names separated by spaces "
                    "and staring with '#', and the rest must be a square "
                    "matrix, with one row per line.",
                    self.covmat,
                )
            loaded_params = header.strip("#").strip().split()
        elif hasattr(self.covmat, "__getitem__"):
            if not self.covmat_params:
                raise LoggedError(
                    self.log,
                    "If a covariance matrix is passed as a numpy array, "
                    "you also need to pass the parameters it corresponds to "
                    "via 'covmat_params: [name1, name2, ...]'.",
                )
            loaded_params = self.covmat_params
            loaded_covmat = np.array(self.covmat)
        elif self.covmat:
            raise LoggedError(self.log, "Invalid covmat")
        if self.covmat is not None:
            str_msg = "the `covmat_params` list"
            if isinstance(self.covmat, str):
                str_msg = "the header of the covmat file %r" % self.covmat
            if len(loaded_params) != len(set(loaded_params)):
                duplicated = list(
                    {p for p in loaded_params if list(loaded_params).count(p) > 1}
                )
                raise LoggedError(
                    self.log,
                    "Parameter(s) %r appear more than once in %s",
                    duplicated,
                    str_msg,
                )
            loaded_covmat = np.atleast_2d(loaded_covmat)
            if len(loaded_params) != loaded_covmat.shape[0]:
                raise LoggedError(
                    self.log,
                    "The number of parameters in %s and the "
                    "dimensions of the matrix do not agree: %d vs %r",
                    str_msg,
                    len(loaded_params),
                    loaded_covmat.shape,
                )
            is_square_symmetric = (
                len(loaded_covmat.shape) == 2
                and loaded_covmat.shape[0] == loaded_covmat.shape[1]
                and np.allclose(loaded_covmat.T, loaded_covmat)
            )
            # Not checking for positive-definiteness yet: may contain highly degenerate
            # derived parameters that would spoil it now, but will later be dropped.
            if not is_square_symmetric:
                from_msg = (
                    f"loaded from '{self.covmat}'"
                    if isinstance(self.covmat, str)
                    else "passed"
                )
                raise LoggedError(
                    self.log,
                    f"The covariance matrix {from_msg} is not a symmetric square matrix.",
                )
            # Fill with parameters in the loaded covmat
            renames = {
                p: [p] + str_to_list(v.get("renames") or [])
                for p, v in params_infos.items()
            }
            indices_used, indices_sampler = zip(
                *[
                    [
                        loaded_params.index(p),
                        [
                            list(params_infos).index(q)
                            for q, a in renames.items()
                            if p in a
                        ],
                    ]
                    for p in loaded_params
                ]
            )
            if not any(indices_sampler):
                raise LoggedError(
                    self.log,
                    "A proposal covariance matrix has been loaded, but none of its "
                    "parameters are actually sampled here. Maybe a mismatch between"
                    " parameter names in the covariance matrix and the input file?",
                )
            indices_used, indices_sampler = zip(
                *[[i, j] for i, j in zip(indices_used, indices_sampler) if j]
            )
            if any(len(j) - 1 for j in indices_sampler):
                first = next(j for j in indices_sampler if len(j) > 1)
                raise LoggedError(
                    self.log,
                    "The parameters %s have duplicated aliases. Can't assign them an "
                    "element of the covariance matrix unambiguously.",
                    ", ".join([list(params_infos)[i] for i in first]),
                )
            indices_sampler = tuple(chain(*indices_sampler))
            covmat[np.ix_(indices_sampler, indices_sampler)] = loaded_covmat[
                np.ix_(indices_used, indices_used)
            ]
            self.log.info(
                "Covariance matrix loaded for params %r",
                [list(params_infos)[i] for i in indices_sampler],
            )
            missing_params = set(params_infos).difference(
                list(params_infos)[i] for i in indices_sampler
            )
            if missing_params:
                self.log.info(
                    "Missing proposal covariance for params %r",
                    [
                        p
                        for p in self.model.parameterization.sampled_params()
                        if p in missing_params
                    ],
                )
            else:
                self.log.info("All parameters' covariance loaded from given covmat.")
        # Fill gaps with "proposal" property, if present, otherwise ref (or prior)
        where_nan = np.isnan(covmat.diagonal())
        if np.any(where_nan):
            covmat[where_nan, where_nan] = np.array(
                [
                    (info.get("proposal", np.nan) or np.nan) ** 2
                    for info in params_infos.values()
                ]
            )[where_nan]
        where_nan2 = np.isnan(covmat.diagonal())
        if np.any(where_nan2):
            # the variances are likely too large for a good proposal, e.g. conditional
            # widths may be much smaller than the marginalized ones.
            # Divide by 4, better to be too small than too large.
            covmat[where_nan2, where_nan2] = (
                self.model.prior.reference_variances()[where_nan2]
                / self.fallback_covmat_scale
            )
        assert not np.any(np.isnan(covmat))
        return covmat, where_nan

    def covmat_filename(self):
        if self.output:
            return os.path.join(self.output.folder, self.output.prefix + Extension.covmat)
        return None

    def dump_covmat(self, covmat=None):
        if covmat is None:
            covmat = self.covmat
        np.savetxt(
            self.covmat_filename(),
            covmat,
            header=" ".join(list(self.model.parameterization.sampled_params())),
        )
