"""
.. module:: samplers.polychord

:Synopsis: Interface for the PolyChord nested sampler
:Author: Will Handley, Mike Hobson and Anthony Lasenby (for PolyChord),
         Jesus Torrado (for the cobaya wrapper only)
"""

import inspect
import logging
import os
import re
import sys
import warnings
from collections.abc import Callable
from itertools import chain
from tempfile import gettempdir
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from cobaya.collection import SampleCollection
from cobaya.component import ComponentNotInstalledError, load_external_module
from cobaya.conventions import Extension, derived_par_name_separator
from cobaya.install import download_github_release
from cobaya.log import LoggedError, NoLogging, get_logger
from cobaya.mpi import is_main_process, share_mpi, sync_processes
from cobaya.sampler import Sampler
from cobaya.tools import (
    NumberWithUnits,
    find_with_regexp,
    get_compiled_import_path,
    get_external_function,
    read_dnumber,
)
from cobaya.yaml import yaml_dump_file

# Avoid importing GetDist if not necessary
if TYPE_CHECKING:
    from getdist import MCSamples


class polychord(Sampler):
    r"""
    PolyChord sampler \cite{Handley:2015fda,2015MNRAS.453.4384H}, a nested sampler
    tailored for high-dimensional parameter spaces with a speed hierarchy.
    """

    sampler_type: str = "nested"

    # Name of the PolyChord repo and version to download
    _pc_repo_name = "PolyChord/PolyChordLite"
    _pc_repo_version = "1.20.1"
    _base_dir_suffix = "polychord_raw"
    _clusters_dir = "clusters"
    _at_resume_prefer_old = Sampler._at_resume_prefer_old + ["blocking"]
    _at_resume_prefer_new = Sampler._at_resume_prefer_new + ["callback_function"]
    pypolychord: Any

    # variables from yaml
    do_clustering: bool
    num_repeats: int | str
    confidence_for_unbounded: float
    callback_function: Callable
    blocking: Any
    measure_speeds: bool
    oversample_power: float
    nlive: NumberWithUnits
    path: str
    logzero: float
    max_ndead: int

    def initialize(self):
        """Imports the PolyChord sampler and prepares its arguments."""
        install_path = self.get_path(self.packages_path) if self.packages_path else None
        try:
            self.pc = load_external_module(
                "pypolychord",
                path=self.path,
                install_path=install_path,
                min_version=self._pc_repo_version,
                get_import_path=get_compiled_import_path,
                logger=self.log,
                not_installed_level="debug",
            )
        except ComponentNotInstalledError as excpt:
            raise ComponentNotInstalledError(
                self.log,
                (
                    f"Could not find PolyChord: {excpt}. "
                    "To install it, run `cobaya-install polychord`"
                ),
            ) from excpt
        with NoLogging(logging.CRITICAL):
            settings = load_external_module(
                "pypolychord.settings",
                path=self.path,
                install_path=install_path,
                get_import_path=get_compiled_import_path,
                logger=self.log,
            )
        # Prepare arguments and settings
        self.n_sampled = len(self.model.parameterization.sampled_params())
        self.n_derived = len(self.model.parameterization.derived_params())
        self.n_priors = len(self.model.prior)
        self.n_likes = len(self.model.likelihood)
        self.nDims = self.model.prior.d()
        self.nDerived = self.n_derived + self.n_priors + self.n_likes
        if self.logzero is None:
            self.logzero = np.nan_to_num(-np.inf)
        if self.max_ndead == np.inf:
            self.max_ndead = -1
        self._quants_d_units = ["nlive", "max_ndead"]
        for p in self._quants_d_units:
            if getattr(self, p) is not None:
                setattr(
                    self,
                    p,
                    NumberWithUnits(
                        getattr(self, p), "d", scale=self.nDims, dtype=int
                    ).value,
                )
        self._quants_nlive_units = ["nprior", "nfail"]
        for p in self._quants_nlive_units:
            if getattr(self, p) is not None:
                setattr(
                    self,
                    p,
                    NumberWithUnits(
                        getattr(self, p), "nlive", scale=self.nlive, dtype=int
                    ).value,
                )
        # Fill the automatic ones
        if getattr(self, "feedback", None) is None:
            values = {
                logging.CRITICAL: 0,
                logging.ERROR: 0,
                logging.WARNING: 0,
                logging.INFO: 1,
                logging.DEBUG: 2,
            }
            self.feedback = values[self.log.getEffectiveLevel()]
        # Prepare output folders and prefixes
        if self.output:
            self.file_root = self.output.prefix
            self.read_resume = self.output.is_resuming()
        else:
            output_prefix = share_mpi(
                hex(int(self._rng.random() * 16**6))[2:] if is_main_process() else None
            )
            self.file_root = output_prefix
            # dummy output -- no resume!
            self.read_resume = False
        self.base_dir = self.get_base_dir(self.output)
        self.raw_clusters_dir = os.path.join(self.base_dir, self._clusters_dir)
        self.output.create_folder(self.base_dir)
        if self.do_clustering:
            self.clusters_folder = self.get_clusters_dir(self.output)
            self.output.create_folder(self.clusters_folder)
        self.mpi_info("Storing raw PolyChord output in '%s'.", self.base_dir)
        # Exploiting the speed hierarchy
        if self.blocking:
            blocks, oversampling_factors = self.model.check_blocking(self.blocking)
        else:
            if self.measure_speeds:
                self.model.measure_and_set_speeds(
                    n=self.measure_speeds, random_state=self._rng
                )
            blocks, oversampling_factors = self.model.get_param_blocking_for_sampler(
                oversample_power=self.oversample_power
            )
        self.mpi_info("Parameter blocks and their oversampling factors:")
        max_width = len(str(max(oversampling_factors)))
        for f, b in zip(oversampling_factors, blocks):
            self.mpi_info("* %" + "%d" % max_width + "d : %r", f, b)
        # Save blocking in updated info, in case we want to resume
        self._updated_info["blocking"] = list(zip(oversampling_factors, blocks))
        blocks_flat = list(chain(*blocks))
        self.ordering = [
            blocks_flat.index(p) for p in self.model.parameterization.sampled_params()
        ]
        self.grade_dims = [len(block) for block in blocks]
        # Steps per block
        # NB: num_repeats is ignored by PolyChord when int "grade_frac" given,
        # so needs to be applied by hand.
        # In num_repeats, `d` is interpreted as dimension of each block
        self.grade_frac = [
            int(o * read_dnumber(self.num_repeats, dim_block))
            for o, dim_block in zip(oversampling_factors, self.grade_dims)
        ]
        # Assign settings
        pc_args = [
            "nlive",
            "num_repeats",
            "nprior",
            "nfail",
            "do_clustering",
            "feedback",
            "precision_criterion",
            "logzero",
            "max_ndead",
            "boost_posterior",
            "posteriors",
            "equals",
            "cluster_posteriors",
            "write_resume",
            "read_resume",
            "write_stats",
            "write_live",
            "write_dead",
            "write_prior",
            "maximise",
            "compression_factor",
            "synchronous",
            "base_dir",
            "file_root",
            "grade_dims",
            "grade_frac",
            "nlives",
        ]
        # As stated above, num_repeats is ignored, so let's not pass it
        pc_args.pop(pc_args.index("num_repeats"))
        self.pc_settings = settings.PolyChordSettings(
            self.nDims,
            self.nDerived,
            seed=(self.seed if self.seed is not None else -1),
            **{p: getattr(self, p) for p in pc_args if getattr(self, p) is not None},
        )
        # prior conversion from the hypercube
        bounds = self.model.prior.bounds(
            confidence_for_unbounded=self.confidence_for_unbounded
        )
        # Check if priors are bounded (nan's to inf)
        inf = np.where(np.isinf(bounds))
        if len(inf[0]):
            params_names = list(self.model.parameterization.sampled_params())
            params = [params_names[i] for i in sorted(set(inf[0]))]
            raise LoggedError(
                self.log,
                "PolyChord needs bounded priors, but the parameter(s) '', '".join(params)
                + "' is(are) unbounded.",
            )
        locs = bounds[:, 0]
        scales = bounds[:, 1] - bounds[:, 0]
        # This function re-scales the parameters AND puts them in the right order
        self.pc_prior = lambda x: (locs + np.array(x)[self.ordering] * scales).tolist()
        # We will need the volume of the prior domain, since PolyChord divides by it
        self.logvolume = np.log(np.prod(scales))
        # Prepare callback function
        if self.callback_function is not None:
            self.callback_function_callable = get_external_function(
                self.callback_function
            )
        self.last_point_callback = 0
        # Prepare runtime live and dead points collections
        self.live = SampleCollection(self.model, None, name="live")
        self.dead = SampleCollection(self.model, self.output, name="dead")
        # Done!
        if is_main_process():
            self.log.debug("Calling PolyChord with arguments:")
            for p, v in inspect.getmembers(self.pc_settings, lambda a: not callable(a)):
                if not p.startswith("_"):
                    self.log.debug("  %s: %s", p, v)
        self.logZ, self.logZstd = np.nan, np.nan
        self._frac_unphysical = np.nan
        self.collection = None
        self.clusters = None

    def dumper(self, live_points, dead_points, logweights, logZ, logZstd):
        """
        Preprocess output for the callback function and calls it, if present.
        """
        if self.callback_function is None:
            return
        # Store live and dead points and evidence computed so far
        self.live.reset()
        for point in live_points:
            self.live.add(
                point[: self.n_sampled],
                derived=point[self.n_sampled : self.n_sampled + self.n_derived],
                weight=np.nan,
                logpriors=point[
                    self.n_sampled + self.n_derived : self.n_sampled
                    + self.n_derived
                    + self.n_priors
                ],
                loglikes=point[
                    self.n_sampled + self.n_derived + self.n_priors : self.n_sampled
                    + self.n_derived
                    + self.n_priors
                    + self.n_likes
                ],
            )
        for logweight, point in zip(
            logweights[self.last_point_callback :],
            dead_points[self.last_point_callback :],
        ):
            self.dead.add(
                point[: self.n_sampled],
                derived=point[self.n_sampled : self.n_sampled + self.n_derived],
                weight=np.exp(logweight),
                logpriors=point[
                    self.n_sampled + self.n_derived : self.n_sampled
                    + self.n_derived
                    + self.n_priors
                ],
                loglikes=point[
                    self.n_sampled + self.n_derived + self.n_priors : self.n_sampled
                    + self.n_derived
                    + self.n_priors
                    + self.n_likes
                ],
            )
        self.logZ, self.logZstd = logZ, logZstd
        self._correct_unphysical_fraction()
        # Callback function
        if self.callback_function is not None:
            try:
                self.callback_function_callable(self)
            except Exception as e:
                self.log.error("The callback function produced an error: %r", str(e))
            self.last_point_callback = len(self.dead)

    def run(self):
        """
        Prepares the prior and likelihood functions, calls ``PolyChord``'s ``run``, and
        processes its output.
        """
        # Prepare the posterior
        # Don't forget to multiply by the volume of the physical hypercube,
        # since PolyChord divides by it

        def logpost(params_values):
            result = self.model.logposterior(params_values)
            loglikes = result.loglikes
            if len(loglikes) != self.n_likes:
                loglikes = np.full(self.n_likes, np.nan)
            derived = result.derived
            if len(derived) != self.n_derived:
                derived = np.full(self.n_derived, np.nan)
            derived = list(derived) + list(result.logpriors) + list(loglikes)
            return (
                max(result.logpost + self.logvolume, self.pc_settings.logzero),
                derived,
            )

        sync_processes()
        self.mpi_info("Calling PolyChord...")
        self.pc.run_polychord(
            logpost,
            self.nDims,
            self.nDerived,
            self.pc_settings,
            self.pc_prior,
            self.dumper,
        )
        self.process_raw_output()

    @property
    def raw_prefix(self):
        return os.path.join(self.pc_settings.base_dir, self.pc_settings.file_root)

    def dump_paramnames(self, prefix):
        labels = self.model.parameterization.labels()
        with open(prefix + ".paramnames", "w", encoding="utf-8-sig") as f_paramnames:
            for p in self.model.parameterization.sampled_params():
                f_paramnames.write("{}\t{}\n".format(p, labels.get(p, "")))
            for p in self.model.parameterization.derived_params():
                f_paramnames.write("{}*\t{}\n".format(p, labels.get(p, "")))
            for p in self.model.prior:
                f_paramnames.write(
                    "{}*\t{}\n".format(
                        "logprior" + derived_par_name_separator + p,
                        r"\log\pi_\mathrm{" + p.replace("_", r"\ ") + r"}",
                    )
                )
            for p in self.model.likelihood:
                f_paramnames.write(
                    "{}*\t{}\n".format(
                        "loglike" + derived_par_name_separator + p,
                        r"\log\mathcal{L}_\mathrm{" + p.replace("_", r"\ ") + r"}",
                    )
                )

    def save_sample(self, fname, name):
        with warnings.catch_warnings():  # in case of empty file
            warnings.simplefilter("ignore")
            sample = np.atleast_2d(np.loadtxt(fname))
        if not sample.size:
            return None
        collection = SampleCollection(
            self.model, self.output, name=str(name), sample_type="nested"
        )
        for row in sample:
            collection.add(
                row[2 : 2 + self.n_sampled],
                derived=row[2 + self.n_sampled : 2 + self.n_sampled + self.n_derived],
                weight=row[0],
                logpriors=row[-(self.n_priors + self.n_likes) : -self.n_likes],
                loglikes=row[-self.n_likes :],
            )
        # make sure that the points are written
        collection.out_update()
        return collection

    def _correct_unphysical_fraction(self):
        """
        Correction for the fraction of the prior that is unphysical -- see issue #77
        """
        if np.isnan(self._frac_unphysical):
            with open(self.raw_prefix + ".prior_info", encoding="utf-8-sig") as pf:
                lines = list(pf.readlines())

            def get_value_str(line):
                return line[line.find("=") + 1 :]

            def get_value_str_var(var):
                return get_value_str(
                    next(line for line in lines if line.lstrip().startswith(var))
                )

            nprior = int(get_value_str_var("nprior"))
            ndiscarded = int(get_value_str_var("ndiscarded"))
            self._frac_unphysical = nprior / ndiscarded
        if self._frac_unphysical != 1:
            self.log.debug(
                "Correcting for unphysical region fraction: %g", self._frac_unphysical
            )
            self.logZ += np.log(self._frac_unphysical)
            if self.clusters is not None:
                for cluster in self.clusters.values():
                    cluster["logZ"] += np.log(self._frac_unphysical)

    def process_raw_output(self):
        """
        Loads the sample of live points from ``PolyChord``'s raw output and writes it
        (if ``txt`` output requested).
        """
        if not is_main_process():
            return
        self.log.info("Loading PolyChord's results: samples and evidences.")
        self.dump_paramnames(self.raw_prefix)
        self.collection = self.save_sample(self.raw_prefix + ".txt", "1")
        # Load clusters, and save if output
        if self.pc_settings.do_clustering:
            self.clusters = {}
            clusters_raw_regexp = re.compile(
                re.escape(self.pc_settings.file_root + "_") + r"\d+\.txt"
            )
            cluster_raw_files = sorted(
                find_with_regexp(
                    clusters_raw_regexp,
                    os.path.join(self.pc_settings.base_dir, self._clusters_dir),
                    walk_tree=True,
                )
            )
            for f in cluster_raw_files:
                i = int(f[f.rfind("_") + 1 : -len(".txt")])
                if self.output:
                    old_folder = self.output.folder
                    self.output.folder = self.clusters_folder
                sample = self.save_sample(f, str(i))
                if self.output:
                    self.output.folder = old_folder
                self.clusters[i] = {"sample": sample}
        # Prepare the evidence(s) and write to file
        pre = "log(Z"
        active = "(Still active)"
        with open(self.raw_prefix + ".stats", encoding="utf-8-sig") as statsfile:
            lines = [line for line in statsfile.readlines() if line.startswith(pre)]
        for line in lines:
            logZ, logZstd = (
                float(n.replace(active, "")) for n in line.split("=")[-1].split("+/-")
            )
            component = line.split("=")[0].lstrip(pre + "_").rstrip(") ")
            if not component:
                self.logZ, self.logZstd = logZ, logZstd
            elif self.pc_settings.do_clustering:
                i = int(component)
                self.clusters[i]["logZ"], self.clusters[i]["logZstd"] = logZ, logZstd
        with warnings.catch_warnings():  # evidence too large (overflow)
            warnings.simplefilter("ignore")
            self.log.debug(
                "RAW log(Z) = %g +/- %g ; RAW Z in [%.8g, %.8g] (68%% C.L. log-gaussian)",
                self.logZ,
                self.logZstd,
                *[np.exp(self.logZ + n * self.logZstd) for n in [-1, 1]],
            )
        self._correct_unphysical_fraction()
        if self.output:
            out_evidences = {"logZ": self.logZ, "logZstd": self.logZstd}
            if self.clusters is not None:
                out_evidences["clusters"] = {}
                for i in sorted(list(self.clusters)):
                    out_evidences["clusters"][i] = {
                        "logZ": self.clusters[i]["logZ"],
                        "logZstd": self.clusters[i]["logZstd"],
                    }
            fname = os.path.join(
                self.output.folder, self.output.prefix + Extension.evidence
            )
            yaml_dump_file(
                fname, out_evidences, comment="log-evidence", error_if_exists=False
            )
        self.log.info(
            "Finished! Raw PolyChord output stored in '%s', with prefix '%s'",
            self.pc_settings.base_dir,
            self.pc_settings.file_root,
        )
        with warnings.catch_warnings():  # evidence too large (overflow)
            warnings.simplefilter("ignore")
            self.log.info(
                "log(Z) = %g +/- %g ; Z in [%.8g, %.8g] (68%% C.L. log-gaussian)",
                self.logZ,
                self.logZstd,
                *[np.exp(self.logZ + n * self.logZstd) for n in [-1, 1]],
            )

    def samples(
        self,
        combined: bool = False,
        skip_samples: float = 0,
        to_getdist: bool = False,
    ) -> Union[SampleCollection, "MCSamples", None]:
        """
        Returns the sample of the posterior built out of dead points.

        Parameters
        ----------
        combined: bool, default: False
            If ``True`` returns the same, single posterior for all processes. Otherwise,
            it is only returned for the root process (this behaviour is kept for
            compatibility with the equivalent function for MCMC).
        skip_samples: int or float, default: 0
            No effect (skipping initial samples from a sorted nested sampling sample would
            bias it). Raises a warning if greater than 0.
        to_getdist: bool, default: False
            If ``True``, returns a single :class:`getdist.MCSamples` instance, containing
            all samples, for all MPI processes (``combined`` is ignored).

        Returns
        -------
        SampleCollection, getdist.MCSamples
           The posterior sample.
        """
        if skip_samples:
            self.mpi_warning(
                "Initial samples should not be skipped in nested sampling. "
                "Ignoring 'skip_samples' keyword."
            )
        collection = self.collection
        if not combined and not to_getdist:
            return collection  # None for MPI ranks > 0
        # In all remaining cases, we return the same for all ranks
        if to_getdist:
            if is_main_process():
                collection = collection.to_getdist()
        return share_mpi(collection)

    def samples_clusters(
        self,
        to_getdist: bool = False,
    ) -> None | dict[int, Union[SampleCollection, "MCSamples", None]]:
        """
        Returns the samples corresponding to all clusters, if doing clustering, or
        ``None`` otherwise.

        Parameters
        ----------
        to_getdist: bool, default: False
            If ``True``, returns the cluster samples as :class:`getdist.MCSamples`.

        Returns
        -------
        None, dict[int, Union[SampleCollection, MCSamples, None]]
           The cluster posterior samples.
        """
        if not self.pc_settings.do_clustering:
            return None
        if not is_main_process():
            return None
        clusters: dict[int, Union[SampleCollection, "MCSamples", None]] = {}
        for i, c in self.clusters.items():
            if to_getdist:
                try:
                    clusters[i] = c["sample"].to_getdist()
                except (ValueError, AttributeError):
                    self.log.warning(
                        "Cluster #%d could not be converted to a GetDist sample. "
                        "Storing 'None'.",
                        i,
                    )
                    clusters[i] = None
            else:
                clusters[i] = c["sample"]
        return clusters

    def products(
        self,
        combined: bool = False,
        skip_samples: float = 0,
        to_getdist: bool = False,
    ) -> dict:
        """
        Returns the products of the sampling process.

        Parameters
        ----------
        combined: bool, default: False
            If ``True`` returns the same, single posterior for all processes. Otherwise,
            it is only returned for the root process (this behaviour is kept for
            compatibility with the equivalent function for MCMC).
        skip_samples: int or float, default: 0
            No effect (skipping initial samples from a sorted nested sampling sample would
            bias it). Raises a warning if greater than 0.
        to_getdist: bool, default: False
            If ``True``, returns :class:`getdist.MCSamples` instances for the full
            posterior sample and the clusters, for all MPI processes (``combined`` is
            ignored).

        Returns
        -------
        dict, None
            A dictionary containing the :class:`cobaya.collection.SampleCollection` of
            accepted steps under ``"sample"``, the log-evidence and its uncertainty
            under ``logZ`` and ``logZstd`` respectively, and the same for the individual
            clusters, if present, under the ``clusters`` key.

        Notes
        -----
        If either ``combined`` or ``to_getdist`` are ``True``, the same products dict is
        returned for all processes. Otherwise, ``None`` is returned for processes of rank
        larger than 0.
        """
        products = {}
        if is_main_process():
            products = {
                "logZ": self.logZ,
                "logZstd": self.logZstd,
                "sample": self.samples(
                    combined=combined, skip_samples=skip_samples, to_getdist=to_getdist
                ),
            }
            if self.pc_settings.do_clustering:
                products["clusters"] = {i: {} for i in self.clusters}
                for i, s in self.samples_clusters(to_getdist=to_getdist).items():
                    products["clusters"][i]["logZ"] = self.clusters[i]["logZ"]
                    products["clusters"][i]["logZstd"] = self.clusters[i]["logZstd"]
                    products["clusters"][i]["sample"] = s
        do_bcast = combined or to_getdist
        if do_bcast:
            return share_mpi(products)
        return products

    @classmethod
    def get_base_dir(cls, output):
        if output:
            return output.add_suffix(cls._base_dir_suffix, separator="_")
        return os.path.join(gettempdir(), cls._base_dir_suffix)

    @classmethod
    def get_clusters_dir(cls, output):
        if output:
            return output.add_suffix(cls._clusters_dir, separator="_")

    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        # Resume file
        regexps_tuples = [
            (re.compile(re.escape(output.prefix + ".resume")), cls.get_base_dir(output))
        ]
        if minimal:
            return regexps_tuples
        return regexps_tuples + [
            # Raw products base dir
            (None, cls.get_base_dir(output)),
            # Main sample
            (output.collection_regexp(name=None), None),
            # Evidence
            (re.compile(re.escape(output.prefix + Extension.evidence)), None),
            # Clusters
            (None, cls.get_clusters_dir(output)),
        ]

    @classmethod
    def get_version(cls):
        return None

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(
            os.path.join(
                path, "code", cls._pc_repo_name[cls._pc_repo_name.find("/") + 1 :]
            )
        )

    @classmethod
    def is_compatible(cls):
        import platform

        if platform.system() == "Windows":
            return False
        return True

    @classmethod
    def is_installed(cls, reload=False, **kwargs):
        if not kwargs.get("code", True):
            return True
        try:
            return bool(
                load_external_module(
                    "pypolychord",
                    path=kwargs["path"],
                    get_import_path=get_compiled_import_path,
                    min_version=cls._pc_repo_version,
                    reload=reload,
                    logger=get_logger(cls.__name__),
                    not_installed_level="debug",
                )
            )
        except ComponentNotInstalledError:
            return False

    @classmethod
    def install(cls, path=None, code=False, no_progress_bars=False, **_kwargs):
        if not code:
            return True
        log = get_logger(__name__)
        log.info("Downloading PolyChord...")
        success = download_github_release(
            os.path.join(path, "code"),
            cls._pc_repo_name,
            cls._pc_repo_version,
            no_progress_bars=no_progress_bars,
            logger=log,
        )
        if not success:
            log.error("Could not download PolyChord.")
            return False
        log.info("Compiling (Py)PolyChord...")
        from subprocess import PIPE, Popen

        # Needs to re-define os' PWD,
        # because MakeFile calls it and is not affected by the cwd of Popen
        cwd = os.path.join(
            path, "code", cls._pc_repo_name[cls._pc_repo_name.find("/") + 1 :]
        )
        my_env = os.environ.copy()
        my_env.update({"PWD": cwd})
        if "CC" not in my_env:
            my_env["CC"] = "mpicc"
        if "CXX" not in my_env:
            my_env["CXX"] = "mpicxx"
        process_make = Popen(
            [sys.executable, "setup.py", "build"],
            cwd=cwd,
            env=my_env,
            stdout=PIPE,
            stderr=PIPE,
        )
        out, err = process_make.communicate()
        if process_make.returncode:
            log.info(out.decode("utf-8"))
            log.info(err.decode("utf-8"))
            log.error("Python build failed!")
            return False
        return True
