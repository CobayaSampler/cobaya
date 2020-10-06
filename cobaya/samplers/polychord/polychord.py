"""
.. module:: samplers.polychord

:Synopsis: Interface for the PolyChord nested sampler
:Author: Will Handley, Mike Hobson and Anthony Lasenby (for PolyChord),
         Jesus Torrado (for the cobaya wrapper only)
"""
# Global
import os
import sys
import numpy as np
import logging
import inspect
from itertools import chain
from random import random
from typing import Any
from tempfile import gettempdir
import re

# Local
from cobaya.tools import read_dnumber, get_external_function, PythonPath, \
    find_with_regexp, NumberWithUnits, load_module, VersionCheckError
from cobaya.sampler import Sampler
from cobaya.mpi import is_main_process, share_mpi, sync_processes
from cobaya.collection import Collection
from cobaya.log import LoggedError
from cobaya.install import download_github_release, NotInstalledError
from cobaya.yaml import yaml_dump_file
from cobaya.conventions import _separator, _evidence_extension, _packages_path_arg


class polychord(Sampler):
    # Name of the PolyChord repo and version to download
    _pc_repo_name = "PolyChord/PolyChordLite"
    _pc_repo_version = "1.17.1"
    _base_dir_suffix = "polychord_raw"
    _clusters_dir = "clusters"
    _at_resume_prefer_old = Sampler._at_resume_prefer_old + ["blocking"]
    _at_resume_prefer_new = Sampler._at_resume_prefer_new + ["callback_function"]

    # variables from yaml
    do_clustering: bool
    num_repeats: int
    confidence_for_unbounded: float
    callback_function: callable
    blocking: Any
    measure_speeds: bool
    oversample_power: float
    nlive: NumberWithUnits

    def initialize(self):
        """Imports the PolyChord sampler and prepares its arguments."""
        # Allow global import if no direct path specification
        allow_global = not self.path
        if not self.path and self.packages_path:
            self.path = self.get_path(self.packages_path)
        self.pc = self.is_installed(path=self.path, allow_global=allow_global)
        if not self.pc:
            raise NotInstalledError(
                self.log, "Could not find PolyChord. Check error message above. "
                          "To install it, run 'cobaya-install polychord --%s "
                          "[packages_path]'", _packages_path_arg)
        # Prepare arguments and settings
        from pypolychord.settings import PolyChordSettings
        self.n_sampled = len(self.model.parameterization.sampled_params())
        self.n_derived = len(self.model.parameterization.derived_params())
        self.n_priors = len(self.model.prior)
        self.n_likes = len(self.model.likelihood)
        self.nDims = self.model.prior.d()
        self.nDerived = (self.n_derived + self.n_priors + self.n_likes)
        if self.logzero is None:
            self.logzero = np.nan_to_num(-np.inf)
        if self.max_ndead == np.inf:
            self.max_ndead = -1
        self._quants_d_units = ["nlive", "max_ndead"]
        for p in self._quants_d_units:
            if getattr(self, p) is not None:
                setattr(self, p, NumberWithUnits(
                    getattr(self, p), "d", scale=self.nDims, dtype=int).value)
        self._quants_nlive_units = ["nprior"]
        for p in self._quants_nlive_units:
            if getattr(self, p) is not None:
                setattr(self, p, NumberWithUnits(
                    getattr(self, p), "nlive", scale=self.nlive, dtype=int).value)
        # Fill the automatic ones
        if getattr(self, "feedback", None) is None:
            values = {logging.CRITICAL: 0, logging.ERROR: 0, logging.WARNING: 0,
                      logging.INFO: 1, logging.DEBUG: 2}
            self.feedback = values[self.log.getEffectiveLevel()]
        # Prepare output folders and prefixes
        if self.output:
            self.file_root = self.output.prefix
            self.read_resume = self.output.is_resuming()
        else:
            output_prefix = share_mpi(hex(int(random() * 16 ** 6))[2:]
                                      if is_main_process() else None)
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
                self.model.measure_and_set_speeds(n=self.measure_speeds)
            blocks, oversampling_factors = self.model.get_param_blocking_for_sampler(
                oversample_power=self.oversample_power)
        self.mpi_info("Parameter blocks and their oversampling factors:")
        max_width = len(str(max(oversampling_factors)))
        for f, b in zip(oversampling_factors, blocks):
            self.mpi_info("* %" + "%d" % max_width + "d : %r", f, b)
        # Save blocking in updated info, in case we want to resume
        self._updated_info["blocking"] = list(zip(oversampling_factors, blocks))
        blocks_flat = list(chain(*blocks))
        self.ordering = [
            blocks_flat.index(p) for p in self.model.parameterization.sampled_params()]
        self.grade_dims = [len(block) for block in blocks]
        # Steps per block
        # NB: num_repeats is ignored by PolyChord when int "grade_frac" given,
        # so needs to be applied by hand.
        # In num_repeats, `d` is interpreted as dimension of each block
        self.grade_frac = [
            int(o * read_dnumber(self.num_repeats, dim_block))
            for o, dim_block in zip(oversampling_factors, self.grade_dims)]
        # Assign settings
        pc_args = ["nlive", "num_repeats", "nprior", "do_clustering",
                   "precision_criterion", "max_ndead", "boost_posterior", "feedback",
                   "logzero", "posteriors", "equals", "compression_factor",
                   "cluster_posteriors", "write_resume", "read_resume", "write_stats",
                   "write_live", "write_dead", "base_dir", "grade_frac", "grade_dims",
                   "feedback", "read_resume", "base_dir", "file_root", "grade_frac",
                   "grade_dims"]
        # As stated above, num_repeats is ignored, so let's not pass it
        pc_args.pop(pc_args.index("num_repeats"))
        self.pc_settings = PolyChordSettings(
            self.nDims, self.nDerived, seed=(self.seed if self.seed is not None else -1),
            **{p: getattr(self, p) for p in pc_args if getattr(self, p) is not None})
        # prior conversion from the hypercube
        bounds = self.model.prior.bounds(
            confidence_for_unbounded=self.confidence_for_unbounded)
        # Check if priors are bounded (nan's to inf)
        inf = np.where(np.isinf(bounds))
        if len(inf[0]):
            params_names = self.model.parameterization.sampled_params()
            params = [params_names[i] for i in sorted(list(set(inf[0])))]
            raise LoggedError(
                self.log, "PolyChord needs bounded priors, but the parameter(s) '"
                          "', '".join(params) + "' is(are) unbounded.")
        locs = bounds[:, 0]
        scales = bounds[:, 1] - bounds[:, 0]
        # This function re-scales the parameters AND puts them in the right order
        self.pc_prior = lambda x: (locs + np.array(x)[self.ordering] * scales).tolist()
        # We will need the volume of the prior domain, since PolyChord divides by it
        self.logvolume = np.log(np.prod(scales))
        # Prepare callback function
        if self.callback_function is not None:
            self.callback_function_callable = (
                get_external_function(self.callback_function))
        self.last_point_callback = 0
        # Prepare runtime live and dead points collections
        self.live = Collection(
            self.model, None, name="live", initial_size=self.pc_settings.nlive)
        self.dead = Collection(self.model, self.output, name="dead")
        # Done!
        if is_main_process():
            self.log.debug("Calling PolyChord with arguments:")
            for p, v in inspect.getmembers(self.pc_settings, lambda a: not (callable(a))):
                if not p.startswith("_"):
                    self.log.debug("  %s: %s", p, v)
        self.mpi_info("Initialized!")

    def dumper(self, live_points, dead_points, logweights, logZ, logZstd):
        # Store live and dead points and evidence computed so far
        self.live.reset()
        for point in live_points:
            self.live.add(
                point[:self.n_sampled],
                derived=point[self.n_sampled:self.n_sampled + self.n_derived],
                weight=np.nan,
                logpriors=point[self.n_sampled + self.n_derived:
                                self.n_sampled + self.n_derived + self.n_priors],
                loglikes=point[self.n_sampled + self.n_derived + self.n_priors:
                               self.n_sampled + self.n_derived + self.n_priors +
                               self.n_likes])
        for logweight, point in zip(logweights[self.last_point_callback:],
                                    dead_points[self.last_point_callback:]):
            self.dead.add(
                point[:self.n_sampled],
                derived=point[self.n_sampled:self.n_sampled + self.n_derived],
                weight=np.exp(logweight),
                logpriors=point[self.n_sampled + self.n_derived:
                                self.n_sampled + self.n_derived + self.n_priors],
                loglikes=point[self.n_sampled + self.n_derived + self.n_priors:
                               self.n_sampled + self.n_derived + self.n_priors +
                               self.n_likes])
        self.logZ, self.logZstd = logZ, logZstd
        self._correct_unphysical_fraction()
        # Callback function
        if self.callback_function is not None:
            try:
                self.callback_function_callable(self)
            except Exception as e:
                self.log.error("The callback function produced an error: %r", str(e))
            self.last_point_callback = len(self.dead)

    def _run(self):
        """
        Prepares the posterior function and calls ``PolyChord``'s ``run`` function.
        """

        # Prepare the posterior
        # Don't forget to multiply by the volume of the physical hypercube,
        # since PolyChord divides by it
        def logpost(params_values):
            logposterior, logpriors, loglikes, derived = (
                self.model.logposterior(params_values))
            if len(derived) != self.n_derived:
                derived = np.full(self.n_derived, np.nan)
            if len(loglikes) != self.n_likes:
                loglikes = np.full(self.n_likes, np.nan)
            derived = list(derived) + list(logpriors) + list(loglikes)
            return (
                max(logposterior + self.logvolume, self.pc_settings.logzero),
                derived)

        sync_processes()
        self.mpi_info("Calling PolyChord...")
        self.pc.run_polychord(logpost, self.nDims, self.nDerived, self.pc_settings,
                              self.pc_prior, self.dumper)
        self.process_raw_output()

    @property
    def raw_prefix(self):
        return os.path.join(
            self.pc_settings.base_dir, self.pc_settings.file_root)

    def dump_paramnames(self, prefix):
        paramnames = (list() +
                      [p + "*" for p in (
                              list(self.model.parameterization.derived_params()) +
                              list(self.model.prior) + list(self.model.likelihood))])
        labels = self.model.parameterization.labels()
        with open(prefix + ".paramnames", "w") as f_paramnames:
            for p in self.model.parameterization.sampled_params():
                f_paramnames.write("%s\t%s\n" % (p, labels.get(p, "")))
            for p in self.model.parameterization.derived_params():
                f_paramnames.write("%s*\t%s\n" % (p, labels.get(p, "")))
            for p in self.model.prior:
                f_paramnames.write("%s*\t%s\n" % (
                    "logprior" + _separator + p,
                    r"\pi_\mathrm{" + p.replace("_", r"\ ") + r"}"))
            for p in self.model.likelihood:
                f_paramnames.write("%s*\t%s\n" % (
                    "loglike" + _separator + p,
                    r"\log\mathcal{L}_\mathrm{" + p.replace("_", r"\ ") + r"}"))

    def save_sample(self, fname, name):
        sample = np.atleast_2d(np.loadtxt(fname))
        if not sample.size:
            return None
        collection = Collection(self.model, self.output, name=str(name))
        for row in sample:
            collection.add(
                row[2:2 + self.n_sampled],
                derived=row[2 + self.n_sampled:2 + self.n_sampled + self.n_derived],
                weight=row[0],
                logpriors=row[-(self.n_priors + self.n_likes):-self.n_likes],
                loglikes=row[-self.n_likes:])
        # make sure that the points are written
        collection.out_update()
        return collection

    def _correct_unphysical_fraction(self):
        """
        Correction for the fraction of the prior that is unphysical -- see issue #77
        """
        if not hasattr(self, "_frac_unphysical"):
            with open(self.raw_prefix + ".prior_info", "r", encoding="utf-8-sig") as pf:
                lines = list(pf.readlines())
            get_value_str = lambda line: line[line.find("=") + 1:]
            get_value_str_var = lambda var: get_value_str(
                next(l for l in lines if l.lstrip().startswith(var)))
            nprior = int(get_value_str_var("nprior"))
            ndiscarded = int(get_value_str_var("ndiscarded"))
            self._frac_unphysical = nprior / ndiscarded
        if self._frac_unphysical != 1:
            self.log.debug(
                "Correcting for unphysical region fraction: %g", self._frac_unphysical)
            self.logZ += np.log(self._frac_unphysical)
            if hasattr(self, "clusters"):
                for cluster in self.clusters.values():
                    cluster["logZ"] += np.log(self._frac_unphysical)

    def process_raw_output(self):
        """
        Loads the sample of live points from ``PolyChord``'s raw output and writes it
        (if ``txt`` output requested).
        """
        if is_main_process():
            self.log.info("Loading PolyChord's results: samples and evidences.")
            self.dump_paramnames(self.raw_prefix)
            self.collection = self.save_sample(self.raw_prefix + ".txt", "1")
            # Load clusters, and save if output
            if self.pc_settings.do_clustering:
                self.clusters = {}
                clusters_raw_regexp = re.compile(
                    re.escape(self.pc_settings.file_root + "_") + r"\d+\.txt")
                cluster_raw_files = sorted(find_with_regexp(
                    clusters_raw_regexp, os.path.join(
                        self.pc_settings.base_dir, self._clusters_dir), walk_tree=True))
                for f in cluster_raw_files:
                    i = int(f[f.rfind("_") + 1:-len(".txt")])
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
            with open(self.raw_prefix + ".stats", "r", encoding="utf-8-sig") as statsfile:
                lines = [l for l in statsfile.readlines() if l.startswith(pre)]
            for l in lines:
                logZ, logZstd = [float(n.replace(active, "")) for n in
                                 l.split("=")[-1].split("+/-")]
                component = l.split("=")[0].lstrip(pre + "_").rstrip(") ")
                if not component:
                    self.logZ, self.logZstd = logZ, logZstd
                elif self.pc_settings.do_clustering:
                    i = int(component)
                    self.clusters[i]["logZ"], self.clusters[i]["logZstd"] = logZ, logZstd
            self.log.debug(
                "RAW log(Z) = %g +/- %g ; RAW Z in [%.8g, %.8g] (68%% C.L. log-gaussian)",
                self.logZ, self.logZstd,
                *[np.exp(self.logZ + n * self.logZstd) for n in [-1, 1]])
            self._correct_unphysical_fraction()
            if self.output:
                out_evidences = dict(logZ=self.logZ, logZstd=self.logZstd)
                if getattr(self, "clusters", None):
                    out_evidences["clusters"] = {}
                    for i in sorted(list(self.clusters)):
                        out_evidences["clusters"][i] = dict(
                            logZ=self.clusters[i]["logZ"],
                            logZstd=self.clusters[i]["logZstd"])
                fname = os.path.join(self.output.folder,
                                     self.output.prefix + _evidence_extension)
                yaml_dump_file(fname, out_evidences, comment="log-evidence",
                               error_if_exists=False)
        # TODO: try to broadcast the collections
        # if get_mpi():
        #     bcast_from_0 = lambda attrname: setattr(self,
        #         attrname, get_mpi_comm().bcast(getattr(self, attrname, None), root=0))
        #     map(bcast_from_0, ["collection", "logZ", "logZstd", "clusters"])
        if is_main_process():
            self.log.info("Finished! Raw PolyChord output stored in '%s', "
                          "with prefix '%s'",
                          self.pc_settings.base_dir, self.pc_settings.file_root)
            self.log.info(
                "log(Z) = %g +/- %g ; Z in [%.8g, %.8g] (68%% C.L. log-gaussian)",
                self.logZ, self.logZstd,
                *[np.exp(self.logZ + n * self.logZstd) for n in [-1, 1]])


    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the sequentially discarded live points.
        """
        if is_main_process():
            products = {
                "sample": self.collection, "logZ": self.logZ, "logZstd": self.logZstd}
            if self.pc_settings.do_clustering:
                products.update({"clusters": self.clusters})
            return products
        else:
            return {}

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
            (re.compile(re.escape(output.prefix + ".resume")), cls.get_base_dir(output))]
        if minimal:
            return regexps_tuples
        return regexps_tuples + [
            # Raw products base dir
            (None, cls.get_base_dir(output)),
            # Main sample
            (output.collection_regexp(name=None), None),
            # Evidence
            (re.compile(re.escape(output.prefix + _evidence_extension)), None),
            # Clusters
            (None, cls.get_clusters_dir(output))
        ]

    @classmethod
    def get_version(cls):
        return None

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(
            os.path.join(path, "code",
                         cls._pc_repo_name[cls._pc_repo_name.find("/") + 1:]))

    @classmethod
    def get_import_path(cls, path):
        log = logging.getLogger(cls.__name__)
        poly_build_path = os.path.join(path, "build")
        if not os.path.isdir(poly_build_path):
            log.error("Either PolyChord is not in the given folder, "
                      "'%s', or you have not compiled it.", path)
            return None
        py_version = "%d.%d" % (sys.version_info.major, sys.version_info.minor)
        try:
            post = next(d for d in os.listdir(poly_build_path)
                        if (d.startswith("lib.") and py_version in d))
        except StopIteration:
            log.error("The PolyChord installation at '%s' has not been compiled for the "
                      "current Python version.", path)
            return None
        return os.path.join(poly_build_path, post)

    @classmethod
    def is_compatible(cls):
        import platform
        if platform.system() == "Windows":
            return False
        return True

    @classmethod
    def is_installed(cls, **kwargs):
        log = logging.getLogger(cls.__name__)
        if not kwargs.get("code", True):
            return True
        path = kwargs["path"]
        if path is not None and path.lower() == "global":
            path = None
        if path and not kwargs.get("allow_global"):
            if is_main_process():
                log.info("Importing *local* PolyChord from '%s'.", path)
            if not os.path.exists(path):
                if is_main_process():
                    log.error("The given folder does not exist: '%s'", path)
                return False
            poly_build_path = cls.get_import_path(path)
            if not poly_build_path:
                return False
        elif not path:
            if is_main_process():
                log.info("Importing *global* PolyChord.")
            poly_build_path = None
        else:
            if is_main_process():
                log.info("Importing *auto-installed* PolyChord (but defaulting to *global*).")
            poly_build_path = cls.get_import_path(path)
        try:
            # TODO: add min_version when polychord module version available
            return load_module(
                'pypolychord', path=poly_build_path, min_version=None)
        except ImportError:
            if path is not None and path.lower() != "global":
                log.error("Couldn't find the PolyChord python interface at '%s'. "
                          "Are you sure it has been installed there?", path)
            else:
                log.error("Could not import global PolyChord installation. "
                          "Specify a Cobaya or PolyChord installation path, "
                          "or install the PolyChord Python interface globally with "
                          "'cd /path/to/polychord/ ; python setup.py install'")
            return False
        except VersionCheckError as e:
            log.error(str(e))
            return False

    @classmethod
    def install(cls, path=None, force=False, code=False, data=False,
                no_progress_bars=False):
        if not code:
            return True
        log = logging.getLogger(__name__.split(".")[-1])
        log.info("Downloading PolyChord...")
        success = download_github_release(os.path.join(path, "code"), cls._pc_repo_name,
                                          cls._pc_repo_version,
                                          no_progress_bars=no_progress_bars,
                                          logger=log)
        if not success:
            log.error("Could not download PolyChord.")
            return False
        log.info("Compiling (Py)PolyChord...")
        from subprocess import Popen, PIPE
        # Needs to re-define os' PWD,
        # because MakeFile calls it and is not affected by the cwd of Popen
        cwd = os.path.join(path, "code",
                           cls._pc_repo_name[cls._pc_repo_name.find("/") + 1:])
        my_env = os.environ.copy()
        my_env.update({"PWD": cwd})
        process_make = Popen(["make", "pypolychord", "MPI=1"], cwd=cwd, env=my_env,
                             stdout=PIPE, stderr=PIPE)
        out, err = process_make.communicate()
        if process_make.returncode:
            log.info(out.decode("utf-8"))
            log.info(err.decode("utf-8"))
            log.error("Compilation failed!")
            return False
        my_env.update({"CC": "mpicc", "CXX": "mpicxx"})
        process_make = Popen([sys.executable, "setup.py", "build"],
                             cwd=cwd, env=my_env, stdout=PIPE, stderr=PIPE)
        out, err = process_make.communicate()
        if process_make.returncode:
            log.info(out.decode("utf-8"))
            log.info(err.decode("utf-8"))
            log.error("Python build failed!")
            return False
        return True
