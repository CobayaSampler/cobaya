"""
.. module:: samplers.polychord

:Synopsis: Interface for the PolyChord nested sampler
:Author: Will Handley, Mike Hobson and Anthony Lasenby (for PolyChord),
         Jesus Torrado (for the cobaya wrapper only)
"""
# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

# Global
import os
import sys
import numpy as np
import logging
import inspect
from itertools import chain
from collections import OrderedDict as odict

# Local
from cobaya.tools import read_dnumber, get_external_function
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi_comm
from cobaya.mpi import am_single_or_primary_process, more_than_one_process, sync_processes
from cobaya.collection import Collection
from cobaya.log import HandledException
from cobaya.install import download_github_release
from cobaya.yaml import yaml_dump_file

clusters = "clusters"


class polychord(Sampler):
    def initialize(self):
        """Imports the PolyChord sampler and prepares its arguments."""
        if am_single_or_primary_process():  # rank = 0 (MPI master) or None (no MPI)
            self.log.info("Initializing")
        # If path not given, try using general path to modules
        if not self.path and self.path_install:
            self.path = get_path(self.path_install)
        if self.path:
            if am_single_or_primary_process():
                self.log.info("Importing *local* PolyChord from " + self.path)
                if not os.path.exists(os.path.realpath(self.path)):
                    self.log.error("The given path does not exist.")
                    raise HandledException
            pc_build_path = get_build_path(self.path)
            if not pc_build_path:
                self.log.error("Either PolyChord is not in the given folder, "
                               "'%s', or you have not compiled it.", self.path)
                raise HandledException
            # Inserting the previously found path into the list of import folders
            sys.path.insert(0, pc_build_path)
        else:
            self.log.info("Importing *global* PolyChord.")
        try:
            import pypolychord
            from pypolychord.settings import PolyChordSettings
            self.pc = pypolychord
        except ImportError:
            self.log.error(
                "Couldn't find the PolyChord python interface. "
                "Make sure that you have compiled it, and that you either\n"
                " (a) specify a path (you didn't) or\n"
                " (b) install the Python interface globally with\n"
                "     '/path/to/PolyChord/python setup.py install --user'")
            raise HandledException
        # Prepare arguments and settings
        self.nDims = self.model.prior.d()
        self.nDerived = (len(self.model.parameterization.derived_params()) +
                         len(self.model.prior) + len(self.model.likelihood._likelihoods))
        if self.logzero is None:
            self.logzero = np.nan_to_num(-np.inf)
        if self.max_ndead == np.inf:
            self.max_ndead = -1
        for p in ["nlive", "num_repeats", "nprior", "max_ndead"]:
            setattr(self, p, read_dnumber(getattr(self, p), self.nDims, dtype=int))
        # Fill the automatic ones
        if getattr(self, "feedback", None) is None:
            values = {logging.CRITICAL: 0, logging.ERROR: 0, logging.WARNING: 0,
                      logging.INFO: 1, logging.DEBUG: 2}
            self.feedback = values[self.log.getEffectiveLevel()]
        try:
            output_folder = getattr(self.output, "folder")
            output_prefix = getattr(self.output, "prefix") or ""
            self.read_resume = self.resuming
        except AttributeError:
            # dummy output -- no resume!
            self.read_resume = False
            from tempfile import gettempdir
            output_folder = gettempdir()
            if am_single_or_primary_process():
                from random import random
                output_prefix = hex(int(random() * 16 ** 6))[2:]
            else:
                output_prefix = None
            if more_than_one_process():
                output_prefix = get_mpi_comm().bcast(output_prefix, root=0)
        self.base_dir = os.path.join(output_folder, self.base_dir)
        self.file_root = output_prefix
        if am_single_or_primary_process():
            # Creating output folder, if it does not exist (just one process)
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
            # Idem, a clusters folder if needed -- notice that PolyChord's default
            # is "True", here "None", hence the funny condition below
            if self.do_clustering is not False:  # None here means "default"
                try:
                    os.makedirs(os.path.join(self.base_dir, clusters))
                except OSError:  # exists!
                    pass
            self.log.info("Storing raw PolyChord output in '%s'.",
                          self.base_dir)
        # Exploiting the speed hierarchy
        speeds, blocks = self.model.likelihood._speeds_of_params(int_speeds=True)
        blocks_flat = list(chain(*blocks))
        self.ordering = [
            blocks_flat.index(p) for p in self.model.parameterization.sampled_params()]
        self.grade_dims = [len(block) for block in blocks]
        #        self.grade_frac = np.array(
        #            [i*j for i,j in zip(self.grade_dims, speeds)])
        #        self.grade_frac = (
        #            self.grade_frac/sum(self.grade_frac))
        # Disabled for now. We need a way to override the "time" part of the meaning of grade_frac
        self.grade_frac = [1 / len(self.grade_dims) for _ in self.grade_dims]
        # Assign settings
        pc_args = ["nlive", "num_repeats", "nprior", "do_clustering",
                   "precision_criterion", "max_ndead", "boost_posterior", "feedback",
                   "logzero", "posteriors", "equals", "compression_factor",
                   "cluster_posteriors", "write_resume", "read_resume", "write_stats",
                   "write_live", "write_dead", "base_dir", "grade_frac", "grade_dims",
                   "feedback", "read_resume", "base_dir", "file_root", "grade_frac",
                   "grade_dims"]
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
            self.log.error("PolyChord needs bounded priors, but the parameter(s) '"
                           "', '".join(params) + "' is(are) unbounded.")
            raise HandledException
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
        self.n_sampled = len(self.model.parameterization.sampled_params())
        self.n_derived = len(self.model.parameterization.derived_params())
        self.n_priors = len(self.model.prior)
        self.n_likes = len(self.model.likelihood._likelihoods)
        # Done!
        if am_single_or_primary_process():
            self.log.info("Calling PolyChord with arguments:")
            for p, v in inspect.getmembers(self.pc_settings, lambda a: not (callable(a))):
                if not p.startswith("_"):
                    self.log.info("  %s: %s", p, v)

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
                               self.n_sampled + self.n_derived + self.n_priors + self.n_likes])
        for logweight, point in zip(logweights[self.last_point_callback:],
                                    dead_points[self.last_point_callback:]):
            self.dead.add(
                point[:self.n_sampled],
                derived=point[self.n_sampled:self.n_sampled + self.n_derived],
                weight=np.exp(logweight),
                logpriors=point[self.n_sampled + self.n_derived:
                                self.n_sampled + self.n_derived + self.n_priors],
                loglikes=point[self.n_sampled + self.n_derived + self.n_priors:
                               self.n_sampled + self.n_derived + self.n_priors + self.n_likes])
        self.logZ, self.logZstd = logZ, logZstd
        # Callback function
        if self.callback_function is not None:
            self.callback_function_callable(self)
            self.last_point_callback = self.dead.n()

    def run(self):
        """
        Prepares the posterior function and calls ``PolyChord``'s ``run`` function.
        """

        # Prepare the posterior
        # Don't forget to multiply by the volume of the physical hypercube,
        # since PolyChord divides by it
        def logpost(params_values):
            logposterior, logpriors, loglikes, derived = (
                self.model.logposterior(params_values))
            if len(derived) != len(self.model.parameterization.derived_params()):
                derived = np.full(
                    len(self.model.parameterization.derived_params()), np.nan)
            if len(loglikes) != len(self.model.likelihood._likelihoods):
                loglikes = np.full(
                    len(self.model.likelihood._likelihoods), np.nan)
            derived = list(derived) + list(logpriors) + list(loglikes)
            return (
                max(logposterior + self.logvolume, 0.99 * self.pc_settings.logzero), derived)

        sync_processes()
        if am_single_or_primary_process():
            self.log.info("Sampling!")
        self.pc.run_polychord(logpost, self.nDims, self.nDerived, self.pc_settings,
                              self.pc_prior, self.dumper)

    def save_sample(self, fname, name):
        sample = np.atleast_2d(np.loadtxt(fname))
        if not sample.size:
            return None
        collection = Collection(self.model, self.output, name=str(name))
        for row in sample:
            collection.add(
                row[2:2 + self.n_sampled],
                derived=row[2 + self.n_sampled:2 + self.n_sampled + self.n_derived + 1],
                weight=row[0],
                logpriors=row[-(self.n_priors + self.n_likes):-self.n_likes],
                loglikes=row[-self.n_likes:])
        # make sure that the points are written
        collection._out_update()
        return collection

    def close(self, exception_type=None, exception_value=None, traceback=None):
        """
        Loads the sample of live points from ``PolyChord``'s raw output and writes it
        (if ``txt`` output requested).
        """
        if exception_type:
            raise
        if am_single_or_primary_process():
            self.log.info("Loading PolyChord's results: samples and evidences.")
            prefix = os.path.join(self.pc_settings.base_dir, self.pc_settings.file_root)
            self.collection = self.save_sample(prefix + ".txt", "1")
            if self.pc_settings.do_clustering is not False:  # NB: "None" == "default"
                self.clusters = {}
                do_output = hasattr(self.output, "folder")
                for f in os.listdir(os.path.join(self.pc_settings.base_dir, clusters)):
                    if not f.startswith(self.pc_settings.file_root):
                        continue
                    if do_output:
                        cluster_folder = os.path.join(
                            self.output.folder, self.output.prefix +
                                                ("_" if self.output.prefix else "") + clusters)
                        if not os.path.exists(cluster_folder):
                            os.mkdir(cluster_folder)
                    try:
                        i = int(f[len(self.pc_settings.file_root) + 1:-len(".txt")])
                    except ValueError:
                        continue
                    if do_output:
                        old_folder = self.output.folder
                        self.output.folder = cluster_folder
                    fname = os.path.join(self.pc_settings.base_dir, clusters, f)
                    sample = self.save_sample(fname, str(i))
                    self.clusters[i] = {"sample": sample}
                    if do_output:
                        self.output.folder = old_folder
            # Prepare the evidence(s) and write to file
            pre = "log(Z"
            active = "(Still active)"
            lines = []
            with open(prefix + ".stats", "r") as statsfile:
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
            if do_output:
                out_evidences = odict([["logZ", self.logZ], ["logZstd", self.logZstd]])
                if self.clusters:
                    out_evidences["clusters"] = odict()
                    for i in sorted(list(self.clusters.keys())):
                        out_evidences["clusters"][i] = odict(
                            [["logZ", self.clusters[i]["logZ"]],
                             ["logZstd", self.clusters[i]["logZstd"]]])
                fname = os.path.join(self.output.folder, self.output.prefix+".logZ")
                yaml_dump_file(fname, out_evidences, comment="log-evidence", error_if_exists=False)
        # TODO: try to broadcast the collections
        #        if get_mpi():
        #            bcast_from_0 = lambda attrname: setattr(self,
        #                attrname, get_mpi_comm().bcast(getattr(self, attrname, None), root=0))
        #            map(bcast_from_0, ["collection", "logZ", "logZstd", "clusters"])
        if am_single_or_primary_process():
            self.log.info("Finished! Raw PolyChord output stored in '%s', "
                          "with prefix '%s'",
                          self.pc_settings.base_dir, self.pc_settings.file_root)

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the sequentially discarded live points.
        """
        if am_single_or_primary_process():
            products = {
                "sample": self.collection, "logZ": self.logZ, "logZstd": self.logZstd}
            if self.pc_settings.do_clustering:
                products.update({"clusters": self.clusters})
            return products
        else:
            return {}


# Installation routines ##################################################################

# Name of the PolyChord repo and version to download
pc_repo_name = "PolyChord/PolyChordLite"
pc_repo_version = "ef02bb6d94dca218c7d8daa98e8ac010022e457e"


def get_path(path):
    return os.path.realpath(
        os.path.join(path, "code", pc_repo_name[pc_repo_name.find("/")+1:]))


def get_build_path(polychord_path):
    try:
        build_path = os.path.join(polychord_path, "build")
        post = next(d for d in os.listdir(build_path) if d.startswith("lib."))
        build_path = os.path.join(build_path, post)
        return build_path
    except (OSError, StopIteration):
        return False


def is_installed(**kwargs):
    if not kwargs["code"]:
        return True
    poly_path = get_path(kwargs["path"])
    if not os.path.isfile(os.path.realpath(os.path.join(poly_path, "lib/libchord.so"))):
        return False
    poly_build_path = get_build_path(poly_path)
    if not poly_build_path:
        return False
    sys.path.insert(0, poly_build_path)
    try:
        import pypolychord
        return True
    except ImportError:
        return False


def install(path=None, force=False, code=False, data=False, no_progress_bars=False):
    if not code:
        return True
    log = logging.getLogger(__name__.split(".")[-1])
    log.info("Downloading PolyChord...")
    success = download_github_release(os.path.join(path, "code"), pc_repo_name,
                                      pc_repo_version, no_progress_bars=no_progress_bars)
    if not success:
        log.error("Could not download PolyChord.")
        return False
    log.info("Compiling (Py)PolyChord...")
    from subprocess import Popen, PIPE
    # Needs to re-define os' PWD,
    # because MakeFile calls it and is not affected by the cwd of Popen
    cwd = os.path.join(path, "code", pc_repo_name[pc_repo_name.find("/")+1:])
    my_env = os.environ.copy()
    my_env.update({"PWD": cwd})
    process_make = Popen(["make", "pypolychord", "MPI=1"], cwd=cwd, env=my_env,
                         stdout=PIPE, stderr=PIPE)
    out, err = process_make.communicate()
    if process_make.returncode:
        log.info(out)
        log.info(err)
        log.error("Compilation failed!")
        return False
    my_env.update({"CC": "mpicc", "CXX": "mpicxx"})
    process_make = Popen([sys.executable, "setup.py", "build"],
                         cwd=cwd, env=my_env, stdout=PIPE, stderr=PIPE)
    out, err = process_make.communicate()
    if process_make.returncode:
        log.info(out)
        log.info(err)
        log.error("Python build failed!")
        return False
    return True
