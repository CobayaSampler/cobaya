"""
.. module:: samplers.polychord

:Synopsis: PolyChord nested sampler (REFERENCES)
:Author: Will Handley, Mike Hobson and Anthony Lasenby (for PolyChord), Jesus Torrado (for the wrapper only)

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import os
import sys
import numpy as np
import logging
import inspect

# Local
from cobaya.conventions import _path_install
from cobaya.tools import get_path_to_installation
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi, get_mpi_comm, get_mpi_rank
from cobaya.collection import Collection
from cobaya.log import HandledException
from cobaya.install import download_github_release

clusters = "clusters"


class polychord(Sampler):
    def initialise(self):
        """Imports the PolyChord sampler and prepares its arguments."""
        if not get_mpi_rank():  # rank = 0 (MPI master) or None (no MPI)
            self.log.info("Initializing")
        if not self.path:
            path_to_installation = get_path_to_installation()
            if path_to_installation:
                self.path = os.path.join(
                    path_to_installation, "code", pc_repo_name)
        if self.path is None:
            self.log.error("No path given for PolyChord. Set the "
                           "likelihood property 'path' or the common property "
                           "'%s'.", _path_install)
            raise HandledException
        if not get_mpi_rank():
            self.log.info("Importing PolyChord from %s", self.path)
            if not os.path.exists(self.path):
                self.log.error("The path you indicated for PolyChord "
                               "does not exist: %s", self.path)
                raise HandledException
        sys.path.insert(0, self.path)
        import PyPolyChord_ctypes as PyPolyChord
        from PyPolyChord_ctypes.settings import PolyChordSettings
        self.pc = PyPolyChord
        # Prepare arguments and settings
        self.nDims = self.prior.d()
        self.nDerived = (len(self.parametrization.derived_params()) + 1 +
                         len(self.likelihood._likelihoods))
        self.pc_settings = PolyChordSettings(self.nDims, self.nDerived)
        for p in ["nlive", "num_repeats", "nprior", "do_clustering",
                  "precision_criterion", "max_ndead", "boost_posterior", "feedback",
                  "update_files", "posteriors", "equals", "cluster_posteriors",
                  "write_resume", "read_resume", "write_stats", "write_live",
                  "write_dead", "base_dir", "grade_frac", "grade_dims"]:
            v = getattr(self,p)
            if v is not None:
                setattr(self.pc_settings, p, v)
        # Fill the automatic ones
        if getattr(self, "feedback", None) is None:
            values = {logging.CRITICAL: 0, logging.ERROR: 0, logging.WARNING: 0,
                      logging.INFO: 1, logging.DEBUG: 2}
            self.pc_settings.feedback = values[self.log.getEffectiveLevel()]
        try:
            output_folder = getattr(self.output, "folder")
            output_prefix = getattr(self.output, "prefix") or "pc"
        except AttributeError:
            # dummy output -- no resume!
            from tempfile import gettempdir
            output_folder = gettempdir()
            from random import random
            output_prefix = hex(int(random()*16**6))[2:]
            self.pc_settings.read_resume = False
        self.pc_settings.base_dir = os.path.join(output_folder, self.pc_settings.base_dir)
        self.pc_settings.file_root = output_prefix
        if not get_mpi_rank():
            # Creating output folder, if it does not exist (just one process)
            if not os.path.exists(self.pc_settings.base_dir):
                os.makedirs(self.pc_settings.base_dir)
            # Idem, a clusters folder if needed -- notice that PolyChord's default
            # is "True", here "None", hence the funny condition below
            if self.pc_settings.do_clustering is not False:  # None here means "default"
                try:
                    os.makedirs(os.path.join(self.pc_settings.base_dir, clusters))
                except OSError:  # exists!
                    pass
            self.log.info("Storing raw PolyChord output in '%s'.",
                          self.pc_settings.base_dir)
        # explotining the speed hierarchy
        # sort blocks by paramters order and check contiguity (required by PolyChord!!!)
#        speeds, blocks = zip(*self.likelihood.speed_blocked_params(as_indices=True))
#        speeds, blocks = np.array(speeds), np.array(blocks)
        # weird behaviour of np.argsort with there is only 1 block
#        if len(blocks) > 1:
#            sorting_indices = np.argsort(blocks, axis=0)
#        else:
#            sorting_indices = [0]
#        speeds, blocks = speeds[sorting_indices], blocks[sorting_indices]
#        if np.all([np.all(block==range(block[0], block[-1]+1)) for block in blocks]):
        self.log.warning("Speed hierarchy exploitation disabled for now!")
#            self.pc_args["grade_frac"] = list(speeds)
#            self.pc_args["grade_dims"] = [len(block) for block in blocks]
#            self.log.info("Exploiting a speed hierarchy with speeds %r and blocks %r",
#                     speeds, blocks)
#        else:
#            self.log.warning("Some speed blocks are not contiguous: PolyChord cannot deal "
#                        "with the speed hierarchy. Not exploting it.")
        # prior conversion from the hypercube
        bounds = self.prior.bounds(confidence_for_unbounded=self.confidence_for_unbounded)
        # Check if priors are bounded (nan's to inf)
        inf = np.where(np.isinf(bounds))
        if len(inf[0]):
            params_names = self.prior.names()
            params = [params_names[i] for i in sorted(list(set(inf[0])))]
            self.log.error("PolyChord needs bounded priors, but the parameter(s) '"
                           "', '".join(params)+"' is(are) unbounded.")
            raise HandledException
        locs = bounds[:,0]
        scales = bounds[:,1] - bounds[:,0]
        self.pc_prior = lambda x: (locs + np.array(x)*scales).tolist()
        # We will need the volume of the prior domain, since PolyChord divides by it
        self.logvolume = np.log(np.prod(scales))
        # Done!
        if not get_mpi_rank():
            self.log.info("Calling PolyChord with arguments:")
            for p,v in inspect.getmembers(self.pc_settings, lambda a: not(callable(a))):
                if not p.startswith("_"):
                    self.log.info("  %s: %s", p, v)

    def run(self):
        """
        Prepares the posterior function and calls ``PolyChord``'s ``run`` function.
        """
        # Prepare the posterior
        # Don't forget to multiply by the volume of the physical hypercube,
        # since PolyChord divides by it
        def logpost(params_values):
            logposterior, logprior, logliks, derived = self.logposterior(params_values)
            if len(derived) != len(self.parametrization.derived_params()):
                derived = np.full(
                    len(self.parametrization.derived_params()), np.nan)
            if len(logliks) != len(self.likelihood._likelihoods):
                logliks = np.full(
                    len(self.likelihood._likelihoods), np.nan)
            derived = list(derived) + [logprior] + list(logliks)
            return logposterior+self.logvolume, derived
        self.log.info("Sampling!")
        self.pc.run_polychord(logpost, self.nDims, self.nDerived,
                              self.pc_settings, self.pc_prior)

    def save_sample(self, fname, name):
        sample = np.loadtxt(fname)
        collection = Collection(
            self.parametrization, self.likelihood, self.output, name=str(name))
        for row in sample:
            collection.add(
                row[2:2+self.n_sampled],
                derived=row[2+self.n_sampled:2+self.n_sampled+self.n_derived+1],
                weight=row[0], logpost=-row[1], logprior=row[-(1+self.n_liks)],
                logliks=row[-self.n_liks:])
        # make sure that the points are written
        collection.out_update()
        return collection

    def close(self):
        """
        Loads the sample of live points from ``PolyChord``'s raw output and writes it
        (if ``txt`` output requested).
        """
        if not get_mpi_rank():  # process 0 or single (non-MPI process)
            self.log.info("Loading PolyChord's results: samples and evidences.")
            self.n_sampled = len(self.parametrization.sampled_params())
            self.n_derived = len(self.parametrization.derived_params())
            self.n_liks = len(self.likelihood._likelihoods)
            prefix = os.path.join(self.pc_settings.base_dir, self.pc_settings.file_root)
            self.collection = self.save_sample(prefix+".txt", "1")
            if self.pc_settings.do_clustering is not False:  # None here means "default"
                self.clusters = {}
                cluster_folder = os.path.join(
                    self.output.folder, self.output.prefix + "_" + clusters)
                for f in os.listdir(os.path.join(self.pc_settings.base_dir, clusters)):
                    if not os.path.exists(cluster_folder):
                        os.mkdir(cluster_folder)
                    try:
                        i = int(f[len(self.pc_settings.file_root)+1:-len(".txt")])
                    except ValueError:
                        continue
                    old_folder = self.output.folder
                    self.output.folder = cluster_folder
                    fname = os.path.join(self.pc_settings.base_dir, clusters, f)
                    self.clusters[i] = {"sample": self.save_sample(fname, str(i))}
                    self.output.folder = old_folder
            # Prepare the evidence
            pre = "log(Z"
            lines = []
            with open(prefix+".stats", "r") as statsfile:
                lines = [l for l in statsfile.readlines() if l.startswith(pre)]
            for l in lines:
                logZ, logZstd = [float(n) for n in l.split("=")[-1].split("+/-")]
                component = l.split("=")[0].lstrip(pre+"_").rstrip(") ")
                if not component:
                    self.logZ, self.logZstd = logZ, logZstd
                elif self.pc_settings.do_clustering:
                    i = int(component)
                    self.clusters[i]["logZ"], self.clusters[i]["logZstd"] = logZ, logZstd
#        if get_mpi():
#            bcast_from_0 = lambda attrname: setattr(self,
#                attrname, get_mpi_comm().bcast(getattr(self, attrname, None), root=0))
#            map(bcast_from_0, ["collection", "logZ", "logZstd", "clusters"])
        if not get_mpi_rank():  # process 0 or single (non-MPI process)
            self.log.info("Finished! Raw PolyChord output stored in '%s'.",
                          self.pc_settings.base_dir)

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the sequentially discarded live points.
        """
        if not get_mpi_rank():
            products = {
                "sample": self.collection, "logZ": self.logZ, "logZstd": self.logZstd}
            if self.pc_settings.do_clustering:
                products.update({"clusters": self.clusters})
            return products
        else:
            return {}


# Installation routines ##################################################################

# Name of the PolyChord repo and version to download
pc_repo_name = "PolyChord"
pc_repo_version = "v1.12.ctypes1.0"


def get_path(path):
    return os.path.realpath(os.path.join(path, "code", pc_repo_name))


def is_installed(**kwargs):
    if not kwargs["code"]:
        return True
    return os.path.isfile(os.path.realpath(
        os.path.join(kwargs["path"], "code", pc_repo_name, "lib/libchord.so")))


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
    cwd = os.path.join(path, "code", pc_repo_name)
    my_env = os.environ.copy()
    my_env["PWD"] = cwd
    process_make = Popen(["make", "PyPolyChord", "MPI=1"], cwd=cwd, env=my_env,
                         stdout=PIPE, stderr=PIPE)
    out, err = process_make.communicate()
    if process_make.returncode:
        log.info(out)
        log.info(err)
        log.error("Compilation failed!")
        return False
    return True
