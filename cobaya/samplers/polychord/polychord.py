"""
.. module:: samplers.polychord

:Synopsis: PolyChord nested sampler (REFERENCES)
:Author: Will Handley, Mike Hobson and Anthony Lasenby (for PolyChord), Jesus Torrado (for the wrapper only)

.. |br| raw:: html

   <br />

.. note::
   **If you use this sampler, please cite it as:**
   |br|
   `W.J. Handley, M.P. Hobson, A.N. Lasenby, "PolyChord: nested sampling for cosmology"
   (arXiv:1502.01856) <https://arxiv.org/abs/1502.01856>`_
   |br|
   `W.J. Handley, M.P. Hobson, A.N. Lasenby, "PolyChord: next-generation nested sampling"
   (arXiv:1506.00171) <https://arxiv.org/abs/1506.00171>`_

``PolyChord`` is an advanced
`nested sampler <http://projecteuclid.org/euclid.ba/1340370944>`_,
that uses slice sampling to sample within the
nested isolikelihoods contours. The use of slice sampling instead of rejection sampling
makes ``PolyChord`` specially suited for high-dimensional parameter spaces, and allows for
exploiting speed hierarchies in the parameter space. Also, ``PolyChord`` can explore
multi-modal distributions efficiently.

``PolyChord`` is an *external* sampler, not installed by default (just a wrapper for it).
You need to install it yourself following the instructions below.

Usage
-----

To use ``PolyChord``, you just need to include the sampler block (assuming the directory
structure described in the :ref:`installation instructions <pc_installation>`.

.. code-block:: yaml

   sampler:
     polychord:
       path: /path/to/cosmo/PolyChord
       [optional PolyChord input parameters]


The parameters of interest to the user, if not defined in the ``polychord:`` block,
are left to their defaults for PolyChord 1.9 (see ``README`` file in ``PolyChord``'s
installation folder). If you wish to modify them, copy them from the ``defaults.yaml``
file (reproduced below).

The wrapper for ``PolyChord`` deals automatically with some of ``PolyChord``'s original
input parameters (e.g. the number of dimensions and of derived parameters, and the
parameters related to the output). The speed hierarchy, specified as described in
:ref:`mcmc_speed_hierarchy`, is exploited by default.

.. warning::

   If you want to sample with ``PolyChord``, your priors need to be bounded. This is
   because ``PolyChord`` samples uniformly from a bounded *hypercube*, defined by a
   non-trivial transformation for general unbounded priors.

   The option ``confidence_for_unbounded`` will automatically bind the priors at 5-sigma
   c.l., but this may cause problems with likelihood modes at the edge of the prior.
   In those cases, check for stability with respect to increasing that parameter.
   Of course, if ``confidence_for_unbounded`` is too small, the resulting evidence may be
   biased towards a lower value.

The main output is the Monte Carlo sample of sequentially discarded *live points*, saved
in the standard sample format together with the ``input.yaml`` and ``full.yaml``
files (see :doc:`output`). The raw ``PolyChord`` products are saved in a
subfolder of the output folder
(determined by the option ``base_dir`` -- default: ``polychord_output``).

.. note::

   Getting a `Segmentation fault`? Try to evaluate your likelihood before calling
   `PolyChord`: change to the `evaluate` sampler to test it.

.. _pc_installation:

Installation
------------

You will need a MPI-wrapped Fortran compiler
(it is rarely worth running PolyChord without MPI capabilities in realistic scenarios).
You should have an MPI implementation installed if you followed
:ref:`the instructions to install mpi4py <install_mpi>`.
On top of that, you need the Fortran compiler (we recommend the GNU one) and the
*development* package of MPI. Use your system's package manager to install them
(``gfortran`` and ``libopenmpi-dev`` in Debian-based systems), or contact your local
IT service. If everything is correctly installed, you should be able to type ``mpif90``
in the shell and not get a ``Command not found`` error.

Download PolyChord version 1.9 from
`CCPForge <https://ccpforge.cse.rl.ac.uk/gf/project/polychord/>`_
(login required, free registration).
Uncompress the file you just downloaded in the folder where you are installing the
samplers, and follow the instructions in the ``README`` file.

.. note::
   The fast way, assuming that you are in a GNU/Linux system, that you are following the
   directory structure described in :ref:`directory_structure`, and the PolyChord file
   downloaded is called ``PolyChord_v1.9.tar.gz``:

   .. code:: bash

      $ cd /path/to/cosmo
      $ tar xzvf PolyChord_v1.9.tar.gz
      $ rm PolyChord_v1.9.tar.gz
      $ cd PolyChord
      $ make PyPolyChord
      $ echo -e "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$PWD/lib" >> ~/.bashrc
      $ source ~/.bashrc

If PolyChord has been compiled with MPI, you **must** call cobaya with ``mpirun`` if you
want to use PolyChord as a sampler.

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
from cobaya.mpi import get_mpi_rank
from cobaya.collection import Collection
from cobaya.log import HandledException
from cobaya.install import download_github_release


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
            if v:
                setattr(self.pc_settings, p, v)
        # Fill the automatic ones
        if getattr(self, "feedback", None) is None:
            values = {logging.CRITICAL: 0, logging.ERROR: 0, logging.WARNING: 0,
                      logging.INFO: 1, logging.DEBUG: 2}
            self.pc_settings.feedback = values[self.log.getEffectiveLevel()]
        try:
            output_folder = getattr(self.output, "folder")
            output_prefix = getattr(self.output, "prefix")
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
            if self.pc_settings.do_clustering is not False:
                try:
                    os.makedirs(os.path.join(self.pc_settings.base_dir, "clusters"))
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
        self.log.warning("TODO: SPEED HIERARCHY EXPLOITATION DISABLED FOR NOW!!!")
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

    def close(self):
        """
        Loads the sample of live points from ``PolyChord``'s raw output and writes it
        (if ``txt`` output requested).
        """
        if not get_mpi_rank():  # process 0 or single (non-MPI process)
            self.log.info("Loading PolyChord's results: samples and evidences.")
            prefix = os.path.join(self.pc_settings.base_dir, self.pc_settings.file_root)
            sample = np.loadtxt(prefix+".txt")
            self.collection = Collection(
                self.parametrization, self.likelihood, self.output, name="1")
            n_sampled = len(self.parametrization.sampled_params())
            n_derived = len(self.parametrization.derived_params())
            n_liks = len(self.likelihood._likelihoods)
            for row in sample:
                self.collection.add(row[2:2+n_sampled],
                                    derived=row[2+n_sampled:2+n_sampled+n_derived+1],
                                    weight=row[0], logpost=-row[1],
                                    logprior=row[-(1+n_liks)],
                                    logliks=row[-n_liks:])
            # make sure that the points are written
            self.collection.out_update()
            # Prepare the evidence
            with open(prefix+".stats", "r") as statsfile:
                line = ""
                while "Global evidence:" not in line:
                    line = statsfile.readline()
                while "log(Z)" not in line:
                    line = statsfile.readline()
                self.logZ, self.logZstd = [
                    float(n) for n in line.split("=")[-1].split("+/-")]
# TODO: that's not true anymore!!!
# THE RESULTS CANNOT BE BROADCASTED BECAUSE POLYCHORD KILLS MPI!
#        # Broadcast results
#        if get_mpi():
#            bcast_from_0 = lambda attrname: setattr(self, attrname,
#                get_mpi_comm().bcast(getattr(self, attrname, None), root=0))
#            map(bcast_from_0, ["collection", "logZ", "logZstd"])
        if not get_mpi_rank():  # process 0 or single (non-MPI process)
            self.log.info("Finished! "
                          "Raw PolyChord output stored in '%s'.", self.pc_settings.base_dir)

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the sequentially discarded live points.
        """
        if not get_mpi_rank():
            return {"sample": self.collection, "logZ": self.logZ, "logZstd": self.logZstd}


# Installation routines ##################################################################

# Name of the PolyChord repo and version to download
pc_repo_name = "PolyChord"
pc_repo_version = "v0.94"


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
