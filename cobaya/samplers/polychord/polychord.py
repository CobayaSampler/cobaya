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

The main output is the Monte Carlo sample of sequentially discarded *live points*, saved
in the standard sample format together with the ``input.yaml`` and ``full.yaml``
files (see :doc:`output`). The raw ``PolyChord`` products are saved in a 
subfolder of the output folder
(determined by the option ``base_dir`` -- default: ``polychord_output``).


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

# Local
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi_rank
from cobaya.collection import Collection
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)


# attributes that need to be broadcasted
to_broadcast = ["pc_args"]

class polychord(Sampler):
    def initialise(self):
        """Imports the PolyChord sampler and prepares its arguments."""
        if not get_mpi_rank(): # rank = 0 (MPI master) or None (no MPI)
            log.info("Initialising")
        # Importing PolyChord from the correct path
        if self.path:
            if not get_mpi_rank():
                log.info("Importing PolyChord from %s", self.path)
                if not os.path.exists(self.path):
                    log.error(
                    "The path you indicated for PolyChord does not exist: "+self.path)
                    raise HandledException
            sys.path.insert(0, self.path)
            from PyPolyChord import PyPolyChord
        else:
            # Currently, not installable as a python package! This will ALWAYS fail
            log.error("You need to specify PolyChord's path.")
            raise HandledException
            # log.info("Importing *global* PolyChord")
            # try:
            #     import PyPolyChord
            # except ImportError:
            #     log.error(
            #         "Couldn't find PolyChord's python interface.\n"
            #         "Make sure that you have compiled it, and that you either\n"
            #         " (a) specify a path (you didn't) or\n"
            #         " (b) install PolyChord's python interface globally with ... ???\n")
            #     raise HandledException
        self.pc = PyPolyChord
        # Prepare arguments - get just the PolyChord arguments
        self.pc_args = dict([(p, getattr(self,p)) for p in
                        ["nlive", "num_repeats", "do_clustering", "precision_criterion",
                         "max_ndead", "boost_posterior", "feedback", "update_files",
                         "posteriors", "equals", "cluster_posteriors", "write_resume",
                         "read_resume", "write_stats", "write_live", "write_dead",
                         "base_dir", "grade_frac", "grade_dims"]])
        # Ignore null-defined ones, so PolyChord sets them to the defaults
        self.pc_args = dict([(k,v) for k,v in self.pc_args.iteritems() if not v is None])
        # Fill the automatic ones
        if not "feedback" in self.pc_args:
            values = {logging.CRITICAL: 0, logging.ERROR: 0, logging.WARNING: 0,
                      logging.INFO: 1, logging.DEBUG: 2}
            self.pc_args["feedback"] = values[log.getEffectiveLevel()]
        self.pc_args["nDims"] = self.prior.d()
        self.pc_args["nDerived"] = (
            1 + len(self.likelihood.derived) + len(self.likelihood.names()))
        try:
            output_folder = getattr(self.output, "folder")
            output_prefix = getattr(self.output, "prefix")
        except AttributeError:
            # dummy output -- no resume!
            from tempfile import gettempdir
            output_folder = gettempdir()
            from random import random
            output_prefix = hex(int(random()*16**6))[2:]
            self.pc_args["read_resume"] = False
        self.pc_args["base_dir"] = os.path.join(output_folder, self.pc_args["base_dir"])
        self.pc_args["file_root"] = output_prefix
        if not get_mpi_rank():
            # Creating output folder, if it does not exist (just one process)
            if not os.path.exists(self.pc_args["base_dir"]):
                os.makedirs(self.pc_args["base_dir"])
            # Idem, a clusters folder if needed -- notice that PolyChord's default
            # is "True", here "None", hence the funny condition below
            if self.pc_args.get("do_clustering") is not False:
                try:
                    os.makedirs(os.path.join(self.pc_args["base_dir"], "clusters"))
                except OSError: # exists!
                    pass
            log.info("Storing raw PolyChord output in '%s'.", self.pc_args["base_dir"])
        # explotining the speed hierarchy
        # sort blocks by paramters order and check contiguity (required by PolyChord!!!)
        speeds, blocks = zip(*self.likelihood.speed_blocked_params(as_indices=True))
        speeds, blocks = np.array(speeds), np.array(blocks)
        # weird behaviour of np.argsort with there is only 1 block
        if len(blocks) > 1:
            sorting_indices = np.argsort(blocks, axis=0)
        else:
            sorting_indices = [0]
        speeds, blocks = speeds[sorting_indices], blocks[sorting_indices]
        if np.all([np.all(block==range(block[0], block[-1]+1)) for block in blocks]):
            log.warning("TODO: SPEED HIERARCHY EXPLOITATION DISABLED FOR NOW!!!")
#            self.pc_args["grade_frac"] = list(speeds)
#            self.pc_args["grade_dims"] = [len(block) for block in blocks]
#            log.info("Exploiting a speed hierarchy with speeds %r and blocks %r",
#                     speeds, blocks)
        else:
            log.warning("Some speed blocks are not contiguous: PolyChord cannot deal "
                        "with the speed hierarchy. Not exploting it.")
        # prior conversion from the hypercube
        limits = self.prior.limits()
        # Check if priors are bounded (nan's to inf)
        inf = np.where(np.isfinite(limits)==False)
        if len(inf[0]):
            params_names = self.prior.names()
            params = [params_names[i] for i in sorted(list(set(inf[0])))]
            log.error("PolyChord needs bounded priors, but the parameter(s) '"+
                      "', '".join(params)+"' is(are) unbounded.")
            raise HandledException
        locs = limits[:,0]
        scales = limits[:,1] - limits[:,0]
        self.pc_args["prior"] = lambda x: (locs + np.array(x)*scales).tolist()
        # We will need the volume of the prior domain, since PolyChord divides by it
        self.volume = np.prod(scales)
        # Done!
        if not get_mpi_rank():
            log.info("Calling PolyChord with arguments"+
                     "\n  ".join([""]+["%s : "%p+str(v) for p,v in self.pc_args.iteritems()]))

    def run(self):
        """
        Prepares the posterior function and calls ``PolyChord``'s ``run`` function.
        """
        # Prepare the posterior
        # Don't forget to multiply by the volume of the physical hypercube,
        # since PolyChord divides by it
        def logpost(p):
            logprior = self.prior.logp(p)
            logpost = logprior + np.log(self.volume)
            if logprior > -np.inf:
                derived = []
                logliks = self.likelihood.logps(p, derived=derived)
                logpost += sum(logliks)
            else:
                logliks = np.array([np.nan]*len(self.likelihoods))
            # derived parameters -- add physical ones!!!
            derived += [-logprior] + (-2*logliks).tolist()
            log.debug("logpost=%r, derived=%r", logpost, derived)
            return logpost, derived
        log.info("Sampling!")
        self.pc.run_nested_sampling(logpost, **self.pc_args)
        
    def close(self):
        """
        Loads the sample of live points from ``PolyChord``'s raw output and writes it
        (if ``txt`` output requested).
        """
        if not get_mpi_rank(): # process 0 or single (non-MPI process)
            log.info("Loading PolyChord's resuts: samples and evidences.")
            prefix = os.path.join(self.pc_args["base_dir"], self.pc_args["file_root"])
            sample = np.loadtxt(prefix+".txt")
            self.collection = Collection(self.prior, self.likelihood, self.output, name="1")
            n_sampled = len(self.prior.names())
            n_derived = len(self.likelihood.derived)
            n_liks = len(self.likelihood.names())
            for row in sample:
                self.collection.add(row[2:2+n_sampled],
                                    derived=row[2+n_sampled:2+n_sampled+n_derived+1],
                                    weight=row[0], logpost=-row[1],
                                    logprior=-row[-(1+n_liks)],
                                    logliks=[-0.5*lik for lik in row[-n_liks:]])
            # make sure that the points are written
            self.collection.out_update()
            # Prepare the evidence
            with open(prefix+".stats", "r") as statsfile:
                line = ""
                while not "Global evidence:" in line:
                    line = statsfile.readline()
                while not "log(Z)" in line:
                    line = statsfile.readline()
                self.logZ, self.logZstd = [
                    float(n) for n in line.split("=")[-1].split("+/-")]
# THE RESULTS CANNOT BE BROADCASTED BECAUSE POLYCHORD KILLS MPI!
#        # Broadcast results
#        if get_mpi():
#            bcast_from_0 = lambda attrname: setattr(self, attrname,
#                get_mpi_comm().bcast(getattr(self, attrname, None), root=0))
#            map(bcast_from_0, ["collection", "logZ", "logZstd"])
        if not get_mpi_rank(): # process 0 or single (non-MPI process)
            log.info("Finished! "+
                     "Raw PolyChord output stored in '%s'.", self.pc_args["base_dir"])

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the sequentially discarded live points.
        """
        if not get_mpi_rank():
            return {"sample": self.collection, "logZ": self.logZ, "logZstd": self.logZstd}
