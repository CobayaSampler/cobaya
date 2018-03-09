"""
.. module:: theories.classy

:Synopsis: Managing the CLASS cosmological code
:Author: Jesus Torrado, Benjamin Audren (CLASS import and ``get_cl``)

.. |br| raw:: html

   <br />

This module imports and manages the CLASS cosmological code.

.. note::

   **If you use this cosmological code, please cite it as:**
   |br|
   `D. Blas, J. Lesgourgues, T. Tram, "The Cosmic Linear Anisotropy Solving System (CLASS). Part II: Approximation schemes"
   (arXiv:1104.2933) <https://arxiv.org/abs/1104.2933>`_

.. note::

   CLASS is renamed ``classy`` for most purposes within cobaya, due to CLASS' name being
   a python keyword.

Usage
-----

If you are using a likelihood that requires some observable from CLASS,
simply add it to the theory block, specifying its ``path`` if you have not installed CLASS
globally. You can specify any parameter that CAMB or CLASS understand within the ``theory``
sub-block of the ``params`` block:

.. code-block:: yaml

   theory:
     classy:

   params:
     theory:
       [any param that CLASS understands, fixed, sampled or derived]

.. note::

   Some parameter names can be specified in CAMB nomenclature, such that the ``classy`` and
   ``camb`` theory blocks can be swapped easily. If there is a conflict, i.e. CLASS and
   CAMB give different parameters the same name, CAMB meaning takes priority.

   .. todo:: Not ideal. Working on it.


Installation
------------

.. note::

   If you already have your own version of CLASS, just make sure that the Python interface
   has been compiled, take note of its installation path and specify it using the
   ``path`` option in the ``classy`` block of the input.

CLASS' python interface utilises the ``cython`` compiler. If typing ``cython`` in the
shell produces an error, install it with ``pip install cython --user``.

.. note::
   The fast way, assuming you are installing all your cosmological codes under
   ``/path/to/cosmo/``:

   .. code:: bash

      $ cd /path/to/cosmo/
      $ git clone https://github.com/lesgourg/class_public.git
      $ mv class_public CLASS
      $ cd CLASS
      $ make

   If the **second** line produces an error (because you don't have ``git`` installed),
   try

   .. code:: bash

      $ cd /path/to/cosmo/
      $ wget https://github.com/lesgourg/class_public/archive/master.zip
      $ unzip master.zip
      $ rm master.zip
      $ mv class_public-master CLASS
      $ cd CLASS
      $ make

If the instructions above failed, follow those in the
`official CLASS web page <http://class-code.net/>`_ to get CLASS compiled with the Python
interface ready.

If you modify CLASS and add new variables, you don't need to let cobaya now, but make
sure that the variables you create are exposed in the Python
interface (contact CLASS' developers if you need help with that).

.. warning::

   At this moment, CLASS is not compatible with ``gcc`` version 5.0 and above
   (type ``gcc --version`` to check yours). This would cause an error when trying to use
   CLASS in cobaya, containing a line similar to

   .. code::

      /home/<username>/.local/lib/python2.7/site-packages/classy.so: undefined symbol: _ZGVbN2v_sin

   To solve it, open the file ``Makefile`` in the CLASS folder and add change the line
   stating ``CCFLAG = -g -fPIC`` to

   .. code:: make

      CCFLAG = -g -fPIC -fno-tree-vectorize

   Finally, clean and recompile, using ``make clean ; make``.

   **Source:** `CLASS issue #99 in github <https://github.com/lesgourg/class_public/issues/99>`_

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import sys
import os
import numpy as np
from copy import deepcopy
import logging
from collections import namedtuple, OrderedDict as odict

# Local
from cobaya.theory import Theory
from cobaya.log import HandledException
from cobaya.tools import get_path_to_installation
from cobaya.install import user_flag_if_needed


# Result collector
collector = namedtuple("collector", ["method", "kwargs"])


class classy(Theory):

    def initialise(self):
        """Importing CLASS from the correct path, if given, and if not, globally."""
        # If path not given, try using general path to modules
        path_to_installation = get_path_to_installation()
        if not self.path and path_to_installation:
            self.path = os.path.join(
                path_to_installation, "code", "CLASS")
        if self.path:
            self.log.info("Importing *local* CLASS from "+self.path)
            if not os.path.exists(self.path):
                self.log.error("The given folder does not exist: '%s'", self.path)
                raise HandledException
            try:
                classy_path = ''
                classy_build_path = os.path.join(self.path, "python", "build")
                for elem in os.listdir(classy_build_path):
                    if elem.find("lib.") != -1:
                        classy_path = os.path.join(classy_build_path, elem)
                        break
                # Inserting the previously found path into the list of import folders
                sys.path.insert(0, classy_path)
                from classy import Class, CosmoSevereError, CosmoComputationError
            except OSError:
                self.log.error("Either CLASS is not in the given folder,\n"
                               "'%s',\n or you have not compiled it.", self.path)
                raise HandledException
        else:
            self.log.info("Importing *global* CLASS.")
            try:
                from classy import Class, CosmoSevereError, CosmoComputationError
            except ImportError:
                self.log.error(
                    "Couldn't find the CLASS python interface.\n"
                    "Make sure that you have compiled it, and that you either\n"
                    " (a) specify a path (you didn't) or\n"
                    " (b) install the Python interface globally with\n"
                    "     '/path/to/class/python/python setup.py install --user'")
                raise HandledException
        self.classy = Class()
        # Propagate errors up
        global CosmoComputationError, CosmoSevereError
        # Generate states, to avoid recomputing
        self.n_states = 3
        self.states = [{"params": None, "derived": None, "derived_extra": None,
                        "last": 0} for i in range(self.n_states)]
        # Dict of named tuples to collect requirements and computation methods
        self.collectors = {}
        # Additional input parameters to pass to CAMB
        self.extra_args = self.extra_args or {}
        self.extra_args["output"] = self.extra_args.get("output", "")
        if "sBBN file" in self.extra_args:
            self.extra_args["sBBN file"] = (
                os.path.join(self.path, self.extra_args["sBBN file"]))
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []

    def current_state(self):
        lasts = [self.states[i]["last"] for i in range(self.n_states)]
        return self.states[lasts.index(max(lasts))]

    def needs(self, arguments):
        # Computed quantities requiered by the likelihood
        for k,v in arguments.items():
            # Precision parameters and boundaries (in general, take max of all requested)
            if k == "l_max":
                self.extra_args["l_max_scalars"] = (
                    max(v, self.extra_args.get("l_max_scalars", 0)))
            elif k == "k_max":
                self.extra_args["P_k_max_h/Mpc"] = (
                    max(v, self.extra_args.get("P_k_max_h/Mpc", 0)))
            # Products and other computations
            elif k == "Cl":
                if any([("t" in cl.lower()) for cl in v]):
                    self.extra_args["output"] += " tCl"
                if any([(("e" in cl.lower()) or ("b" in cl.lower())) for cl in v]):
                    self.extra_args["output"] += " pCl"
                # For modern experiments, always lensed Cl's!
                self.extra_args["output"] += " lCl"
                self.extra_args["lensing"] = "yes"
                self.collectors[k] = collector(method="lensed_cl", kwargs={})
                self.collectors["TCMB"] = collector(method="T_cmb", kwargs={})
            else:
                # Extra derived parameters
                if v is None:
                    self.derived_extra += [k]
                else:
                    self.log.error("Unknown required product: '%s:%s'.", k, v)
                    raise HandledException
        # Derived parameters (if some need some additional computations)
        if "sigma8" in self.output_params:
            self.extra_args["output"] += " mPk"
            self.extra_args["P_k_max_h/Mpc"] = (
                max(1, self.extra_args.get("P_k_max_h/Mpc", 0)))
        # Since the Cl collector needs lmax, update it now, in case it has increased
        # *after* declaring the Cl collector
        self.collectors["Cl"].kwargs.update({"lmax": self.extra_args["l_max_scalars"]})
        # Cleanup of products string
        self.extra_args["output"] = " ".join(set(self.extra_args["output"].split()))

    def translate_param(self, p):
        if self.use_camb_names:
            return self.camb_to_classy.get(p,p)
        return p

    def set(self, params_values_dict, i_state):
        # Store them, to use them later to identify the state
        self.states[i_state]["params"] = deepcopy(params_values_dict)
        # Prepare parameters to be passed: this-iteration + extra
        args = {self.translate_param(p):v for p,v in params_values_dict.items()}
        args.update(self.extra_args)
        # Generate and save
        self.log.debug("Setting parameters: %r", args)
        self.classy.struct_cleanup()
        self.classy.set(**args)

    def compute(self, derived=None, **params_values_dict):
        lasts = [self.states[i]["last"] for i in range(self.n_states)]
        try:
            # are the parameter values there already?
            i_state = (i for i in range(self.n_states)
                       if self.states[i]["params"] == params_values_dict).next()
            # Get (pre-computed) derived parameters
            if derived == {}:
                derived.update(self.states[i_state]["derived"])
            self.log.debug("Re-using computed results (state %d)", i_state)
        except StopIteration:
            # update the (first) oldest one and compute
            i_state = lasts.index(min(lasts))
            self.log.debug("Computing (state %d)", i_state)
            # Set parameters
            self.set(params_values_dict, i_state)
            # Compute!
            try:
                self.classy.compute()
            # "Valid" failure of CLASS: parameters too extreme -> log and report
            except CosmoComputationError:
                self.log.debug("Computation of cosmological products failed. "
                               "Assigning 0 likelihood and going on.")
                return False
            # CLASS not correctly initialised, or input parameters not correct
            except CosmoSevereError:
                self.log.error("Serious error setting parameters or computing results. "
                               "The parameters passed were %r and %r. "
                               "See original CLASS's error traceback below.\n",
                               self.states[i_state]["params"], self.extra_args)
                raise  # No HandledException, so that CLASS traceback gets printed
            # Gather products
            for product, collector in self.collectors.items():
                method = getattr(self.classy, collector.method)
                self.states[i_state][product] = method(**self.collectors[product].kwargs)
            # Prepare derived parameters
            d, d_extra = self.get_derived_all(derived_requested=(derived == {}))
            derived.update(d)
            self.states[i_state]["derived"] = odict(
                [[p,derived.get(p)] for p in self.output_params])
            # Prepare necessary extra derived parameters
            self.states[i_state]["derived_extra"] = deepcopy(d_extra)
        # make this one the current one by decreasing the antiquity of the rest
        for i in range(self.n_states):
            self.states[i]["last"] -= max(lasts)
        self.states[i_state]["last"] = 1
        return True

    def get_derived_all(self, derived_requested=True):
        """
        Returns a dictionary of derived parameters with their values,
        using the *current* state (i.e. it should only be called from
        the ``compute`` method).

        To get a parameter *from a likelihood* use `get_param` instead.
        """
        list_requested_derived = self.output_params if derived_requested else []
        de_translated = {self.translate_param(p):p for p in list_requested_derived}
        derived_aux = self.classy.get_current_derived_parameters(
            list(de_translated.keys())+list(self.derived_extra))
        derived = {de_translated[p]:derived_aux[p] for p in list_requested_derived}
        derived_extra = {p:derived_aux[p] for p in self.derived_extra}
        try:
            (p for p,v in derived.items() if v is None).next()
            self.log.error("Derived param '%s' not working in the CLASS interface", p)
            raise HandledException
        except StopIteration:
            pass  # all well!
        return derived, derived_extra

    def get_param(self, p):
        """
        Interface function for likelihoods to get sampled and derived parameters.
        """
        current_state = self.current_state()
        for pool in ["params", "derived", "derived_extra"]:
            value = current_state[pool].get(p, None)
            if value is not None:
                return value
        self.log.error("Parameter not known: '%s'", p)
        raise HandledException
    
    def get_cl(self, ell_factor=False):
        """
        Returns the :math:`C_{\ell}` from the cosmological code in :math:`\mu {\\rm K}^2`
        """
        current_state = self.current_state()
        # get C_l^XX from the cosmological code
        cl = current_state["Cl"]
        ell_factor = ((cl["ell"]+1)*cl["ell"]/(2*np.pi))[2:] if ell_factor else 1
        # convert dimensionless C_l's to C_l in muK**2
        T = current_state["TCMB"]
        for key in cl:
            # All quantities need to be multiplied by this factor, except the
            # phi-phi term, that is already dimensionless
            if key not in ['pp', 'ell']:
                cl[key][2:] *= (T*1.e6)**2 * ell_factor
        return cl


# Installation routines ##################################################################

def is_installed(**kwargs):
    try:
        if kwargs["code"]:
            import classy
    except:
        return False
    return True


def install(path=None, force=False, code=True, no_progress_bars=False, **kwargs):
    log = logging.getLogger("classy")
    if not code:
        log.info("Code not requested. Nothing to do.")
        return True
    log.info("Installing pre-requisites...")
    import pip
    exit_status = pip.main(["install", "cython", "--upgrade"] + user_flag_if_needed())
    if exit_status:
        log.error("Could not install pre-requisite: cython")
        return False
    log.info("Downloading...")
    parent_path = os.path.abspath(os.path.join(path, "code"))
    from wget import download, bar_thermometer
    try:
        filename = download(
            "https://github.com/lesgourg/class_public/archive/v2.6.1.tar.gz",
            out=parent_path, bar=(bar_thermometer if not no_progress_bars else None))
    except:
        log.error("Error downloading the latest release of CLASS.")
        return False
    print("")
    classy_path_decompressed = os.path.join(
        parent_path, os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0])
    classy_path = os.path.join(parent_path, "CLASS")
    if force and os.path.exists(classy_path):
        from shutil import rmtree
        rmtree(classy_path)
    log.info("Uncompressing...")
    import tarfile
    tar = tarfile.open(filename, "r:gz")
    try:
        tar.extractall(parent_path)
        tar.close()
        os.remove(filename)
    except:
        log.error("Error decompressing downloaded file! Corrupt file?)")
    os.rename(classy_path_decompressed, classy_path)
    from subprocess import Popen, PIPE
    working = False
    patch = False
    # Patch for gcc>=5
    while not working:
        os.chdir(classy_path)
        if patch:
            # patch (hopefully will be removed in the future)
            log.info("TRYING AGAIN: Patching for gcc>=5...")
            process_makeclean = Popen(["make", "clean"], stdout=PIPE, stderr=PIPE)
            out, err = process_makeclean.communicate()
            os.chdir("./python")
            process_pythonclean = Popen(
                ["python", "setup.py", "clean"], stdout=PIPE, stderr=PIPE)
            out, err = process_pythonclean.communicate()
            os.chdir("..")
            with open(os.path.join(classy_path, "python/setup.py"), "r") as setup:
                lines = setup.readlines()
                i_offending = (i for i,l in enumerate(lines) if "libraries=" in l).next()
                lines[i_offending] = "libraries=['class', 'mvec', 'm'],\n"
            with open(os.path.join(classy_path, "python/setup.py"), "w") as setup:
                setup.write("".join(lines))
        log.info("Compiling...")
        process_make = Popen(["make"], stdout=PIPE, stderr=PIPE)
        out, err = process_make.communicate()
        if process_make.returncode:
            log.info(out)
            log.info(err)
            log.error("Compilation failed!")
            return False
        log.info("Installing python package...")
        os.chdir("./python")
        process_install = Popen(["python", "setup.py", "install"] + user_flag_if_needed(),
                                stdout=PIPE, stderr=PIPE)
        out, err = process_install.communicate()
        if process_install.returncode:
            log.info(out)
            log.info(err)
            log.error("Installation failed!")
            return False
        # If installed but not importable, patch and try again
        if not is_installed(code=True):
            if patch:
                break
            patch = True
        else:
            working = True
    return True
