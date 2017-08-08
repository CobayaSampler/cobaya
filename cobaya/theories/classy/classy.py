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

# Local
from cobaya.theory import Theory
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)


# Dictionary of CAMB->CLASS names (only containing the different ones)
camb_to_class = {"ombh2":"omega_b",
                 "omegab": "Omega_b",
                 "omch2":"omega_cdm",
                 "As": "A_s",
                 "ns": "n_s",
                 "tau": "tau_reio"}

class classy(Theory):
    def initialise(self):
        """Importing CLASS from the correct path, if given, and if not, globally."""
        # If path not given, try using general path to modules
        if not self.path and self.path_to_installation:
            self.path = os.path.join(
                self.path_to_installation, subfolders[input_theories], "CLASS")
        if self.path:
            log.info("Importing *local* CLASS from "+self.path)
            if not os.path.exists(self.path):
                log.error("The given folder does not exist: '%s'", self.path)
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
                from classy import Class    
            except OSError:
                log.error("Either CLASS is not in the given folder,\n"
                          "'%s',\n or you have not compiled it.", self.path)
                raise HandledException
        else:
            log.info("Importing *global* CLASS.")
            try:
                from classy import Class
            except ImportError:
                log.error(
                    "Couldn't find the CLASS python interface.\n"
                    "Make sure that you have compiled it, and that you either\n"
                    " (a) specify a path (you didn't) or\n"
                    " (b) install the Python interface globally with\n"
                    "     '/path/to/class/python/python setup.py install --user'")
                raise HandledException
        # No output from CLASS!
        self.fixed["output"] = ""
        # Generate states, to avoid recomputing
        self.n_states = 3
        self.states = [{"classy": Class(), "params": None, "last": 0, "derived": None}
                       for i in range(self.n_states)]
        
    def current_state(self):
       lasts = [self.states[i]["last"] for i in range(self.n_states)]
       return self.states[lasts.index(max(lasts))]

    def set(self, params_values_dict, i_state):
        # Feed the arguments defining the cosmology to the cosmological code
        # Fixed (translate from camb naming)
        args = dict([(camb_to_class.get(p,p),v) for p,v in self.fixed.iteritems()])
        # Sampled -- save the state for avoiding recomputing later
        for p,v in params_values_dict.iteritems():
            args[camb_to_class.get(p,p)] = v
        # Generate and save
        log.debug("Setting parameters: %r", args)
        self.states[i_state]["params"] = params_values_dict
        self.states[i_state]["classy"].struct_cleanup()
        try:
            self.states[i_state]["classy"].set(**args)
        except Exception:
            log.error("Error setting CLASS parameters -- see CLASS's error trace below.\n"
                      "The parameters were %r", args)
            raise

    def compute(self, derived=None, **params_values_dict):
        lasts = [self.states[i]["last"] for i in range(self.n_states)]
        try:
            # are the parameter values there already?
            i_state = (i for i in range(self.n_states)
                       if self.states[i]["params"] == params_values_dict).next()
            # Get (pre-computed) derived parameters
            if derived == {}:
                derived.update(dict([[p,v] for p,v in
                                     zip(self.derived, self.states[i_state]["derived"])]))
            log.debug("Re-using computed results (state %d)", i_state)
        except StopIteration:
            # update the (first) oldest one and compute
            i_state = lasts.index(min(lasts))
            log.debug("Computing (state %d)", i_state)
            self.set(params_values_dict, i_state)
            self.states[i_state]["classy"].compute(["lensing"])
            # Prepare derived parameters
            if derived == {}:
                self.get_derived(derived, i_state)
                # Careful: next step must keep the order
                self.states[i_state]["derived"] = [derived[p] for p in self.derived]
        # make this one the current one by decreasing the antiquity of the rest
        for i in range(self.n_states):
            self.states[i]["last"] -= max(lasts)
        self.states[i_state]["last"] = 1

    def needs(self, arguments):
        for k,v in arguments.items():
            if k == "l_max":
                # Take the max of the requested ones
                self.fixed["l_max_scalars"] = max(v,self.fixed.get("l_max_scalars", 0))
            elif k == "Cl":
                self.fixed["output"] = "tCl pCl lCl"
                self.fixed["lensing"] = "yes"
            else:
                log.error("'%s' does not understand the requirement '%s:%s'.",
                    self.__class__.__name__,k,v)
                raise HandledException

    def get_derived(self, derived, i_state):
        """Populates a dictionary of derived parameters with their values, using the
        state #`i_state`."""
        for p in self.derived:
            p_classy = camb_to_class.get(p,p)
            # needs standard way to get derived parameters!
            if hasattr(self.states[i_state]["classy"], p_classy):
                derived[p] = getattr(self.states[i_state]["classy"], p_classy)()
            else:
                log.error("Derived param '%s' not implemented in the CLASS interface", p)
                raise HandledException

    def get_cl(self):
        """
        Returns the :math:`C_{\ell}` from the cosmological code in :math:`\mu {\\rm K}^2`
        """
        current_classy = self.current_state()["classy"]
        # get C_l^XX from the cosmological code
        cl = current_classy.lensed_cl(self.fixed["l_max_scalars"])
        # convert dimensionless C_l's to C_l in muK**2
        T = current_classy.T_cmb()
        for key in cl.iterkeys():
            # All quantities need to be multiplied by this factor, except the
            # phi-phi term, that is already dimensionless
            if key not in ['pp', 'ell']:
                cl[key] *= (T*1.e6)**2
        return cl


# Installation routines ###################################################################

def is_installed(**kwargs):
    try:
        import classy
    except:
        return False
    return True

def install(force=False, path=None ,**kwargs):
    log.info("Installing pre-requisites...")
    import pip
    exit_status = pip.main(["install", "cython", "--upgrade"]+
                           (["--user"] if not "TRAVIS" in os.environ else []))
    if exit_status:
        log.error("Could not install pre-requisite: cython")
        return False
    log.info("Downloading...")
    parent_path = os.path.abspath(os.path.join(path, ".."))
    from wget import download, bar_thermometer
    try:
        filename = download("https://github.com/lesgourg/class_public/archive/v2.6.1.tar.gz",
                            out=parent_path, bar=bar_thermometer)
    except:
        log.error("Error downloading the latest release of CLASS.")
        return False
    print ""
    classy_path = os.path.join(
        parent_path, os.path.splitext(os.path.splitext(os.path.basename(filename))[0])[0])
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
    # Patch for gcc>=5   
    from subprocess import Popen, PIPE
    # patch (hopefully will be removed in the future)
    log.info("Patching for gcc>=5...")
    with open(os.path.join(classy_path, "python/setup.py"), "r") as setup:
        lines = setup.readlines()
    i_offending = (i for i,l in enumerate(lines) if "libraries=" in l).next()
    lines[i_offending] = "libraries=['class', 'mvec', 'm'],\n"
    with open(os.path.join(classy_path, "python/setup.py"), "w") as setup:
        setup.write("".join(lines))
    log.info("Compiling...")
    os.chdir(classy_path)
    process_make = Popen(["make"], stdout=PIPE, stderr=PIPE)
    out, err = process_make.communicate()
    if process_make.returncode:
        log.info(out)
        log.info(err)
        log.error("Compilation failed!")
        return False
    log.info("Installing python package...")
    os.chdir("./python")
    process_install = Popen(["python", "setup.py", "install"]+
                            (["--user"] if not "TRAVIS" in os.environ else []),
                            stdout=PIPE, stderr=PIPE)
    out, err = process_install.communicate()
    if process_install.returncode:
        log.info(out)
        log.info(err)
        log.error("Installation failed!")
        return False
    return True
