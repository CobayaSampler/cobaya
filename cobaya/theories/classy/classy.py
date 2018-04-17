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
from cobaya.install import user_flag_if_needed, download_github_release
from cobaya.conventions import _c

# Result collector
collector = namedtuple("collector", ["method", "args", "kwargs", "arg_array"])
collector.__new__.__defaults__ = (None, [], {}, None)


class classy(Theory):

    def initialise(self):
        """Importing CLASS from the correct path, if given, and if not, globally."""
        # If path not given, try using general path to modules
        path_to_installation = get_path_to_installation()
        if not self.path and path_to_installation:
            self.path = os.path.join(
                path_to_installation, "code", classy_repo_rename)
        if self.path:
            self.log.info("Importing *local* classy from "+self.path)
            classy_build_path = os.path.join(self.path, "python", "build")
            post = next(d for d in os.listdir(classy_build_path) if d.startswith("lib."))
            classy_build_path = os.path.join(classy_build_path, post)
            if not os.path.exists(classy_build_path):
                self.log.error("Either CLASS is not in the given folder, "
                               "'%s', or you have not compiled it.", self.path)
                raise HandledException
            # Inserting the previously found path into the list of import folders
            sys.path.insert(0, classy_build_path)
        else:
            self.log.info("Importing *global* CLASS.")
        try:
            from classy import Class, CosmoSevereError, CosmoComputationError
        except ImportError:
            self.log.error(
                "Couldn't find the CLASS python interface. "
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
        arguments = arguments or {}
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
                self.extra_args["non linear"] = "halofit"
                self.collectors[k] = collector(method="lensed_cl", kwargs={})
                self.collectors["TCMB"] = collector(method="T_cmb", kwargs={})
            elif k == "fsigma8":
                self.collectors["growth_factor_f"] = collector(
                    method="scale_independent_growth_factor_f",
                    args=[np.atleast_1d(v["redshifts"])],
                    arg_array=0)
                self.collectors["sigma8"] = collector(
                    method="sigma",
                    # Notice: Needs H0 for 1st arg (R), so added later
                    args=[None, np.atleast_1d(v["redshifts"])],
                    arg_array=1)
                if "H0" not in self.input_params:
                    self.derived_extra += ["H0"]
                self.extra_args["output"] += " mPk"
                self.extra_args["P_k_max_h/Mpc"] = (
                    max(1, self.extra_args.get("P_k_max_h/Mpc", 0)))
                self.add_z_for_matter_power(v["redshifts"])
            elif k == "h_of_z":
                self.collectors[k] = collector(
                    method="Hubble",
                    args=[np.atleast_1d(v["redshifts"])],
                    arg_array=0)
                self.H_units_conv_factor = {"/Mpc": 1, "km/s/Mpc": _c}[v["units"]]
            elif k == "angular_diameter_distance":
                self.collectors[k] = collector(
                    method="angular_distance",
                    args=[np.atleast_1d(v["redshifts"])],
                    arg_array=0)
            else:
                # Extra derived parameters
                if v is None:
                    self.derived_extra += [self.translate_param(k)]
                else:
                    self.log.error("Unknown required product: '%s:%s'.", k, v)
                    raise HandledException
        # Derived parameters (if some need some additional computations)
        if "sigma8" in self.output_params or arguments:
            self.extra_args["output"] += " mPk"
            self.extra_args["P_k_max_h/Mpc"] = (
                max(1, self.extra_args.get("P_k_max_h/Mpc", 0)))
        # Since the Cl collector needs lmax, update it now, in case it has increased
        # *after* declaring the Cl collector
        if "Cl" in self.collectors:
            self.collectors["Cl"].kwargs["lmax"] = self.extra_args["l_max_scalars"]
        # Cleanup of products string
        self.extra_args["output"] = " ".join(set(self.extra_args["output"].split()))

    def add_z_for_matter_power(self, z):
        if not hasattr(self, "z_for_matter_power"):
            self.z_for_matter_power = np.empty((0))
        self.z_for_matter_power = np.flip(np.sort(np.unique(np.concatenate(
            [self.z_for_matter_power, np.atleast_1d(z)]))), axis=0)
        self.extra_args["z_pk"] = " ".join(["%g"%zi for zi in self.z_for_matter_power])

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
            i_state = next(i for i in range(self.n_states)
                           if self.states[i]["params"] == params_values_dict)
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
                # Special case: sigma8 needs H0, which cannot be known beforehand:
                if "sigma8" in self.collectors:
                    self.collectors["sigma8"].args[0] = 8/self.classy.h()
                method = getattr(self.classy, collector.method)
                if self.collectors[product].arg_array is None:
                    self.states[i_state][product] = method(
                        *self.collectors[product].args, **self.collectors[product].kwargs)
                else:
                    i_array = self.collectors[product].arg_array
                    self.states[i_state][product] = np.zeros(
                        len(self.collectors[product].args[i_array]))
                    for i,v in enumerate(self.collectors[product].args[i_array]):
                        args = (list(self.collectors[product].args[:i_array]) + [v] +
                                list(self.collectors[product].args[i_array+1:]))
                        self.states[i_state][product][i] = method(
                            *args, **self.collectors[product].kwargs)
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
        requested_derived_with_extra = list(de_translated.keys())+list(self.derived_extra)
        derived_aux = {}
        # Exceptions
        if "rs_drag" in requested_derived_with_extra:
            requested_derived_with_extra.remove("rs_drag")
            derived_aux["rs_drag"] = self.classy.rs_drag()
        derived_aux.update(
            self.classy.get_current_derived_parameters(requested_derived_with_extra))
        # Fill return dictionaries
        derived = {de_translated[p]:derived_aux[self.translate_param(p)]
                   for p in de_translated}
        derived_extra = {p:derived_aux[p] for p in self.derived_extra}
        # No need for error control: classy.get_current_derived_parameters is passed
        # every derived parameter not excluded before, and cause an error if if founds a
        # parameter that it does not recognise
        return derived, derived_extra

    def get_param(self, p):
        """
        Interface function for likelihoods to get sampled and derived parameters.
        """
        current_state = self.current_state()
        for pool in ["params", "derived", "derived_extra"]:
            value = current_state[pool].get(self.translate_param(p), None)
            if value is not None:
                return value
        self.log.error("Parameter not known: '%s'", p)
        raise HandledException

    def get_cl(self, ell_factor=False):
        """
        Returns the power spectra in microK^2
        (unitless for lensing potential),
        using the *current* state.
        """
        current_state = self.current_state()
        # get C_l^XX from the cosmological code
        try:
            cl = deepcopy(current_state["Cl"])
        except:
            self.log.error(
                "No Cl's were computed. Are you sure that you have requested them?")
            raise HandledException
        ell_factor = ((cl["ell"]+1)*cl["ell"]/(2*np.pi))[2:] if ell_factor else 1
        # convert dimensionless C_l's to C_l in muK**2
        T = current_state["TCMB"]
        for key in cl:
            # All quantities need to be multiplied by this factor, except the
            # phi-phi term, that is already dimensionless
            if key not in ['pp', 'ell']:
                cl[key][2:] *= (T*1.e6)**2 * ell_factor
        if "pp" in cl and ell_factor is not 1:
            cl['pp'][2:] *= ell_factor**2 * (2*np.pi)
        return cl

    def get_fsigma8(self, z):
        indices = np.where(self.z_for_matter_power == z)
        return (self.current_state()["growth_factor_f"][indices] *
                self.current_state()["sigma8"][indices])

    def get_h_of_z(self, z):
        return self.current_state()["h_of_z"][
            np.where(self.collectors["h_of_z"].args[
                self.collectors["h_of_z"].arg_array] == z)]*self.H_units_conv_factor

    def get_angular_diameter_distance(self, z):
        return self.current_state()["angular_diameter_distance"][
            np.where(self.collectors["angular_diameter_distance"].args[
                self.collectors["angular_diameter_distance"].arg_array] == z)]


# Installation routines ##################################################################

# Name of the Class repo/folder and version to download
classy_repo_name = "class_public"
classy_repo_rename = "classy"
classy_repo_user = "lesgourg"
classy_repo_version = "v2.6.3"


def is_installed(**kwargs):
    if not kwargs["code"]:
        return True
    return os.path.isfile(os.path.realpath(
        os.path.join(kwargs["path"], "code", classy_repo_rename, "libclass.a")))


def install(path=None, force=False, code=True, no_progress_bars=False, **kwargs):
    log = logging.getLogger(__name__.split(".")[-1])
    if not code:
        log.info("Code not requested. Nothing to do.")
        return True
    log.info("Installing pre-requisites...")
    import pip
    exit_status = pip.main(["install", "cython", "--upgrade"] + user_flag_if_needed())
    if exit_status:
        log.error("Could not install pre-requisite: cython")
        return False
    log.info("Downloading classy...")
    success = download_github_release(
        os.path.join(path, "code"), classy_repo_name,classy_repo_version,
        github_user=classy_repo_user, repo_rename=classy_repo_rename,
        no_progress_bars=no_progress_bars)
    if not success:
        log.error("Could not download classy.")
        return False
    classy_path = os.path.join(path, "code", classy_repo_rename)
    log.info("Compiling classy...")
    from subprocess import Popen, PIPE
    process_make = Popen(["make"], cwd=classy_path, stdout=PIPE, stderr=PIPE)
    out, err = process_make.communicate()
    if process_make.returncode:
        log.info(out)
        log.info(err)
        log.error("Compilation failed!")
        return False
    return True
