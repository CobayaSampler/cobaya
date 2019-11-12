"""
.. module:: theories.classy

:Synopsis: Managing the CLASS cosmological code
:Author: Jesus Torrado
         (import and ``get_Cl`` based on MontePython's CLASS wrapper Benjamin Audren)

.. |br| raw:: html

   <br />

This module imports and manages the CLASS cosmological code.

.. note::

   **If you use this cosmological code, please cite it as:**
   |br|
   D. Blas, J. Lesgourgues, T. Tram,
   *The Cosmic Linear Anisotropy Solving System (CLASS). Part II: Approximation schemes*
   (`arXiv:1104.2933 <https://arxiv.org/abs/1104.2933>`_)

.. note::

   CLASS is renamed ``classy`` for most purposes within cobaya, due to CLASS's name being
   a python keyword.

Usage
-----

If you are using a likelihood that requires some observable from CLASS, simply add
``classy`` to the theory block.

You can specify any parameter that CLASS understands in the ``params`` block:

.. code-block:: yaml

   theory:
     classy:
       extra_args:
         [any param that CLASS understands]

   params:
       [any param that CLASS understands, fixed, sampled or derived]

If you want to use your own version of CLASS, you need to specify its location with a
``path`` option inside the ``classy`` block. If you do not specify a ``path``,
CLASS will be loaded from the automatic-install ``modules`` folder, if specified, or
otherwise imported as a globally-installed Python package. Cobaya will print at
initialisation where it is getting CLASS from.

.. _classy_modify:

Modifying CLASS
^^^^^^^^^^^^^^^

If you modify CLASS and add new variables, make sure that the variables you create are
exposed in the Python interface
(`instructions here <https://github.com/lesgourg/class_public/wiki/Python-wrapper>`__).
If you follow those instructions you do not need to make any additional modification in
Cobaya.

You can use the :doc:`model wrapper <cosmo_model>` to test your modification by
evaluating observables or getting derived quantities at known points in the parameter
space (set ``debug: True`` to get more detailed information of what exactly is passed to
CLASS).

In your CLASS modification, remember that you can raise a ``CosmoComputationError``
whenever the computation of any observable would fail, but you do not expect that
observable to be compatible with the data (e.g. at the fringes of the parameter
space). Whenever such an error is raised during sampling, the likelihood is assumed to be
zero, and the run is not interrupted.


Installation
------------

   .. _classy_install_warn:

.. warning::

   If the installation folder of CLASS is moved, due to CLASS hard-coding some folders,
   CLASS needs to be recompiled, either manually or by deleting the CLASS installation and
   repeating the ``cobaya-install`` command in the renamed *modules* folder.

   If you do not recompile CLASS, it causes a memory leak (`thanks to Stefan Heimersheim
   <https://github.com/CobayaSampler/cobaya/issues/10>`_).

Automatic installation
^^^^^^^^^^^^^^^^^^^^^^

If you do not plan to modify CLASS, the easiest way to install it is using the
:doc:`automatic installation script <installation_cosmo>`. Just make sure that
``theory: classy:`` appears in one of the files passed as arguments to the installation
script.

Manual installation (or using your own version)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are planning to modify CLASS or use an already modified version,
you should not use the automatic installation script. Use the method below instead.

CLASS's python interface utilizes the ``cython`` compiler. If typing ``cython`` in the
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
"""
# Python 2/3 compatibility
from __future__ import absolute_import, division

# Global
import sys
import os
import numpy as np
from copy import deepcopy
import logging
from collections import namedtuple, OrderedDict as odict
from time import time
from numbers import Number

# Local
from cobaya.theories._cosmo import _cosmo, PowerSpectrumInterpolator
from cobaya.log import LoggedError
from cobaya.install import download_github_release, pip_install
from cobaya.conventions import _c_km_s, _T_CMB_K
from cobaya.tools import deepcopy_where_possible

# Result collector
collector = namedtuple("collector",
                       ["method", "args", "args_names", "kwargs", "arg_array", "post"])
collector.__new__.__defaults__ = (None, [], [], {}, None, None)

# default non linear code
non_linear_default_code = "halofit"


class classy(_cosmo):
    # Name of the Class repo/folder and version to download
    classy_repo_name = "lesgourg/class_public"
    classy_repo_version = "v2.7.2"

    def initialize(self):
        """Importing CLASS from the correct path, if given, and if not, globally."""
        # If path not given, try using general path to modules
        if not self.path and self.path_install:
            self.path = self.get_path(self.path_install)
        if self.path:
            self.log.info("Importing *local* classy from " + self.path)
            classy_build_path = os.path.join(self.path, "python", "build")
            post = next(d for d in os.listdir(classy_build_path) if d.startswith("lib."))
            classy_build_path = os.path.join(classy_build_path, post)
            if not os.path.exists(classy_build_path):
                # If path was given as an install path, try to install global one anyway
                if self.path_install:
                    self.log.info("Importing *global* CLASS (because not installed).")
                else:
                    raise LoggedError(
                        self.log, "Either CLASS is not in the given folder, "
                                  "'%s', or you have not compiled it.", self.path)
            else:
                # Inserting the previously found path into the list of import folders
                sys.path.insert(0, classy_build_path)
        else:
            self.log.info("Importing *global* CLASS.")
        try:
            from classy import Class, CosmoSevereError, CosmoComputationError
        except ImportError:
            raise LoggedError(
                self.log, "Couldn't find the CLASS python interface. "
                          "Make sure that you have compiled it, and that you either\n"
                          " (a) specify a path (you didn't) or\n"
                          " (b) install the Python interface globally with\n"
                          "     '/path/to/class/python/python setup.py install --user'")
        self.classy = Class()
        # Propagate errors up
        global CosmoComputationError, CosmoSevereError
        # Generate states, to avoid recomputing
        self._n_states = 3
        self._states = [{"params": None, "derived": None, "derived_extra": None,
                        "last": 0} for i in range(self._n_states)]
        # Dict of named tuples to collect requirements and computation methods
        self.collectors = {}
        # Additional input parameters to pass to CLASS
        self.extra_args = deepcopy_where_possible(self.extra_args) or {}
        # Add general CLASS stuff
        self.extra_args["output"] = self.extra_args.get("output", "")
        if "sBBN file" in self.extra_args:
            self.extra_args["sBBN file"] = (
                self.extra_args["sBBN file"].format(classy=self.path))
        # Set aliases
        self.planck_to_classy = self.renames
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []

    def current_state(self):
        lasts = [self._states[i]["last"] for i in range(self._n_states)]
        return self._states[lasts.index(max(lasts))]

    def needs(self, **requirements):
        # Computed quantities required by the likelihood
        super(classy, self).needs(**requirements)
        for k, v in self._needs.items():
            # Products and other computations
            if k == "Cl":
                if any([("t" in cl.lower()) for cl in v]):
                    self.extra_args["output"] += " tCl"
                if any([(("e" in cl.lower()) or ("b" in cl.lower())) for cl in v]):
                    self.extra_args["output"] += " pCl"
                # For modern experiments, always lensed Cl's!
                self.extra_args["output"] += " lCl"
                self.extra_args["lensing"] = "yes"
                # For l_max_scalars, remember previous entries.
                self.extra_args["l_max_scalars"] = max(v.values())
                self.collectors[k] = collector(
                    method="lensed_cl", kwargs={"lmax": self.extra_args["l_max_scalars"]})
            elif k == "H":
                self.collectors[k] = collector(
                    method="Hubble",
                    args=[np.atleast_1d(v["z"])],
                    args_names=["z"],
                    arg_array=0)
                self.H_units_conv_factor = {"1/Mpc": 1, "km/s/Mpc": _c_km_s}
            elif k == "angular_diameter_distance":
                self.collectors[k] = collector(
                    method="angular_distance",
                    args=[np.atleast_1d(v["z"])],
                    args_names=["z"],
                    arg_array=0)
            elif k == "comoving_radial_distance":
                self.collectors[k] = collector(
                    method="z_of_r",
                    args_names=["z"],
                    args=[np.atleast_1d(v["z"])])
            elif k == "Pk_interpolator":
                self.extra_args["output"] += " mPk"
                self.extra_args["P_k_max_h/Mpc"] = max(
                    v.pop("k_max"), self.extra_args.get("P_k_max_h/Mpc", 0))
                self.add_z_for_matter_power(v.pop("z"))
                # Use halofit by default if non-linear requested but no code specified
                if v.get("nonlinear", False) and "non linear" not in self.extra_args:
                    self.extra_args["non linear"] = non_linear_default_code
                for pair in v.pop("vars_pairs", [["delta_tot", "delta_tot"]]):
                    if any([x != "delta_tot" for x in pair]):
                        raise LoggedError(self.log, "NotImplemented in CLASS: %r", pair)
                    self._Pk_interpolator_kwargs = {
                        "logk": True, "extrap_kmax": v.pop("extrap_kmax", None)}
                    name = "Pk_interpolator_%s_%s" % (pair[0], pair[1])
                    self.collectors[name] = collector(
                        method="get_pk_and_k_and_z",
                        kwargs=v,
                        post=(lambda P, k, z: PowerSpectrumInterpolator(
                            z, k, P.T, **self._Pk_interpolator_kwargs)))
            elif v is None:
                k_translated = self.translate_param(k, force=True)
                if k_translated not in self.derived_extra:
                    self.derived_extra += [k_translated]
            else:
                raise LoggedError(self.log, "Requested product not known: %r", {k: v})
        # Derived parameters (if some need some additional computations)
        if any([("sigma8" in s) for s in self.output_params or requirements]):
            self.extra_args["output"] += " mPk"
            self.extra_args["P_k_max_h/Mpc"] = (
                max(1, self.extra_args.get("P_k_max_h/Mpc", 0)))
        # Adding tensor modes if requested
        if self.extra_args.get("r") or "r" in self.input_params:
            self.extra_args["modes"] = "s,t"
        # If B spectrum with l>50, or lensing, recommend using Halofit
        cls = self._needs.get("Cl", {})
        if (((any([("b" in cl.lower()) for cl in cls]) and
              max([cls[cl] for cl in cls if "b" in cl.lower()]) > 50) or
             any([("p" in cl.lower()) for cl in cls]) and
             not self.extra_args.get("non linear"))):
            self.log.warning("Requesting BB for ell>50 or lensing Cl's: "
                             "using a non-linear code is recommended (and you are not "
                             "using any). To activate it, set "
                             "'non_linear: halofit|hmcode|...' in classy's 'extra_args'.")
        # Cleanup of products string
        self.extra_args["output"] = " ".join(set(self.extra_args["output"].split()))
        # If no output requested, remove arguments that produce an error
        # (e.g. complaints if halofit requested but no Cl's computed.)
        # Needed for facilitating post-processing
        if not self.extra_args["output"]:
            for k in ["non linear"]:
                if k in self.extra_args:
                    self.log.info("Ignoring {%s: %r}, since no products requested.",
                                  k, self.extra_args[k])
                    self.extra_args.pop(k)
        # Finally, check that there are no repeated parameters between input and extra
        if set(self.input_params).intersection(set(self.extra_args)):
            raise LoggedError(
                self.log,
                "The following parameters appear both as input parameters and as CLASS "
                "extra arguments: %s. Please, remove one of the definitions of each.",
                list(set(self.input_params).intersection(set(self.extra_args))))

    def add_z_for_matter_power(self, z):
        if not getattr(self, "z_for_matter_power"):
            self.z_for_matter_power = np.empty(0)
        self.z_for_matter_power = np.flip(np.sort(np.unique(np.concatenate(
            [self.z_for_matter_power, np.atleast_1d(z)]))), axis=0)
        self.extra_args["z_pk"] = " ".join(["%g" % zi for zi in self.z_for_matter_power])

    def translate_param(self, p, force=False):
        # "force=True" is used when communicating with likelihoods, which speak "planck"
        if self.use_planck_names or force:
            return self.planck_to_classy.get(p, p)
        return p

    def set(self, params_values_dict, i_state):
        # Store them, to use them later to identify the state
        self._states[i_state]["params"] = deepcopy(params_values_dict)
        # Prepare parameters to be passed: this-iteration + extra
        args = {self.translate_param(p): v for p, v in params_values_dict.items()}
        args.update(self.extra_args)
        # Generate and save
        self.log.debug("Setting parameters: %r", args)
        self.classy.struct_cleanup()
        self.classy.set(**args)

    def compute(self, _derived=None, cached=True, **params_values_dict):
        lasts = [self._states[i]["last"] for i in range(self._n_states)]
        try:
            if not cached:
                raise StopIteration
            # are the parameter values there already?
            i_state = next(i for i in range(self._n_states)
                           if self._states[i]["params"] == params_values_dict)
            # has any new product been requested?
            for product in self.collectors:
                next(k for k in self._states[i_state] if k == product)
            reused_state = True
            # Get (pre-computed) derived parameters
            if _derived == {}:
                _derived.update(self._states[i_state]["derived"])
            self.log.debug("Re-using computed results (state %d)", i_state)
        except StopIteration:
            reused_state = False
            # update the (first) oldest one and compute
            i_state = lasts.index(min(lasts))
            self.log.debug("Computing (state %d)", i_state)
            if self.timer:
                self.timer.start()
            # Set parameters
            self.set(params_values_dict, i_state)
            # Compute!
            try:
                self.classy.compute()
            # "Valid" failure of CLASS: parameters too extreme -> log and report
            except CosmoComputationError as e:
                if ("You have asked for an unrealistic high value omega_b" in str(e) or
                    "reionization cannot start after z_start_max" in str(e) or
                    "Shooting failed, try optimising input_get_guess()" in str(e)):
                    pass
                else:
                    self.log.debug("Computation of cosmological products failed. "
                                   "Assigning 0 likelihood and going on. "
                                   "The output of the CLASS error was %s" % e)
                return 0
            # CLASS not correctly initialized, or input parameters not correct
            except CosmoSevereError:
                self.log.error("Serious error setting parameters or computing results. "
                               "The parameters passed were %r and %r. "
                               "See original CLASS's error traceback below.\n",
                               self._states[i_state]["params"], self.extra_args)
                raise  # No LoggedError, so that CLASS traceback gets printed
            # Gather products
            for product, collector in self.collectors.items():
                # Special case: sigma8 needs H0, which cannot be known beforehand:
                if "sigma8" in self.collectors:
                    self.collectors["sigma8"].args[0] = 8 / self.classy.h()
                method = getattr(self.classy, collector.method)
                arg_array = self.collectors[product].arg_array
                if arg_array is None:
                    self._states[i_state][product] = method(
                        *self.collectors[product].args, **self.collectors[product].kwargs)
                elif isinstance(arg_array, Number):
                    self._states[i_state][product] = np.zeros(
                        len(self.collectors[product].args[arg_array]))
                    for i, v in enumerate(self.collectors[product].args[arg_array]):
                        args = (list(self.collectors[product].args[:arg_array]) + [v] +
                                list(self.collectors[product].args[arg_array + 1:]))
                        self._states[i_state][product][i] = method(
                            *args, **self.collectors[product].kwargs)
                elif arg_array in self.collectors[product].kwargs:
                    value = np.atleast_1d(self.collectors[product].kwargs[arg_array])
                    self._states[i_state][product] = np.zeros(value.shape)
                    for i, v in enumerate(value):
                        kwargs = deepcopy(self.collectors[product].kwargs)
                        kwargs[arg_array] = v
                        self._states[i_state][product][i] = method(
                            *self.collectors[product].args, **kwargs)
                if collector.post:
                    self._states[i_state][product] = collector.post(
                        *self._states[i_state][product])
            # Prepare derived parameters
            d, d_extra = self._get_derived_all(derived_requested=(_derived == {}))
            if _derived == {}:
                _derived.update(d)
            self._states[i_state]["derived"] = odict(
                [[p, (_derived or {}).get(p)] for p in self.output_params])
            # Prepare necessary extra derived parameters
            self._states[i_state]["derived_extra"] = deepcopy(d_extra)
            if self.timer:
                self.timer.increment(self.log)
        # make this one the current one by decreasing the antiquity of the rest
        for i in range(self._n_states):
            self._states[i]["last"] -= max(lasts)
        self._states[i_state]["last"] = 1
        return 1 if reused_state else 2

    def _get_derived_all(self, derived_requested=True):
        """
        Returns a dictionary of derived parameters with their values,
        using the *current* state (i.e. it should only be called from
        the ``compute`` method).

        Parameter names are returned in CLASS nomenclature.

        To get a parameter *from a likelihood* use `get_param` instead.
        """
        # Put all pamaremters in CLASS nomenclature (self.derived_extra already is)
        requested = [self.translate_param(p) for p in (
            self.output_params if derived_requested else [])]
        requested_and_extra = {
            p: None for p in set(requested).union(set(self.derived_extra))}
        # Parameters with their own getters
        if "rs_drag" in requested_and_extra:
            requested_and_extra["rs_drag"] = self.classy.rs_drag()
        elif "Omega_nu" in requested_and_extra:
            requested_and_extra["Omega_nu"] = self.classy.Omega_nu
        # Get the rest using the general derived param getter
        # No need for error control: classy.get_current_derived_parameters is passed
        # every derived parameter not excluded before, and cause an error, indicating
        # which parameters are not recognized
        requested_and_extra.update(
            self.classy.get_current_derived_parameters(
                [p for p, v in requested_and_extra.items() if v is None]))
        # Separate the parameters before returning
        # Remember: self.output_params is in sampler nomenclature,
        # but self.derived_extra is in CLASS
        derived = {
            p: requested_and_extra[self.translate_param(p)] for p in self.output_params}
        derived_extra = {p: requested_and_extra[p] for p in self.derived_extra}
        return derived, derived_extra

    def get_param(self, p):
        current_state = self.current_state()
        for pool in ["params", "derived", "derived_extra"]:
            value = deepcopy(
                current_state[pool].get(self.translate_param(p, force=True), None))
            if value is not None:
                return value
        raise LoggedError(self.log, "Parameter not known: '%s'", p)

    def get_Cl(self, ell_factor=False, units="muK2"):
        current_state = self.current_state()
        try:
            cls = deepcopy(current_state["Cl"])
        except:
            raise LoggedError(
                self.log,
                "No Cl's were computed. Are you sure that you have requested them?")
        # unit conversion and ell_factor
        ell_factor = ((cls["ell"] + 1) * cls["ell"] / (2 * np.pi))[2:] if ell_factor else 1
        units_factors = {"1": 1,
                         "muK2": _T_CMB_K * 1.e6,
                         "K2": _T_CMB_K}
        try:
            units_factor = units_factors[units]
        except KeyError:
            raise LoggedError(self.log, "Units '%s' not recognized. Use one of %s.",
                              units, list(units_factors))
        for cl in cls:
            if cl not in ['pp', 'ell']:
                cls[cl][2:] *= units_factor ** 2 * ell_factor
        if "pp" in cls and ell_factor is not 1:
            cls['pp'][2:] *= ell_factor ** 2 * (2 * np.pi)
        return cls

    def _get_z_dependent(self, quantity, z):
        try:
            z_name = next(k for k in ["redshifts", "z"]
                          if k in self.collectors[quantity].kwargs)
            computed_redshifts = self.collectors[quantity].kwargs[z_name]
        except StopIteration:
            computed_redshifts = self.collectors[quantity].args[
                self.collectors[quantity].args_names.index("z")]
        i_kwarg_z = np.concatenate(
            [np.where(computed_redshifts == zi)[0] for zi in np.atleast_1d(z)])
        values = np.array(deepcopy(self.current_state()[quantity]))
        if quantity == "comoving_radial_distance":
            values = values[0]
        return values[i_kwarg_z]

    def get_H(self, z, units="km/s/Mpc"):
        try:
            return self._get_z_dependent("H", z) * self.H_units_conv_factor[units]
        except KeyError:
            raise LoggedError(
                self.log, "Units not known for H: '%s'. Try instead one of %r.",
                units, list(self.H_units_conv_factor))

    def get_angular_diameter_distance(self, z):
        return self._get_z_dependent("angular_diameter_distance", z)

    def get_comoving_radial_distance(self, z):
        return self._get_z_dependent("comoving_radial_distance", z)

    def get_Pk_interpolator(self):
        current_state = self.current_state()
        prefix = "Pk_interpolator_"
        return {k[len(prefix):]: deepcopy(v)
                for k, v in current_state.items() if k.startswith(prefix)}

    def close(self):
        self.classy.struct_cleanup()

    # Installation routines ##################################################################

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(os.path.join(path, "code", cls.__name__))

    @classmethod
    def is_installed(cls, **kwargs):
        if not kwargs["code"]:
            return True
        return os.path.isfile(os.path.join(cls.get_path(kwargs["path"]), "libclass.a"))

    @classmethod
    def install(cls, path=None, force=False, code=True, no_progress_bars=False, **kwargs):
        log = logging.getLogger(cls.__name__)
        if not code:
            log.info("Code not requested. Nothing to do.")
            return True
        log.info("Installing pre-requisites...")
        exit_status = pip_install("cython")
        if exit_status:
            log.error("Could not install pre-requisite: cython")
            return False
        log.info("Downloading classy...")
        success = download_github_release(
            os.path.join(path, "code"), cls.classy_repo_name, cls.classy_repo_version,
            repo_rename=cls.__name__, no_progress_bars=no_progress_bars, logger=log)
        if not success:
            log.error("Could not download classy.")
            return False
        classy_path = cls.get_path(path)
        log.info("Compiling classy...")
        from subprocess import Popen, PIPE
        env = deepcopy(os.environ)
        env.update({"PYTHON": sys.executable})
        process_make = Popen(["make"], cwd=classy_path, stdout=PIPE, stderr=PIPE, env=env)
        out, err = process_make.communicate()
        if process_make.returncode:
            log.info(out)
            log.info(err)
            log.error("Compilation failed!")
            return False
        return True
