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
CLASS will be loaded from the automatic-install ``packages_path`` folder, if specified, or
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
   repeating the ``cobaya-install`` command in the renamed *packages* folder.

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
shell produces an error, install it with ``python -m pip install cython --user``.

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

# Global
import sys
import os
import numpy as np
from copy import deepcopy
import logging
from typing import NamedTuple, Sequence, Union, Optional

# Local
from cobaya.theories._cosmo import BoltzmannBase
from cobaya.log import LoggedError
from cobaya.install import download_github_release, pip_install, NotInstalledError
from cobaya.tools import load_module, VersionCheckError


# Result collector
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None


# default non linear code -- same as CAMB
non_linear_default_code = "hmcode"


class classy(BoltzmannBase):
    # Name of the Class repo/folder and version to download
    _classy_repo_name = "lesgourg/class_public"
    _min_classy_version = "v2.9.3"
    _classy_repo_version = os.environ.get('CLASSY_REPO_VERSION', _min_classy_version)

    def initialize(self):
        """Importing CLASS from the correct path, if given, and if not, globally."""
        # Allow global import if no direct path specification
        allow_global = not self.path
        if not self.path and self.packages_path:
            self.path = self.get_path(self.packages_path)
        self.classy_module = self.is_installed(path=self.path, allow_global=allow_global)
        if not self.classy_module:
            raise NotInstalledError(
                self.log, "Could not find CLASS. Check error message above.")
        from classy import Class, CosmoSevereError, CosmoComputationError
        global CosmoComputationError, CosmoSevereError
        self.classy = Class()
        super().initialize()
        # Add general CLASS stuff
        self.extra_args["output"] = self.extra_args.get("output", "")
        if "sBBN file" in self.extra_args:
            self.extra_args["sBBN file"] = (
                self.extra_args["sBBN file"].format(classy=self.path))
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []
        self.log.info("Initialized!")

    def must_provide(self, **requirements):
        # Computed quantities required by the likelihood
        super().must_provide(**requirements)
        for k, v in self._must_provide.items():
            # Products and other computations
            if k == "Cl":
                if any(("t" in cl.lower()) for cl in v):
                    self.extra_args["output"] += " tCl"
                if any((("e" in cl.lower()) or ("b" in cl.lower())) for cl in v):
                    self.extra_args["output"] += " pCl"
                # For modern experiments, always lensed Cl's!
                self.extra_args["output"] += " lCl"
                self.extra_args["lensing"] = "yes"
                # For l_max_scalars, remember previous entries.
                self.extra_args["l_max_scalars"] = max(v.values())
                self.collectors[k] = Collector(
                    method="lensed_cl", kwargs={"lmax": self.extra_args["l_max_scalars"]})
                if 'T_cmb' not in self.derived_extra:
                    self.derived_extra += ['T_cmb']
            elif k == "Hubble":
                self.collectors[k] = Collector(
                    method="Hubble",
                    args=[np.atleast_1d(v["z"])],
                    args_names=["z"],
                    arg_array=0)
            elif k == "angular_diameter_distance":
                self.collectors[k] = Collector(
                    method="angular_distance",
                    args=[np.atleast_1d(v["z"])],
                    args_names=["z"],
                    arg_array=0)
            elif k == "comoving_radial_distance":
                self.collectors[k] = Collector(
                    method="z_of_r",
                    args_names=["z"],
                    args=[np.atleast_1d(v["z"])])
            elif isinstance(k, tuple) and k[0] == "Pk_grid":
                self.extra_args["output"] += " mPk"
                v = deepcopy(v)
                self.add_P_k_max(v.pop("k_max"), units="1/Mpc")
                # NB: Actually, only the max z is used, and the actual sampling in z
                # for computing P(k,z) is controlled by `perturb_sampling_stepsize`
                # (default: 0.1). But let's leave it like this in case this changes
                # in the future.
                self.add_z_for_matter_power(v.pop("z"))

                if v["nonlinear"] and "non linear" not in self.extra_args:
                    self.extra_args["non linear"] = non_linear_default_code
                pair = k[2:]
                if pair == ("delta_tot", "delta_tot"):
                    v["only_clustering_species"] = False
                elif pair == ("delta_nonu", "delta_nonu"):
                    v["only_clustering_species"] = True
                else:
                    raise LoggedError(self.log, "NotImplemented in CLASS: %r", pair)
                self.collectors[k] = Collector(
                    method="get_pk_and_k_and_z",
                    kwargs=v,
                    post=(lambda P, kk, z: (kk, z, np.array(P).T)))
            elif isinstance(k, tuple) and k[0] == "sigma_R":
                raise LoggedError(
                    self.log, "Classy sigma_R not implemented as yet - use CAMB only")
            elif v is None:
                k_translated = self.translate_param(k)
                if k_translated not in self.derived_extra:
                    self.derived_extra += [k_translated]
            else:
                raise LoggedError(self.log, "Requested product not known: %r", {k: v})
        # Derived parameters (if some need some additional computations)
        if any(("sigma8" in s) for s in self.output_params or requirements):
            self.extra_args["output"] += " mPk"
            self.add_P_k_max(1, units="1/Mpc")
        # Adding tensor modes if requested
        if self.extra_args.get("r") or "r" in self.input_params:
            self.extra_args["modes"] = "s,t"
        # If B spectrum with l>50, or lensing, recommend using Halofit
        cls = self._must_provide.get("Cl", {})
        if (((any(("b" in cl.lower()) for cl in cls) and
              max(cls[cl] for cl in cls if "b" in cl.lower()) > 50) or
             any(("p" in cl.lower()) for cl in cls) and
             not self.extra_args.get("non linear"))):
            self.log.warning("Requesting BB for ell>50 or lensing Cl's: "
                             "using a non-linear code is recommended (and you are not "
                             "using any). To activate it, set "
                             "'non_linear: halofit|hmcode|...' in classy's 'extra_args'.")
        # Cleanup of products string
        self.extra_args["output"] = " ".join(set(self.extra_args["output"].split()))
        self.check_no_repeated_input_extra()

    def add_z_for_matter_power(self, z):
        if getattr(self, "z_for_matter_power", None) is None:
            self.z_for_matter_power = np.empty(0)
        self.z_for_matter_power = np.flip(np.sort(np.unique(np.concatenate(
            [self.z_for_matter_power, np.atleast_1d(z)]))), axis=0)
        self.extra_args["z_pk"] = " ".join(["%g" % zi for zi in self.z_for_matter_power])

    def add_P_k_max(self, k_max, units):
        r"""
        Unifies treatment of :math:`k_\mathrm{max}` for matter power spectrum:
        ``P_k_max_[1|h]/Mpc]``.

        Make ``units="1/Mpc"|"h/Mpc"``.
        """
        # Fiducial h conversion (high, though it may slow the computations)
        h_fid = 1
        if units == "h/Mpc":
            k_max *= h_fid
        # Take into account possible manual set of P_k_max_***h/Mpc*** through extra_args
        k_max_old = self.extra_args.pop(
            "P_k_max_1/Mpc", h_fid * self.extra_args.pop("P_k_max_h/Mpc", 0))
        self.extra_args["P_k_max_1/Mpc"] = max(k_max, k_max_old)

    def set(self, params_values_dict):
        # If no output requested, remove arguments that produce an error
        # (e.g. complaints if halofit requested but no Cl's computed.)
        # Needed for facilitating post-processing
        if not self.extra_args["output"]:
            for k in ["non linear"]:
                self.extra_args.pop(k, None)
        # Prepare parameters to be passed: this-iteration + extra
        args = {self.translate_param(p): v for p, v in params_values_dict.items()}
        args.update(self.extra_args)
        # Generate and save
        self.log.debug("Setting parameters: %r", args)
        self.classy.set(**args)

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Set parameters
        self.set(params_values_dict)
        # Compute!
        try:
            self.classy.compute()
        # "Valid" failure of CLASS: parameters too extreme -> log and report
        except CosmoComputationError as e:
            if self.stop_at_error:
                self.log.error(
                    "Computation error (see traceback below)! "
                    "Parameters sent to CLASS: %r and %r.\n"
                    "To ignore this kind of error, make 'stop_at_error: False'.",
                    state["params"], dict(self.extra_args))
                raise
            else:
                self.log.debug("Computation of cosmological products failed. "
                               "Assigning 0 likelihood and going on. "
                               "The output of the CLASS error was %s" % e)
            return False
        # CLASS not correctly initialized, or input parameters not correct
        except CosmoSevereError:
            self.log.error("Serious error setting parameters or computing results. "
                           "The parameters passed were %r and %r. "
                           "See original CLASS's error traceback below.\n",
                           state["params"], self.extra_args)
            raise  # No LoggedError, so that CLASS traceback gets printed
        # Gather products
        for product, collector in self.collectors.items():
            # Special case: sigma8 needs H0, which cannot be known beforehand:
            if "sigma8" in self.collectors:
                self.collectors["sigma8"].args[0] = 8 / self.classy.h()
            method = getattr(self.classy, collector.method)
            arg_array = self.collectors[product].arg_array
            if arg_array is None:
                state[product] = method(
                    *self.collectors[product].args, **self.collectors[product].kwargs)
            elif isinstance(arg_array, int):
                state[product] = np.zeros(
                    len(self.collectors[product].args[arg_array]))
                for i, v in enumerate(self.collectors[product].args[arg_array]):
                    args = (list(self.collectors[product].args[:arg_array]) + [v] +
                            list(self.collectors[product].args[arg_array + 1:]))
                    state[product][i] = method(
                        *args, **self.collectors[product].kwargs)
            elif arg_array in self.collectors[product].kwargs:
                value = np.atleast_1d(self.collectors[product].kwargs[arg_array])
                state[product] = np.zeros(value.shape)
                for i, v in enumerate(value):
                    kwargs = deepcopy(self.collectors[product].kwargs)
                    kwargs[arg_array] = v
                    state[product][i] = method(
                        *self.collectors[product].args, **kwargs)
            if collector.post:
                state[product] = collector.post(*state[product])
        # Prepare derived parameters
        d, d_extra = self._get_derived_all(derived_requested=want_derived)
        if want_derived:
            state["derived"] = {p: d.get(p) for p in self.output_params}
            # Prepare necessary extra derived parameters
        state["derived_extra"] = deepcopy(d_extra)

    def _get_derived_all(self, derived_requested=True):
        """
        Returns a dictionary of derived parameters with their values,
        using the *current* state (i.e. it should only be called from
        the ``compute`` method).

        Parameter names are returned in CLASS nomenclature.

        To get a parameter *from a likelihood* use `get_param` instead.
        """
        # Put all parameters in CLASS nomenclature (self.derived_extra already is)
        requested = [self.translate_param(p) for p in (
            self.output_params if derived_requested else [])]
        requested_and_extra = dict.fromkeys(set(requested).union(set(self.derived_extra)))
        # Parameters with their own getters
        if "rs_drag" in requested_and_extra:
            requested_and_extra["rs_drag"] = self.classy.rs_drag()
        if "Omega_nu" in requested_and_extra:
            requested_and_extra["Omega_nu"] = self.classy.Omega_nu
        if "T_cmb" in requested_and_extra:
            requested_and_extra["T_cmb"] = self.classy.T_cmb()
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

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        try:
            cls = deepcopy(self._current_state["Cl"])
        except:
            raise LoggedError(
                self.log,
                "No Cl's were computed. Are you sure that you have requested them?")
        # unit conversion and ell_factor
        ells_factor = ((cls["ell"] + 1) * cls["ell"] / (2 * np.pi))[
                      2:] if ell_factor else 1
        units_factor = self._cmb_unit_factor(
            units, self._current_state['derived_extra']['T_cmb'])

        for cl in cls:
            if cl not in ['pp', 'ell']:
                cls[cl][2:] *= units_factor ** 2 * ells_factor
        if "pp" in cls and ell_factor:
            cls['pp'][2:] *= ells_factor ** 2 * (2 * np.pi)
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
        values = np.array(deepcopy(self._current_state[quantity]))
        if quantity == "comoving_radial_distance":
            values = values[0]
        return values[i_kwarg_z]

    def close(self):
        self.classy.empty()

    def get_can_provide_params(self):
        names = ['Omega_Lambda', 'Omega_cdm', 'Omega_b', 'Omega_m', 'rs_drag', 'z_reio',
                 'YHe', 'Omega_k', 'age', 'sigma8']
        for name, mapped in self.renames.items():
            if mapped in names:
                names.append(name)
        return names

    def get_version(self):
        return getattr(self.classy_module, '__version__', None)

    # Installation routines

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(os.path.join(path, "code", cls.__name__))

    @classmethod
    def get_import_path(cls, path):
        log = logging.getLogger(cls.__name__)
        classy_build_path = os.path.join(path, "python", "build")
        if not os.path.isdir(classy_build_path):
            log.error("Either CLASS is not in the given folder, "
                      "'%s', or you have not compiled it.", path)
            return None
        py_version = "%d.%d" % (sys.version_info.major, sys.version_info.minor)
        try:
            post = next(d for d in os.listdir(classy_build_path)
                        if (d.startswith("lib.") and py_version in d))
        except StopIteration:
            log.error("The CLASS installation at '%s' has not been compiled for the "
                      "current Python version.", path)
            return None
        return os.path.join(classy_build_path, post)

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
            log.info("Importing *local* CLASS from '%s'.", path)
            if not os.path.exists(path):
                log.error("The given folder does not exist: '%s'", path)
                return False
            classy_build_path = cls.get_import_path(path)
            if not classy_build_path:
                return False
        elif not path:
            log.info("Importing *global* CLASS.")
            classy_build_path = None
        else:
            log.info("Importing *auto-installed* CLASS (but defaulting to *global*).")
            classy_build_path = cls.get_import_path(path)
        try:
            return load_module(
                'classy', path=classy_build_path, min_version=cls._classy_repo_version)
        except ImportError:
            if path is not None and path.lower() != "global":
                log.error("Couldn't find the CLASS python interface at '%s'. "
                          "Are you sure it has been installed there?", path)
            else:
                log.error("Could not import global CLASS installation. "
                          "Specify a Cobaya or CLASS installation path, "
                          "or install the CLASS Python interface globally with "
                          "'cd /path/to/class/python/ ; python setup.py install'")
            return False
        except VersionCheckError as e:
            log.error(str(e))
            return False

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
            os.path.join(path, "code"), cls._classy_repo_name, cls._classy_repo_version,
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
