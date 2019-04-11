"""
.. module:: theories.camb

:Synopsis: Managing the CAMB cosmological code
:Author: Jesus Torrado and Antony Lewis

.. |br| raw:: html

   <br />

This module imports and manages the CAMB cosmological code.
It requires CAMB 1.0 or higher (for compatibility with older versions, you can temporarily
use cobaya 1.0.4, but update asap, since that version is not maintained any more).

.. note::

   **If you use this cosmological code, please cite it as:**
   |br|
   `A. Lewis, A. Challinor, A. Lasenby, "Efficient computation of CMB anisotropies in closed FRW"
   (arXiv:astro-ph/9911177) <https://arxiv.org/abs/astro-ph/9911177>`_
   |br|
   `C. Howlett, A. Lewis, A. Hall, A. Challinor, "CMB power spectrum parameter degeneracies in the era of precision cosmology"
   (arXiv:1201.3654) <https://arxiv.org/abs/1201.3654>`_


Usage
-----

If you are using a likelihood that requires some observable from CAMB, simply add CAMB
to the theory block.

You can specify any parameter that CAMB understands in the ``params`` block:

.. code-block:: yaml

   theory:
     camb:
       extra_args:
         [any param that CAMB understands, for FIXED and PRECISION]

   params:
       [any param that CAMB understands, fixed, sampled or derived]


Installation
------------

Pre-requisites
^^^^^^^^^^^^^^

**cobaya** calls CAMB using its Python interface, which requires that you compile CAMB
using intel's ifort compiler or the GNU gfortran compiler version 6.4 or later. To check if you have the latter,
type ``gfortran --version`` in the shell, and the first line should look like

.. code::

   GNU Fortran ([your OS version]) [gfortran version] [release date]

Check that ``[gfortran's version]`` is at least 6.4. If you get an error instead, you need
to install gfortran (contact your local IT service).

CAMB comes with binaries pre-built for Windows, so if you don't need to modify the CAMB
source code, no Fortran compiler is needed.


Automatic installation
^^^^^^^^^^^^^^^^^^^^^^

If you do not plan to modify CAMB, the easiest way to install it is using the
:doc:`automatic installation script <installation_cosmo>`. Just make sure that
``theory: camb:`` appears in one of the files passed as arguments to the installation
script.


Manual installation (or using your own version)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are planning to modify CAMB or use an already modified version,
you should not use the automatic installation script. Use the installation method that
best adapts to your needs:

* [**Recommended for staying up-to-date**]
  To install CAMB locally and keep it up-to-date, clone the
  `CAMB repository in Github <https://github.com/cmbant/CAMB>`_
  in some folder of your choice, say ``/path/to/theories/CAMB``:

  .. code:: bash

      $ cd /path/to/theories
      $ git clone --recursive https://github.com/cmbant/CAMB.git
      $ cd CAMB
      $ python setup.py build

  To update to the last changes in CAMB (master), run ``git pull`` from CAMB's folder and
  re-build using the last command.

* [**Recommended for modifying CAMB**]
  First, `fork the CAMB repository in Github <https://github.com/cmbant/CAMB>`_
  (follow `this instructions <https://help.github.com/articles/fork-a-repo/>`_) and then
  follow the same steps as above, substituting the second one with:

  .. code:: bash

      $ git clone --recursive https://[YourGithubUser]@github.com/[YourGithubUser]/CAMB.git

* To use your own version, assuming it's placed under ``/path/to/theories/CAMB``,
  just make sure it is compiled (and that the version on top of which you based your
  modifications is old enough to have the Python interface implemented.

In the cases above, you **must** specify the path to your CAMB installation in
the input block for CAMB (otherwise a system-wide CAMB may be used instead):

.. code:: yaml

   theory:
     camb:
       path: /path/to/theories/CAMB

.. note::

   In any of these methods, you should **not** install CAMB as python package using
   ``python setup.py install --user``, as the official instructions suggest.
   It is actually safer not to do so if you intend to switch between different versions or
   modifications of CAMB.


Modifying CAMB
--------------

If you modify CAMB and add new variables, you don't need to let **cobaya** know,
but make sure
that the variables you create are exposed in its Python interface (contact CAMB's
developers if you need help with that).

.. todo::

   Point somewhere to the CAMB documentation where how to make these modifications
   is explained.

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import sys
import os
import logging
from copy import deepcopy
import numpy as np
from collections import namedtuple, OrderedDict as odict
from time import time
from numbers import Number

# Local
from cobaya.theories._cosmo import _cosmo
from cobaya.log import HandledException
from cobaya.install import download_github_release, check_gcc_version
from cobaya.conventions import _c_km_s, _T_CMB_K

# Result collector
collector = namedtuple("collector", ["method", "args", "kwargs"])
collector.__new__.__defaults__ = (None, [], {})


class camb(_cosmo):

    def initialize(self):
        """Importing CAMB from the correct path, if given."""
        if not self.path and self.path_install:
            self.path = os.path.join(
                self.path_install, "code", camb_repo_name[camb_repo_name.find("/") + 1:])
        if self.path and not os.path.exists(self.path):
            # Fail if this was a directly specified path,
            # or ignore and try to global-import if it came from a path_install
            if self.path_install:
                self.log.info("*local* CAMB not found at " + self.path)
                self.log.info("Importing *global* CAMB.")
            else:
                self.log.error("*local* CAMB not found at " + self.path)
                raise HandledException
        elif self.path:
            self.log.info("Importing *local* CAMB from " + self.path)
            if not os.path.exists(self.path):
                self.log.error("The given folder does not exist: '%s'", self.path)
                raise HandledException
            pycamb_path = self.path
            if not os.path.exists(os.path.join(self.path, "setup.py")):
                self.log.error(
                    "Either CAMB is not in the given folder, '%s', or you are using a "
                    "very old version without the Python interface.", self.path)
                raise HandledException
            sys.path.insert(0, pycamb_path)
        else:
            self.log.info("Importing *global* CAMB.")
        try:
            import camb
        except ImportError:
            self.log.error(
                "Couldn't find the CAMB python interface.\n"
                "Make sure that you have compiled it, and that you either\n"
                " (a) specify a path (you didn't) or\n"
                " (b) install the Python interface globally with\n"
                "     'pip install -e /path/to/camb [--user]'")
            raise HandledException
        self.camb = camb
        # Prepare errors
        from camb.baseconfig import CAMBParamRangeError, CAMBError
        from camb.baseconfig import CAMBValueError, CAMBUnknownArgumentError
        global CAMBParamRangeError, CAMBError, CAMBValueError, CAMBUnknownArgumentError
        # Generate states, to avoid recomputing
        self.n_states = 3
        self.states = [{"params": None, "derived": None, "derived_extra": None, "last": 0}
                       for i in range(self.n_states)]
        # Dict of named tuples to collect requirements and computation methods
        self.collectors = {}
        # Additional input parameters to pass to CAMB, and attributes to set_ manually
        self.extra_args = self.extra_args or {}
        self.extra_attrs = {}
        # Patch: if cosmomc_theta is used, CAMB needs to be passed explicitly "H0=None"
        # This is *not* going to change on CAMB's side.
        if all((p in self.input_params) for p in ["H0", "cosmomc_theta"]):
            self.log.error("Can't pass both H0 and cosmomc_theta to Camb.")
            raise HandledException
        if "cosmomc_theta" in self.input_params:
            self.extra_args["H0"] = None
        # Set aliases
        self.planck_to_camb = self.renames
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []
        self.needs_perts = False

    def current_state(self):
        lasts = [self.states[i]["last"] for i in range(self.n_states)]
        return self.states[lasts.index(max(lasts))]

    def needs(self, **requirements):
        # Computed quantities required by the likelihood
        # Note that redshifts below are treated differently for background quantities,
        #   were no additional transfer computation is needed (e.g. H(z)),
        #   and matter-power-related quantities, that require additional computation
        #   and need the redshifts to be passed at CAMBParams instantiation.
        #   Also, we always make sure that those redshifts are sorted in descending order,
        #   since all CAMB related functions return quantities in that implicit order
        # The following super call makes sure that the requirements are properly
        # accumulated, i.e. taking the max of precision requests, etc.
        super(camb, self).needs(**requirements)
        for k, v in self._needs.items():
            # Products and other computations
            key_name = k.lower()
            if key_name == "cl":
                self.extra_args["lmax"] = max(
                    max(v.values()), self.extra_args.get("lmax", 0))
                cls = [a.lower() for a in v]
                self.collectors[key_name] = collector(
                    method="CAMBdata.get_cmb_power_spectra",
                    kwargs={
                        "spectra": list(set(
                            (self.collectors[key_name].kwargs.get("spectra", [])
                             if key_name in self.collectors else []) +
                            ["total"] + (["lens_potential"] if "pp" in cls else []))),
                        "raw_cl": True})
                self.needs_perts = True
            elif key_name == "h":
                self.collectors[key_name] = collector(
                    method="CAMBdata.h_of_z",
                    kwargs={"z": np.atleast_1d(v["z"])})
                self.H_units_conv_factor = {"1/Mpc": 1, "km/s/Mpc": _c_km_s}
            elif key_name == "angular_diameter_distance":
                self.collectors[key_name] = collector(
                    method="CAMBdata.angular_diameter_distance",
                    kwargs={"z": np.atleast_1d(v["z"])})
            elif key_name == "comoving_radial_distance":
                self.collectors[key_name] = collector(
                    method="CAMBdata.comoving_radial_distance",
                    kwargs={"z": np.atleast_1d(v["z"])})
            elif key_name == "fsigma8":
                self.add_to_redshifts(v["z"])
                self.collectors[key_name] = collector(
                    method="CAMBdata.get_fsigma8",
                    kwargs={})
                self.needs_perts = True
            elif key_name == "pk_interpolator":
                self.extra_args["kmax"] = max(v["k_max"], self.extra_args.get("kmax", 0))
                self.add_to_redshifts(v["z"])
                v["vars_pairs"] = v["vars_pairs"] or [["total", "total"]]
                kwargs = deepcopy(v)
                # change of defaults:
                kwargs["hubble_units"] = kwargs.get("hubble_units", False)
                kwargs["k_hunit"] = kwargs.get("k_hunit", False)
                for p in "k_max", "z", "vars_pairs":
                    kwargs.pop(p)
                for pair in v["vars_pairs"]:
                    name = "Pk_interpolator_%s_%s" % (pair[0], pair[1])
                    kwargs.update(dict(zip(["var1", "var2"], pair)))
                    self.collectors[name] = collector(
                        method="CAMBdata.get_matter_power_interpolator",
                        kwargs=kwargs)
                self.needs_perts = True
            elif v is None:
                k_translated = self.translate_param(k)
                if k_translated not in self.derived_extra:
                    self.derived_extra += [k_translated]
                if k == "sigma8":
                    self.extra_attrs["WantTransfer"] = True
                    self.needs_perts = True
            else:
                self.log.error("This should not be happening. Contact the developers.")
                raise HandledException
        # Finally, check that there are no repeated parameters between input and extra
        if set(self.input_params).intersection(set(self.extra_args)):
            self.log.error(
                "The following parameters appear both as input parameters and as CAMB "
                "extra arguments: %s. Please, remove one of the definitions of each.",
                list(set(self.input_params).intersection(set(self.extra_args))))
            raise HandledException

    def add_to_redshifts(self, z):
        self.extra_args["redshifts"] = np.sort(np.unique(np.concatenate(
            (np.atleast_1d(z), self.extra_args.get("redshifts", [])))))[::-1]

    def translate_param(self, p):
        if self.use_planck_names:
            return self.planck_to_camb.get(p, p)
        return p

    def set(self, params_values_dict, i_state):
        # Store them, to use them later to identify the state
        self.states[i_state]["params"] = deepcopy(params_values_dict)
        # Prepare parameters to be passed: this-iteration + extra
        args = {self.translate_param(p): v for p, v in params_values_dict.items()}
        args.update(self.extra_args)
        self.states[i_state]["set_args"] = deepcopy(args)
        # Generate and save
        self.log.debug("Setting parameters: %r", args)
        try:
            cambparams = self.camb.set_params(**args)
            for attr, value in self.extra_attrs.items():
                if hasattr(cambparams, attr):
                    setattr(cambparams, attr, value)
                else:
                    self.log.error("Some of the attributes to be set manually were not "
                                   "recognized: %s=%s", attr, value)
                    raise HandledException
            return cambparams
        except CAMBParamRangeError:
            if self.stop_at_error:
                self.log.error("Out of bound parameters: %r", params_values_dict)
                raise HandledException
            else:
                self.log.debug("Out of bounds parameters. "
                               "Assigning 0 likelihood and going on.")
        except CAMBValueError:
            self.log.error(
                "Error setting parameters (see traceback below)! "
                "Parameters sent to CAMB: %r and %r.\n"
                "To ignore this kind of errors, make 'stop_at_error: False'.",
                self.states[i_state]["params"], self.extra_args)
            raise
        except CAMBUnknownArgumentError as e:
            self.log.error("Some of the parameters passed to CAMB were not recognized: "
                           "%s", e.message)
            raise HandledException
        return False

    def compute(self, _derived=None, cached=True, **params_values_dict):
        lasts = [self.states[i]["last"] for i in range(self.n_states)]
        try:
            if not cached:
                raise StopIteration
            # are the parameter values there already?
            i_state = next(i for i in range(self.n_states)
                           if self.states[i]["params"] == params_values_dict)
            # has any new product been requested?
            for product in self.collectors:
                next(k for k in self.states[i_state] if k == product)
            reused_state = True
            # Get (pre-computed) derived parameters
            if _derived == {}:
                _derived.update(self.states[i_state]["derived"])
            self.log.debug("Re-using computed results (state %d)", i_state)
        except StopIteration:
            reused_state = False
            # update the (first) oldest one and compute
            i_state = lasts.index(min(lasts))
            self.log.debug("Computing (state %d)", i_state)
            if self.timing:
                a = time()
            # Set parameters
            result = self.set(params_values_dict, i_state)
            # Failed to set parameters but no error raised
            # (e.g. out of computationally feasible range): lik=0
            if not result:
                return 0
            intermediates = {
                "CAMBparams": {"result": result},
                "CAMBdata": {"method": "get_results" if self.needs_perts
                else "get_background",
                             "result": None, "derived_dic": None}}
            # Compute the necessary products (incl. any intermediate result, if needed)
            for product, collector in self.collectors.items():
                parent, method = collector.method.split(".")
                try:
                    if intermediates[parent]["result"] is None:
                        intermediates[parent]["result"] = getattr(
                            self.camb, intermediates[parent]["method"])(
                            intermediates["CAMBparams"]["result"])
                    method = getattr(intermediates[parent]["result"], method)
                    self.states[i_state][product] = method(
                        *self.collectors[product].args, **self.collectors[product].kwargs)
                except CAMBError:
                    if self.stop_at_error:
                        self.log.error(
                            "Computation error! Parameters sent to CAMB: %r and %r.\n"
                            "To ignore this kind of errors, make 'stop_at_error: False'.",
                            self.states[i_state]["params"], self.extra_args)
                        raise HandledException
                    else:
                        # Assumed to be a "parameter out of range" error.
                        return 0
            # Prepare derived parameters
            if _derived == {}:
                _derived.update(self.get_derived_all(intermediates))
                self.states[i_state]["derived"] = odict(
                    [[p, _derived[p]] for p in self.output_params])
            # Prepare necessary extra derived parameters
            self.states[i_state]["derived_extra"] = {
                p: self.get_derived(p, intermediates) for p in self.derived_extra}
            if self.timing:
                self.n += 1
                self.time_avg = (time() - a + self.time_avg * (self.n - 1)) / self.n
                self.log.debug("Average running time: %g seconds", self.time_avg)
        # make this one the current one by decreasing the antiquity of the rest
        for i in range(self.n_states):
            self.states[i]["last"] -= max(lasts)
        self.states[i_state]["last"] = 1
        return 1 if reused_state else 2

    def get_derived_from_params(self, p, intermediates):
        result = intermediates["CAMBdata"].get("result", None)
        if result is None: return None
        pars = result.Params
        try:
            return getattr(pars, p)
        except AttributeError:
            try:
                return getattr(result, p)
            except AttributeError:
                for mod in ["InitPower", "Reion", "Recomb", "Transfer", "DarkEnergy"]:
                    try:
                        return getattr(
                            getattr(pars, mod), p)
                    except AttributeError:
                        pass
            return None

    def get_derived_from_std(self, p, intermediates):
        dic = intermediates["CAMBdata"].get("derived_dic", None)
        if dic is None:
            result = intermediates["CAMBdata"].get("result", None)
            if result is None: return None
            dic = result.get_derived_params()
            intermediates["CAMBdata"]["derived_dic"] = dic
        return dic.get(p, None)

    def get_derived_from_getter(self, p, intermediates):
        return getattr(intermediates["CAMBparams"]["result"], "get_" + p, lambda: None)()

    def get_derived(self, p, intermediates):
        """
        General function to extract a single derived parameter.

        To get a parameter *from a likelihood* use `get_param` instead.
        """
        # Specific calls, if general ones fail:
        if p == "sigma8":
            return intermediates["CAMBdata"]["result"].get_sigma8()[0]
        for f in [self.get_derived_from_params,
                  self.get_derived_from_std,
                  self.get_derived_from_getter]:
            derived = f(p, intermediates)
            if derived is not None:
                return derived

    def get_derived_all(self, intermediates):
        """
        Returns a dictionary of derived parameters with their values,
        using the *current* state (i.e. it should only be called from
        the ``compute`` method).

        To get a parameter *from a likelihood* use `get_param` instead.
        """
        derived = {}
        for p in self.output_params:
            derived[p] = self.get_derived(self.translate_param(p), intermediates)
            if derived[p] is None:
                self.log.error(
                    "Derived param '%s' not implemented in the CAMB interface", p)
                raise HandledException
        return derived

    def get_param(self, p):
        current_state = self.current_state()
        for pool in ["params", "derived", "derived_extra"]:
            value = deepcopy(current_state[pool].get(self.translate_param(p), None))
            if value is not None:
                return value
        self.log.error("Parameter not known: '%s'", p)
        raise HandledException

    def get_cl(self, ell_factor=False, units="muK2"):
        current_state = self.current_state()
        # get C_l^XX from the cosmological code
        try:
            cl_camb = deepcopy(current_state["cl"]["total"])
        except:
            self.log.error(
                "No Cl's were computed. Are you sure that you have requested them?")
            raise HandledException
        mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3}
        cls = {"ell": np.arange(cl_camb.shape[0])}
        cls.update({sp: cl_camb[:, i] for sp, i in mapping.items()})
        if "lens_potential" in current_state["cl"]:
            cls.update({"pp": deepcopy(current_state["cl"]["lens_potential"])[:, 0]})
        # unit conversion and ell_factor
        ell_factor = ((cls["ell"] + 1) * cls["ell"] / (2 * np.pi))[2:] if ell_factor else 1
        units_factors = {"1": 1,
                         "muK2": _T_CMB_K * 1.e6,
                         "K2": _T_CMB_K}
        try:
            units_factor = units_factors[units]
        except KeyError:
            self.log.error("Units '%s' not recognized. Use one of %s.",
                           units, list(units_factors))
            raise HandledException
        for cl in cls:
            if cl not in ['pp', 'ell']:
                cls[cl][2:] *= units_factor ** 2 * ell_factor
        if 'pp' in cls and ell_factor is not 1:
            cls['pp'][2:] *= ell_factor ** 2 * (2 * np.pi)
        return cls

    def _get_z_dependent(self, quantity, z):
        if quantity == "fsigma8":
            computed_redshifts = self.extra_args["redshifts"]
        else:
            z_name = next(k for k in ["redshifts", "z"]
                          if k in self.collectors[quantity].kwargs)
            computed_redshifts = self.collectors[quantity].kwargs[z_name]
        i_kwarg_z = np.concatenate(
            [np.where(computed_redshifts == zi)[0] for zi in np.atleast_1d(z)])
        return np.array(deepcopy(self.current_state()[quantity]))[i_kwarg_z]

    def get_H(self, z, units="km/s/Mpc"):
        try:
            return self._get_z_dependent("h", z) * self.H_units_conv_factor[units]
        except KeyError:
            self.log.error("Units not known for H: '%s'. Try instead one of %r.",
                           units, list(self.H_units_conv_factor))
            raise HandledException

    def get_angular_diameter_distance(self, z):
        return self._get_z_dependent("angular_diameter_distance", z)

    def get_comoving_radial_distance(self, z):
        return self._get_z_dependent("comoving_radial_distance", z)

    def get_fsigma8(self, z):
        return self._get_z_dependent("fsigma8", z)

    def get_Pk_interpolator(self):
        current_state = self.current_state()
        prefix = "Pk_interpolator_"
        return {k[len(prefix):]: deepcopy(v)
                for k, v in current_state.items() if k.startswith(prefix)}


# Installation routines ##################################################################

# Name of the Class repo/folder and version to download
camb_repo_name = "cmbant/CAMB"
camb_repo_version = "master"
camb_min_gcc_version = "6.4"


def is_installed(**kwargs):
    import platform
    if not kwargs["code"]:
        return True
    return os.path.isfile(os.path.realpath(
        os.path.join(
            kwargs["path"], "code", camb_repo_name[camb_repo_name.find("/") + 1:],
            "camb", "cambdll.dll" if (platform.system() == "Windows") else "camblib.so")))


def install(path=None, force=False, code=True, no_progress_bars=False, **kwargs):
    log = logging.getLogger(__name__.split(".")[-1])
    if not code:
        log.info("Code not requested. Nothing to do.")
        return True
    gcc_check = check_gcc_version(camb_min_gcc_version, error_returns=-1)
    if gcc_check == -1:
        log.warn("Failed to get gcc version (maybe not using gcc?). "
                 "Going ahead and hoping for the best.")
    elif not gcc_check:
        log.error("CAMB requires a gcc version >= %s, "
                  "which is higher than your current one.", camb_min_gcc_version)
        raise HandledException
    log.info("Downloading camb...")
    success = download_github_release(
        os.path.join(path, "code"), camb_repo_name, camb_repo_version,
        no_progress_bars=no_progress_bars)
    if not success:
        log.error("Could not download camb.")
        return False
    camb_path = os.path.join(path, "code", camb_repo_name[camb_repo_name.find("/") + 1:])
    log.info("Compiling camb...")
    from subprocess import Popen, PIPE
    process_make = Popen([sys.executable, "setup.py", "build_cluster"],
                         cwd=camb_path, stdout=PIPE, stderr=PIPE)
    out, err = process_make.communicate()
    if process_make.returncode:
        log.info(out)
        log.info(err)
        log.error("Compilation failed!")
        return False
    return True
