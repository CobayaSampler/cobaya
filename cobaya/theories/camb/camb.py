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
   A. Lewis, A. Challinor, A. Lasenby,
   *Efficient computation of CMB anisotropies in closed FRW*
   (`arXiv:astro-ph/9911177 <https://arxiv.org/abs/astro-ph/9911177>`_)
   |br|
   C. Howlett, A. Lewis, A. Hall, A. Challinor,
   *CMB power spectrum parameter degeneracies in the era of precision cosmology*
   (`arXiv:1201.3654 <https://arxiv.org/abs/1201.3654>`_)


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


If you want to use your own version of CAMB, you need to specify its location with a
``path`` option inside the ``classy`` block. If you do not specify a ``path``,
CAMB will be loaded from the automatic-install ``modules`` folder, if specified, or
otherwise imported as a globally-installed Python package. Cobaya will print at
initialisation where it is getting CAMB from.

.. _camb_modify:

Modifying CAMB
^^^^^^^^^^^^^^

If you modify CAMB and add new variables, make sure that the variables you create are
exposed in the Python interface (`instructions here
<https://camb.readthedocs.io/en/latest/model.html#camb.model.CAMBparams>`__).
If you follow those instructions you do not need to make any additional modification in
Cobaya.

You can use the :doc:`model wrapper <cosmo_model>` to test your modification by
evaluating observables or getting derived quantities at known points in the parameter
space (set ``debug: True`` to get more detailed information of what exactly is passed to
CLASS).

In your CAMB modification, remember that you can raise a ``CAMBParamRangeError`` or a
``CAMBError`` whenever the computation of any observable would fail, but you do not
expect that observable to be compatible with the data (e.g. at the fringes of the
parameter space). Whenever such an error is raised during sampling, the likelihood is
assumed to be zero, and the run is not interrupted.


Installation
------------

Pre-requisites
^^^^^^^^^^^^^^

**cobaya** calls CAMB using its Python interface, which requires that you compile CAMB
using intel's ifort compiler or the GNU gfortran compiler version 6.4 or later.
To check if you have the latter, type ``gfortran --version`` in the shell,
and the first line should look like

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

# Local
from cobaya.theories._cosmo import _cosmo
from cobaya.log import LoggedError
from cobaya.install import download_github_release, check_gcc_version
from cobaya.conventions import _c_km_s, _T_CMB_K
from cobaya.tools import deepcopy_where_possible

# Result collector
collector = namedtuple("collector", ["method", "args", "kwargs"])
collector.__new__.__defaults__ = (None, [], {})


class camb(_cosmo):
    # Name of the Class repo/folder and version to download
    camb_repo_name = "cmbant/CAMB"
    camb_repo_version = "master"
    camb_min_gcc_version = "6.4"

    def initialize(self):
        """Importing CAMB from the correct path, if given."""
        if not self.path and self.path_install:
            self.path = self.get_path(self.path_install)
        if self.path and not os.path.exists(self.path):
            # Fail if this was a directly specified path,
            # or ignore and try to global-import if it came from a path_install
            if self.path_install:
                self.log.info("*local* CAMB not found at " + self.path)
                self.log.info("Importing *global* CAMB.")
            else:
                raise LoggedError(self.log, "*local* CAMB not found at " + self.path)
        elif self.path:
            self.log.info("Importing *local* CAMB from " + self.path)
            if not os.path.exists(self.path):
                raise LoggedError(
                    self.log, "The given folder does not exist: '%s'", self.path)
            pycamb_path = self.path
            if not os.path.exists(os.path.join(self.path, "setup.py")):
                raise LoggedError(
                    self.log,
                    "Either CAMB is not in the given folder, '%s', or you are using a "
                    "very old version without the Python interface.", self.path)
            sys.path.insert(0, pycamb_path)
        else:
            self.log.info("Importing *global* CAMB.")
        try:
            import camb
        except ImportError:
            raise LoggedError(
                self.log, "Couldn't find the CAMB python interface.\n"
                          "Make sure that you have compiled it, and that you either\n"
                          " (a) specify a path (you didn't) or\n"
                          " (b) install the Python interface globally with\n"
                          "     'pip install -e /path/to/camb [--user]'")
        self.camb = camb
        # Generate states, to avoid recomputing
        self._n_states = 3
        self._states = [{"params": None, "derived": None, "derived_extra": None, "last": 0}
                        for i in range(self._n_states)]
        # Dict of named tuples to collect requirements and computation methods
        self.collectors = {}
        # Additional input parameters to pass to CAMB, and attributes to set_ manually
        self.extra_args = deepcopy_where_possible(self.extra_args) or {}
        self.extra_attrs = {}
        # Patch: if cosmomc_theta is used, CAMB needs to be passed explicitly "H0=None"
        # This is *not* going to change on CAMB's side.
        if all((p in self.input_params) for p in ["H0", "cosmomc_theta"]):
            raise LoggedError(self.log, "Can't pass both H0 and cosmomc_theta to CAMB.")
        if "cosmomc_theta" in self.input_params:
            self.extra_args["H0"] = None
        # Set aliases
        self.planck_to_camb = self.renames
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []
        # Some default settings
        self.needs_perts = False
        self.limber = False
        self.non_linear_lens = False
        self.non_linear_pk = False

    ###     # TODO: This will hopefully be fixed later
    ###        self.extra_attrs["Want_CMB"] = False

    def current_state(self):
        lasts = [self._states[i]["last"] for i in range(self._n_states)]
        return self._states[lasts.index(max(lasts))]

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
            if k == "Cl":
                self.extra_args["lmax"] = max(
                    max(v.values()), self.extra_args.get("lmax", 0))
                cls = [a.lower() for a in v]
                self.collectors[k] = collector(
                    method="CAMBdata.get_cmb_power_spectra",
                    kwargs={
                        "spectra": list(set(
                            (self.collectors[k].kwargs.get("spectra", [])
                             if k in self.collectors else []) +
                            ["total"] + (["lens_potential"] if "pp" in cls else []))),
                        "raw_cl": True})
                self.needs_perts = True
                self.extra_attrs["Want_CMB"] = True
                self.non_linear_lens = True
            elif k == "H":
                self.collectors[k] = collector(
                    method="CAMBdata.h_of_z",
                    kwargs={"z": np.atleast_1d(v["z"])})
                self.H_units_conv_factor = {"1/Mpc": 1, "km/s/Mpc": _c_km_s}
            elif k == "angular_diameter_distance":
                self.collectors[k] = collector(
                    method="CAMBdata.angular_diameter_distance",
                    kwargs={"z": np.atleast_1d(v["z"])})
            elif k == "comoving_radial_distance":
                self.collectors[k] = collector(
                    method="CAMBdata.comoving_radial_distance",
                    kwargs={"z": np.atleast_1d(v["z"])})
            elif k == "fsigma8":
                self.add_to_redshifts(v["z"])
                self.collectors[k] = collector(
                    method="CAMBdata.get_fsigma8",
                    kwargs={})
                self.needs_perts = True
            elif k == "Pk_interpolator":
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
            elif k == "source_Cl":
                if not getattr(self, "sources", None):
                    self.sources = odict()
                for source, window in v["sources"].items():
                    # If it was already there, _cosmo.needs() has already checked that
                    # old info == new info
                    if source not in self.sources:
                        self.sources[source] = window
                self.limber = self.limber or v.get("limber", False)
                self.non_linear_pk = self.non_linear_pk or v.get("non_linear", False)
                self.non_linear_lens = self.non_linear_lens or v.get("non_linear", False)
                if "lmax" in v:
                    self.extra_args["lmax"] = max(v["lmax"], self.extra_args.get("lmax", 0))
                self.needs_perts = True
                # Create collector
                self.collectors[k] = collector(method="CAMBdata.get_source_cls_dict")
            # General derived parameters
            elif v is None:
                k_translated = self.translate_param(k)
                if k_translated not in self.derived_extra:
                    self.derived_extra += [k_translated]
                if k == "sigma8":
                    self.extra_attrs["WantTransfer"] = True
                    self.needs_perts = True
            else:
                raise LoggedError(self.log, "This should not be happening. Contact the developers.")
        # Check that there are no repeated parameters between input and extra
        if set(self.input_params).intersection(set(self.extra_args)):
            raise LoggedError(
                self.log,
                "The following parameters appear both as input parameters and as CAMB "
                "extra arguments: %s. Please, remove one of the definitions of each.",
                list(set(self.input_params).intersection(set(self.extra_args))))
        # Remove extra args that cause an error if the associated product is not requested
        if "Cl" not in self._needs:
            self.extra_args.pop("lens_potential_accuracy", None)
        # Computing non-linear corrections
        from camb import model
        self.extra_attrs["NonLinear"] = {
            (True, True): model.NonLinear_both,
            (True, False): model.NonLinear_lens,
            (False, True): model.NonLinear_pk,
            (False, False): False}[(self.non_linear_lens, self.non_linear_pk)]

    def add_to_redshifts(self, z):
        self.extra_args["redshifts"] = np.sort(np.unique(np.concatenate(
            (np.atleast_1d(z), self.extra_args.get("redshifts", [])))))[::-1]

    def translate_param(self, p):
        if self.use_planck_names:
            return self.planck_to_camb.get(p, p)
        return p

    def set(self, params_values_dict, i_state):
        # Store them, to use them later to identify the state
        self._states[i_state]["params"] = deepcopy(params_values_dict)
        # Prepare parameters to be passed: this-iteration + extra
        args = {self.translate_param(p): v for p, v in params_values_dict.items()}
        args.update(self.extra_args)
        self._states[i_state]["set_args"] = deepcopy(args)
        # Generate and save
        self.log.debug("Setting parameters: %r", args)
        try:
            cambparams = self.camb.set_params(**args)
            if self.extra_attrs:
                self.log.debug("Setting attributes of CAMBParams: %r", self.extra_attrs)
            for attr, value in self.extra_attrs.items():
                if hasattr(cambparams, attr):
                    setattr(cambparams, attr, value)
                else:
                    raise LoggedError(
                        self.log, "Some of the attributes to be set manually were not "
                                  "recognized: %s=%s", attr, value)
            # Sources
            if getattr(self, "sources", None):
                from camb.sources import GaussianSourceWindow, SplinedSourceWindow
                self.log.debug("Setting sources: %r", self.sources)
                SourceWindows = []
                for source, window in self.sources.items():
                    function = window.pop("function", None)
                    if function == "spline":
                        SourceWindows.append(SplinedSourceWindow(**window))
                    elif function == "gaussian":
                        SourceWindows.append(GaussianSourceWindow(**window))
                    else:
                        raise LoggedError(self.log, "Unknown source window function type %r", function)
                    window["function"] = function
                cambparams.SourceWindows = SourceWindows
                cambparams.SourceTerms.limber_windows = self.limber
            return cambparams
        except self.camb.baseconfig.CAMBParamRangeError:
            if self.stop_at_error:
                raise LoggedError(self.log, "Out of bound parameters: %r", params_values_dict)
            else:
                self.log.debug("Out of bounds parameters. "
                               "Assigning 0 likelihood and going on.")
        except self.camb.baseconfig.CAMBValueError:
            self.log.error(
                "Error setting parameters (see traceback below)! "
                "Parameters sent to CAMB: %r and %r.\n"
                "To ignore this kind of errors, make 'stop_at_error: False'.",
                self._states[i_state]["params"], self.extra_args)
            raise
        except self.camb.baseconfig.CAMBUnknownArgumentError as e:
            raise LoggedError(
                self.log,
                "Some of the parameters passed to CAMB were not recognized: %s" % str(e))
        return False

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
                _derived.update(self._states[i_state]["derived"] or {})
            self.log.debug("Re-using computed results (state %d)", i_state)
        except StopIteration:
            reused_state = False
            # update the (first) oldest one and compute
            i_state = lasts.index(min(lasts))
            self.log.debug("Computing (state %d)", i_state)
            if self.timer:
                self.timer.start()
            # Set parameters
            result = self.set(params_values_dict, i_state)
            # Failed to set parameters but no error raised
            # (e.g. out of computationally feasible range): lik=0
            if not result:
                return 0
            intermediates = {
                "CAMBparams": {"result": result},
                "CAMBdata": {"method": "get_results" if self.needs_perts else "get_background",
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
                    self._states[i_state][product] = method(
                        *self.collectors[product].args, **self.collectors[product].kwargs)
                except self.camb.baseconfig.CAMBError:
                    if self.stop_at_error:
                        self.log.error(
                            "Computation error (see traceback below)! "
                            "Parameters sent to CAMB: %r and %r.\n"
                            "To ignore this kind of errors, make 'stop_at_error: False'.",
                            self._states[i_state]["params"], self.extra_args)
                        raise
                    else:
                        # Assumed to be a "parameter out of range" error.
                        return 0
            # Prepare derived parameters
            if _derived == {}:
                _derived.update(self._get_derived_all(intermediates))
                self._states[i_state]["derived"] = odict([[p, _derived[p]] for p in self.output_params])
            # Prepare necessary extra derived parameters
            self._states[i_state]["derived_extra"] = {
                p: self._get_derived(p, intermediates) for p in self.derived_extra}
            if self.timer:
                self.timer.increment(self.log)
        # make this one the current one by decreasing the antiquity of the rest
        for i in range(self._n_states):
            self._states[i]["last"] -= max(lasts)
        self._states[i_state]["last"] = 1
        return 1 if reused_state else 2

    def _get_derived_from_params(self, p, intermediates):
        for origin in ["CAMBdata", "CAMBparams"]:
            result = intermediates[origin].get("result")
            if result is None:
                continue
            for thing in [result, getattr(result, "Params", {})]:
                try:
                    return getattr(thing, p)
                except AttributeError:
                    for mod in ["InitPower", "Reion", "Recomb", "Transfer", "DarkEnergy"]:
                        try:
                            return getattr(getattr(thing, mod), p)
                        except AttributeError:
                            pass
        return None

    def _get_derived_from_std(self, p, intermediates):
        dic = intermediates["CAMBdata"].get("derived_dic", None)
        if dic is None:
            result = intermediates["CAMBdata"].get("result", None)
            if result is None:
                return None
            dic = result.get_derived_params()
            intermediates["CAMBdata"]["derived_dic"] = dic
        return dic.get(p, None)

    def _get_derived_from_getter(self, p, intermediates):
        return getattr(intermediates["CAMBparams"]["result"], "get_" + p, lambda: None)()

    def _get_derived(self, p, intermediates):
        """
        General function to extract a single derived parameter.

        To get a parameter *from a likelihood* use `get_param` instead.
        """
        # Specific calls, if general ones fail:
        if p == "sigma8":
            return intermediates["CAMBdata"]["result"].get_sigma8()[-1]
        for f in [self._get_derived_from_params,
                  self._get_derived_from_std,
                  self._get_derived_from_getter]:
            derived = f(p, intermediates)
            if derived is not None:
                return derived

    def _get_derived_all(self, intermediates):
        """
        Returns a dictionary of derived parameters with their values,
        using the *current* state (i.e. it should only be called from
        the ``compute`` method).

        To get a parameter *from a likelihood* use `get_param` instead.
        """
        derived = {}
        for p in self.output_params:
            derived[p] = self._get_derived(self.translate_param(p), intermediates)
            if derived[p] is None:
                raise LoggedError(
                    self.log, "Derived param '%s' not implemented in the CAMB interface", p)
        return derived

    def get_param(self, p):
        current_state = self.current_state()
        for pool in ["params", "derived", "derived_extra"]:
            value = deepcopy(
                (current_state[pool] or {}).get(self.translate_param(p), None))
            if value is not None:
                return value
        raise LoggedError(self.log, "Parameter not known: '%s'", p)

    def get_Cl(self, ell_factor=False, units="muK2"):
        current_state = self.current_state()
        # get C_l^XX from the cosmological code
        try:
            cl_camb = deepcopy(current_state["Cl"]["total"])
        except:
            raise LoggedError(
                self.log, "No Cl's were computed. Are you sure that you have requested them?")
        mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3}
        cls = {"ell": np.arange(cl_camb.shape[0])}
        cls.update({sp: cl_camb[:, i] for sp, i in mapping.items()})
        if "lens_potential" in current_state["Cl"]:
            cls.update({"pp": deepcopy(current_state["Cl"]["lens_potential"])[:, 0]})
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
            return self._get_z_dependent("H", z) * self.H_units_conv_factor[units]
        except KeyError:
            raise LoggedError(
                self.log, "Units not known for H: '%s'. Try instead one of %r.",
                units, list(self.H_units_conv_factor))

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

    def get_source_Cl(self):
        current_state = self.current_state()
        # get C_l^XX from the cosmological code
        try:
            cls = deepcopy(current_state["source_Cl"])
        except:
            raise LoggedError(
                self.log, "No source Cl's were computed. "
                          "Are you sure that you have requested some source?")
        cls_dict = dict()
        for term, cl in cls.items():
            term_tuple = tuple(
                [(lambda x: x if x == "P" else list(self.sources)[int(x) - 1])(_.strip("W"))
                 for _ in term.split("x")])
            cls_dict[term_tuple] = cl
        cls_dict["ell"] = np.arange(cls[list(cls)[0]].shape[0])
        return cls_dict

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(os.path.join(path, "code", cls.camb_repo_name[cls.camb_repo_name.find("/") + 1:]))

    @classmethod
    def is_installed(cls, **kwargs):
        import platform
        if not kwargs["code"]:
            return True
        return os.path.isfile(os.path.realpath(
            os.path.join(cls.get_path(kwargs["path"]),
                         "camb", "cambdll.dll" if (platform.system() == "Windows") else "camblib.so")))

    @classmethod
    def install(cls, path=None, force=False, code=True, data=False, no_progress_bars=False, **kwargs):
        log = logging.getLogger(cls.__name__)
        if not code:
            log.info("Code not requested. Nothing to do.")
            return True
        log.info("Downloading camb...")
        success = download_github_release(
            os.path.join(path, "code"), cls.camb_repo_name, cls.camb_repo_version,
            no_progress_bars=no_progress_bars, logger=log)
        if not success:
            log.error("Could not download camb.")
            return False
        camb_path = cls.get_path(path)
        log.info("Compiling camb...")
        from subprocess import Popen, PIPE
        process_make = Popen([sys.executable, "setup.py", "build_cluster"],
                             cwd=camb_path, stdout=PIPE, stderr=PIPE)
        out, err = process_make.communicate()
        if process_make.returncode:
            log.info(out)
            log.info(err)
            gcc_check = check_gcc_version(cls.camb_min_gcc_version, error_returns=False)
            if not gcc_check:
                cause = (" Possible cause: it looks like `gcc` does not have the correct "
                         "version number (CAMB requires %s); and `ifort` is also probably "
                         "not available.", cls.camb_min_gcc_version)
            else:
                cause = ""
            log.error("Compilation failed!" + cause)
            return False
        return True
