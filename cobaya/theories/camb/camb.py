"""
.. module:: theories.camb

:Synopsis: Managing the CAMB cosmological code
:Author: Jesus Torrado and Antony Lewis

.. |br| raw:: html

   <br />

This module imports and manages the CAMB cosmological code.
It requires CAMB 1.5 or higher.

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
``path`` option inside the ``camb`` block. If you do not specify a ``path``,
CAMB will be loaded from the automatic-install ``packages_path`` folder, if specified, or
otherwise imported as a globally-installed Python package. If you want to force that
the global ``camb`` installation is used, pass ``path='global'``. Cobaya will print at
initialisation where CAMB was actually loaded from.


.. _camb_access:

Access to CAMB computation products
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can retrieve CAMB computation products within likelihoods (or other pipeline
components in general) or manually from a :class:`~model.Model` as long as you have added
them as requisites; see :doc:`cosmo_external_likelihood` or
:doc:`cosmo_external_likelihood_class` for the likelihood case, and :doc:`cosmo_model` for
the manual case.

The products that you can request and later retrieve are listed in
:func:`~theories.cosmo.BoltzmannBase.must_provide`.

If you would like to access a CAMB result that is not accessible that way, you can access
the full CAMB results object
`CAMBdata <https://camb.readthedocs.io/en/latest/results.html#camb.results.CAMBdata>`__
directly by adding ``{"CAMBdata": None}`` to your requisites, and then retrieving it with
``provider.get_CAMBdata()``.

In general, the use of ``CAMBdata`` should be avoided in public code, since it breaks
compatibility with other Boltzmann codes at the likelihood interface level. If you need
a quantity for a public code that is not generally interfaced in
:func:`~theories.cosmo.BoltzmannBase.must_provide`, let us know if you think it makes
sense to add it.


.. _camb_modify:

Modifying CAMB
^^^^^^^^^^^^^^

If you modify CAMB and add new variables, make sure that the variables you create are
exposed in the Python interface (`instructions here
<https://camb.readthedocs.io/en/latest/model.html#camb.model.CAMBparams>`__).
If you follow those instructions you do not need to make any additional modification in
Cobaya.

If your modification involves new computed quantities, add a retrieving method to
`CAMBdata <https://camb.readthedocs.io/en/latest/results.html#camb.results.CAMBdata>`__,
and see :ref:`camb_access`.

You can use the :doc:`model wrapper <cosmo_model>` to test your modification by
evaluating observables or getting derived quantities at known points in the parameter
space (set ``debug: True`` to get more detailed information of what exactly is passed to
CAMB).

In your CAMB modification, remember that you can raise a ``CAMBParamRangeError`` or a
``CAMBError`` whenever the computation of any observable would fail, but you do not
expect that observable to be compatible with the data (e.g. at the fringes of the
parameter space). Whenever such an error is raised during sampling, the likelihood is
assumed to be zero, and the run is not interrupted.

.. note::

   If your modified CAMB has a lower version number than the minimum required by Cobaya,
   you will get an error at initialisation. You may still be able to use it by setting the
   option ``ignore_obsolete: True`` in the ``camb`` block (though you would be doing that
   at your own risk; ideally you should translate your modification to a newer CAMB
   version, in case there have been important fixes since the release of your baseline
   version).


Installation
------------

If you are not intending to modify CAMB, you can just use:

.. code:: bash

     $ pip install camb

 (or uv equivalent). This includes pre-built binary wheels for all platforms.

Pre-requisites for source build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**cobaya** calls CAMB using its Python interface, which requires that you compile CAMB
using intel's ifort compiler or the GNU gfortran compiler.
To check if you have the latter, type ``gfortran --version`` in the shell.

Automatic installation
^^^^^^^^^^^^^^^^^^^^^^

The :doc:`automatic installation script <installation_cosmo>` will download and build
CAMB for you (which requires a fortran compiler). Just make sure that
``theory: camb:`` appears in one of the files passed as arguments to the installation
script.

This is not neccessary if you have a pip installed camb globally, however
a source build potentially allows you to optimize it for your specific architecture.


Manual installation (or using your own version)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are planning to modify CAMB or use an already modified version,
you should not use the automatic installation script. Use the installation method that
best adapts to your needs:

* [**Recommended for staying up-to-date**]
  To install CAMB locally and keep it up-to-date, clone the
  `CAMB repository in GitHub <https://github.com/cmbant/CAMB>`_
  in some folder of your choice, say ``/path/to/theories/CAMB``:

  .. code:: bash

      $ cd /path/to/theories
      $ git clone --recursive https://github.com/cmbant/CAMB.git
      $ cd CAMB
      $ python setup.py build

  To update to the last changes in CAMB (master), run ``git pull`` from CAMB's folder and
  re-build using the last command. If you do not want to use multiple versions of CAMB,
  you can also make your local installation available to python generally by installing
  it using

.. code:: bash

     $ python -m pip install -e /path/to/CAMB

* [**Recommended for modifying CAMB**]
  First, `fork the CAMB repository in GitHub <https://github.com/cmbant/CAMB>`_
  (follow `these instructions <https://help.github.com/articles/fork-a-repo/>`_) and then
  follow the same steps as above, substituting the second one with:

  .. code:: bash

      $ git clone --recursive https://[YourGithubUser]@github.com/[YourGithubUser]/CAMB.git

* To use your own version, assuming it's placed under ``/path/to/theories/CAMB``,
  just make sure it is compiled.

In the cases above, you **must** specify the path to your CAMB installation in
the input block for CAMB (otherwise a system-wide CAMB may be used instead):

.. code:: yaml

   theory:
     camb:
       path: /path/to/theories/CAMB

.. note::

   In any of these methods, if you intend to switch between different versions or
   modifications of CAMB you should not install CAMB as python package using
   ``pip install``, as the official instructions suggest. It is not necessary
   if you indicate the path to your preferred installation as explained above.
"""

import ctypes
import numbers
import os
import platform
import sys
from collections.abc import Callable
from copy import deepcopy
from itertools import chain
from typing import Any, NamedTuple

import numpy as np

from cobaya.component import ComponentNotInstalledError, load_external_module
from cobaya.install import check_gcc_version, download_github_release, pip_install
from cobaya.log import LoggedError, get_logger
from cobaya.theories.cosmo import BoltzmannBase
from cobaya.theory import HelperTheory
from cobaya.tools import (
    Pool1D,
    Pool2D,
    PoolND,
    VersionCheckError,
    check_module_version,
    get_class_methods,
    get_properties,
    getfullargspec,
    str_to_list,
)
from cobaya.typing import InfoDict, empty_dict


# Result collector
class Collector(NamedTuple):
    method: Callable
    args: list = []
    kwargs: dict = {}
    z_pool: PoolND | None = None
    post: Callable | None = None


class CAMBOutputs(NamedTuple):
    camb_params: Any
    results: Any
    derived: dict


class CAMB(BoltzmannBase):
    r"""
    CAMB cosmological Boltzmann code \cite{Lewis:1999bs,Howlett:2012mh}.
    """

    # Name of the Class repo/folder and version to download
    _camb_repo_name = "cmbant/CAMB"
    _camb_repo_version = os.environ.get("CAMB_REPO_VERSION", "master")
    _camb_min_gcc_version = "6.4"
    _min_camb_version = "1.5.0"

    file_base_name = "camb"
    external_primordial_pk: bool
    camb: Any
    ignore_obsolete: bool

    def initialize(self):
        """Importing CAMB from the correct path, if given."""
        try:
            install_path = (
                self.get_path(self.packages_path) if self.packages_path else None
            )
            min_version = None if self.ignore_obsolete else self._min_camb_version
            self.camb = load_external_module(
                "camb",
                path=self.path,
                install_path=install_path,
                min_version=min_version,
                get_import_path=self.get_import_path,
                logger=self.log,
                not_installed_level="debug",
            )
        except VersionCheckError as excpt:
            raise VersionCheckError(
                str(excpt) + " If you are using CAMB unmodified, upgrade with"
                "`cobaya-install camb --upgrade`. "
                "If you are using a modified CAMB, "
                "set the option `ignore_obsolete: True` for CAMB."
            )
        except ComponentNotInstalledError as excpt:
            raise ComponentNotInstalledError(
                self.log,
                (
                    f"Could not find CAMB: {excpt}. "
                    "To install it, run `cobaya-install camb`"
                ),
            )
        super().initialize()
        self.extra_attrs = {
            "Want_CMB": False,
            "Want_cl_2D_array": False,
            "WantCls": False,
        }
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []
        # Some default settings
        self.needs_perts = False
        self.limber = False
        self.non_linear_sources = False
        self.non_linear_pk = False
        self._base_params = None
        self._needs_lensing_cross = False
        self._sigmaR_z_indices = {}

        if self.external_primordial_pk:
            self.extra_args["initial_power_model"] = (
                self.camb.initialpower.SplinedInitialPower
            )
            self.initial_power_args, self.power_params = {}, []
        else:
            power_spectrum = self.camb.CAMBparams.make_class_named(
                self.extra_args.get(
                    "initial_power_model", self.camb.initialpower.InitialPowerLaw
                ),
                self.camb.initialpower.InitialPower,
            )
            self.initial_power_args, self.power_params = self._extract_params(
                power_spectrum.set_params
            )

        nonlin = self.camb.CAMBparams.make_class_named(
            self.extra_args.get("non_linear_model", self.camb.nonlinear.Halofit),
            self.camb.nonlinear.NonLinearModel,
        )

        self.nonlin_args, self.nonlin_params = self._extract_params(nonlin.set_params)

        self.requires = str_to_list(getattr(self, "requires", []))
        self._transfer_requires = [
            p for p in self.requires if p not in self.get_can_support_params()
        ]
        self.requires = [p for p in self.requires if p not in self._transfer_requires]

    def _extract_params(self, set_func):
        args = {}
        params = []
        pars = getfullargspec(set_func)
        for arg in pars.args[1 : len(pars.args) - len(pars.defaults or [])]:
            params.append(arg)
        if pars.defaults:
            for arg, v in zip(
                pars.args[len(pars.args) - len(pars.defaults) :], pars.defaults
            ):
                if arg in self.extra_args:
                    args[arg] = self.extra_args.pop(arg)
                elif (
                    isinstance(v, numbers.Number) or v is None
                ) and "version" not in arg:
                    params.append(arg)
        return args, params

    def initialize_with_params(self):
        # must set WantTensors manually if using external_primordial_pk
        if not self.external_primordial_pk and set(self.input_params).intersection(
            {"r", "At"}
        ):
            self.extra_attrs["WantTensors"] = True
            self.extra_attrs["Accuracy.AccurateBB"] = True

        if "sigma8" in self.input_params:
            if "As" in self.input_params:
                raise LoggedError(
                    self.log,
                    "Both As and sigma8 have been provided as input. "
                    "This will likely cause ill-defined outputs.",
                )
            self.extra_attrs["WantTransfer"] = True
            self.add_to_redshifts([0.0])

    def initialize_with_provider(self, provider):
        if "sigma8" in self.input_params or "As" in self.output_params:
            if not self.needs_perts:
                raise LoggedError(
                    self.log,
                    "Using sigma8 as input or As as output "
                    "but not using any power spectrum results",
                )
            if (
                power_model := self.extra_args.get("initial_power_model")
            ) and not isinstance(
                self.camb.CAMBparams.make_class_named(power_model),
                self.camb.initialpower.InitialPowerLaw,
            ):
                raise LoggedError(
                    self.log,
                    "Using sigma8 as an input and As as an "
                    "output is only supported for power law "
                    "initial power spectra.",
                )
        super().initialize_with_provider(provider)

    def get_can_support_params(self):
        params = self.power_params + self.nonlin_params
        if not self.external_primordial_pk:
            params += ["sigma8"]
        return params

    def get_allow_agnostic(self):
        return False

    def set_cl_reqs(self, reqs):
        """
        Sets some common settings for both lensed and unlensed Cl's.
        """
        self.extra_args["lmax"] = max(max(reqs.values()), self.extra_args.get("lmax", 0))
        self.needs_perts = True
        self.extra_attrs["Want_CMB"] = True
        self.extra_attrs["WantCls"] = True
        if "TCMB" not in self.derived_extra:
            self.derived_extra += ["TCMB"]

    def must_provide(self, **requirements):
        # Computed quantities required by the likelihoods
        # Note that redshifts below are treated differently for background quantities,
        #   where no additional transfer computation is needed (e.g. H(z)),
        #   and matter-power-related quantities, that require additional computation
        #   and need the redshifts to be passed at CAMBParams instantiation.
        #   Also, we always make sure that those redshifts are sorted in descending order,
        #   since all CAMB related functions return quantities in that implicit order
        # The following super call makes sure that the requirements are properly
        # accumulated, i.e. taking the max of precision requests, etc.
        super().must_provide(**requirements)
        CAMBdata = self.camb.CAMBdata

        for k, v in self._must_provide.items():
            # Products and other computations
            if k == "Cl" or k == "lensed_scal_Cl":
                self.set_cl_reqs(v)
                cls = [a.lower() for a in v]
                needs_lensing = set(cls).intersection({"pp", "pt", "pe", "tp", "ep"})
                camb_result_key = "total" if k == "Cl" else "lensed_scalar"
                self.collectors[k] = Collector(
                    method=CAMBdata.get_cmb_power_spectra,
                    kwargs={
                        "spectra": list(
                            set(
                                (
                                    self.collectors[k].kwargs.get("spectra", [])
                                    if k in self.collectors
                                    else []
                                )
                                + [camb_result_key]
                                + (["lens_potential"] if needs_lensing else [])
                            )
                        ),
                        "raw_cl": False,
                    },
                )
                if "pp" in cls and self.extra_args.get("lens_potential_accuracy") is None:
                    self.extra_args["lens_potential_accuracy"] = 1
                self.non_linear_sources = (
                    self.extra_args.get("lens_potential_accuracy", 1) >= 1
                )
                if set(cls).intersection({"pt", "pe", "tp", "ep"}):
                    self._needs_lensing_cross = True
            elif k == "unlensed_Cl":
                self.set_cl_reqs(v)
                self.collectors[k] = Collector(
                    method=CAMBdata.get_cmb_power_spectra,
                    kwargs={"spectra": ["unlensed_total"], "raw_cl": False},
                )
            elif k == "Hubble":
                self.set_collector_with_z_pool(k, v["z"], CAMBdata.h_of_z)
            elif k in ["Omega_b", "Omega_cdm", "Omega_nu_massive"]:
                varnames = {
                    "Omega_b": "baryon",
                    "Omega_cdm": "cdm",
                    "Omega_nu_massive": "nu",
                }
                self.set_collector_with_z_pool(
                    k, v["z"], CAMBdata.get_Omega, kwargs={"var": varnames[k]}
                )
            elif k in ("angular_diameter_distance", "comoving_radial_distance"):
                self.set_collector_with_z_pool(k, v["z"], getattr(CAMBdata, k))
            elif k == "angular_diameter_distance_2":
                check_module_version(self.camb, "1.3.5")
                self.set_collector_with_z_pool(
                    k, v["z_pairs"], CAMBdata.angular_diameter_distance2, d=2
                )
            elif k == "sigma8_z":
                self.add_to_redshifts(v["z"])
                self.collectors[k] = Collector(
                    method=CAMBdata.get_sigma8, kwargs={}, post=(lambda *x: x[::-1])
                )  # returned in inverse order
                self.needs_perts = True
            elif k == "fsigma8":
                self.add_to_redshifts(v["z"])
                self.collectors[k] = Collector(
                    method=CAMBdata.get_fsigma8, kwargs={}, post=(lambda *x: x[::-1])
                )  # returned in inverse order
                self.needs_perts = True
            elif isinstance(k, tuple) and k[0] == "sigma_R":
                kwargs = v.copy()
                self.extra_args["kmax"] = max(
                    kwargs.pop("k_max"), self.extra_args.get("kmax", 0)
                )
                redshifts = kwargs.pop("z")
                self.add_to_redshifts(redshifts)
                var_pair = k[1:]

                def get_sigmaR(results, **tmp):
                    _indices = self._sigmaR_z_indices.get(var_pair)
                    if _indices is None or list(_indices) == []:
                        z_indices = []
                        calc = np.array(
                            results.Params.Transfer.PK_redshifts[
                                : results.Params.Transfer.PK_num_redshifts
                            ]
                        )
                        for z in redshifts:
                            for i, zcalc in enumerate(calc):
                                if np.isclose(zcalc, z, rtol=1e-4):
                                    z_indices += [i]
                                    break
                            else:
                                raise LoggedError(
                                    self.log,
                                    "sigma_R redshift not foundin computed P_K array %s",
                                    z,
                                )
                        _indices = np.array(z_indices, dtype=np.int32)
                        self._sigmaR_z_indices[var_pair] = _indices
                    R, z, sigma = results.get_sigmaR(
                        hubble_units=False, return_R_z=True, z_indices=_indices, **tmp
                    )
                    return z, R, sigma

                kwargs.update(dict(zip(["var1", "var2"], var_pair)))
                self.collectors[k] = Collector(method=get_sigmaR, kwargs=kwargs)
                self.needs_perts = True
            elif isinstance(k, tuple) and k[0] == "Pk_grid":
                kwargs = v.copy()
                self.extra_args["kmax"] = max(
                    kwargs.pop("k_max"), self.extra_args.get("kmax", 0)
                )
                self.add_to_redshifts(kwargs.pop("z"))
                # need to ensure can't have conflicts between requests from
                # different likelihoods. Store results without Hubble units.
                if kwargs.get("hubble_units", False) or kwargs.get("k_hunit", False):
                    raise LoggedError(
                        self.log, "hubble_units and k_hunit must be Falsefor consistency"
                    )
                kwargs["hubble_units"] = False
                kwargs["k_hunit"] = False
                if kwargs["nonlinear"]:
                    self.non_linear_pk = True
                var_pair = k[2:]
                kwargs.update(dict(zip(["var1", "var2"], var_pair)))
                self.collectors[k] = Collector(
                    method=CAMBdata.get_linear_matter_power_spectrum, kwargs=kwargs.copy()
                )
                self.needs_perts = True
            elif k == "source_Cl":
                if not getattr(self, "sources", None):
                    self.sources: InfoDict = {}
                for source, window in v["sources"].items():
                    # If it was already there, BoltzmannBase.must_provide() has already
                    # checked that old info == new info
                    if source not in self.sources:
                        self.sources[source] = window
                self.limber = v.get("limber", True)
                self.non_linear_sources = self.non_linear_sources or v.get(
                    "non_linear", False
                )
                if "lmax" in v:
                    self.extra_args["lmax"] = max(
                        v["lmax"], self.extra_args.get("lmax", 0)
                    )
                self.needs_perts = True
                self.collectors[k] = Collector(method=CAMBdata.get_source_cls_dict)
                self.extra_attrs["Want_cl_2D_array"] = True
                self.extra_attrs["WantCls"] = True
            elif k == "CAMBdata":
                # Just get CAMB results object
                self.collectors[k] = None
            elif v is None:
                # General derived parameters
                k_translated = self.translate_param(k)
                if k_translated not in self.derived_extra:
                    self.derived_extra += [k_translated]
                if k == "sigma8":
                    self.extra_attrs["WantTransfer"] = True
                    self.needs_perts = True
                    self.add_to_redshifts([0.0])
            else:
                raise LoggedError(
                    self.log, "This should not be happening. Contact the developers."
                )
        self.check_no_repeated_input_extra()

        # Computing non-linear corrections
        model = self.camb.model
        self.extra_attrs["NonLinear"] = {
            (True, True): model.NonLinear_both,
            (True, False): model.NonLinear_lens,
            (False, True): model.NonLinear_pk,
            (False, False): False,
        }[(self.non_linear_sources, self.non_linear_pk)]
        # set-set base CAMB params if anything might have changed
        self._base_params = None

        must_provide: InfoDict = {
            "CAMB_transfers": {
                "non_linear": self.non_linear_sources,
                "needs_perts": self.needs_perts,
            }
        }
        if self.external_primordial_pk and self.needs_perts:
            must_provide["primordial_scalar_pk"] = {
                "lmax": self.extra_args.get("lmax"),
                "kmax": self.extra_args.get("kmax"),
            }
            if self.extra_args.get("WantTensors"):
                self.extra_attrs["WantTensors"] = True
            if self.extra_attrs.get("WantTensors"):
                must_provide["primordial_tensor_pk"] = {
                    "lmax": self.extra_attrs.get(
                        "max_l_tensor", self.extra_args.get("lmax")
                    )
                }
        return must_provide

    def add_to_redshifts(self, z):
        """
        Adds redshifts to the list of them for which CAMB computes perturbations.
        """
        if not hasattr(self, "z_pool_for_perturbations"):
            self.z_pool_for_perturbations = Pool1D(z)
        else:
            self.z_pool_for_perturbations.update(z)
        self.extra_args["redshifts"] = np.flip(self.z_pool_for_perturbations.values)

    def set_collector_with_z_pool(self, k, zs, method, args=(), kwargs=empty_dict, d=1):
        """
        Creates a collector for a z-dependent quantity, keeping track of the pool of z's.
        """
        if k in self.collectors:
            z_pool = self.collectors[k].z_pool
            z_pool.update(zs)
        else:
            Pool = {1: Pool1D, 2: Pool2D}[d]
            z_pool = Pool(zs)
        if d == 1:
            kwargs_with_z = {"z": z_pool.values}
        else:
            kwargs_with_z = {
                "z1": np.array(z_pool.values[:, 0]),
                "z2": np.array(z_pool.values[:, 1]),
            }
        kwargs_with_z.update(kwargs)
        self.collectors[k] = Collector(
            method=method, z_pool=z_pool, kwargs=kwargs_with_z, args=args
        )

    def calculate(self, state, want_derived=True, **params_values_dict):
        try:
            params, results = self.provider.get_CAMB_transfers()
            if self.collectors or "sigma8" in self.derived_extra:
                if self.external_primordial_pk and self.needs_perts:
                    primordial_pk = self.provider.get_primordial_scalar_pk()
                    if primordial_pk.get("log_regular", True):
                        results.Params.InitPower.set_scalar_log_regular(
                            primordial_pk["kmin"],
                            primordial_pk["kmax"],
                            primordial_pk["Pk"],
                        )
                    else:
                        results.Params.InitPower.set_scalar_table(
                            primordial_pk["k"], primordial_pk["Pk"]
                        )
                    results.Params.InitPower.effective_ns_for_nonlinear = (
                        primordial_pk.get("effective_ns_for_nonlinear", 0.97)
                    )
                    if self.extra_attrs.get("WantTensors"):
                        primordial_pk = self.provider.get_primordial_tensor_pk()
                        if primordial_pk.get("log_regular", True):
                            results.Params.InitPower.set_tensor_log_regular(
                                primordial_pk["kmin"],
                                primordial_pk["kmax"],
                                primordial_pk["Pk"],
                            )
                        else:
                            results.Params.InitPower.set_tensor_table(
                                primordial_pk["k"], primordial_pk["Pk"]
                            )
                else:
                    args = {
                        self.translate_param(p): v
                        for p, v in params_values_dict.items()
                        if p in self.power_params
                    }
                    args.update(self.initial_power_args)
                    results.Params.InitPower.set_params(**args)
                if self.non_linear_sources or self.non_linear_pk:
                    args = {
                        self.translate_param(p): v
                        for p, v in params_values_dict.items()
                        if p in self.nonlin_params
                    }
                    args.update(self.nonlin_args)
                    results.Params.NonLinearModel.set_params(**args)
                results.power_spectra_from_transfer()
                if "sigma8" in params_values_dict:
                    sigma8 = results.get_sigma8_0()
                    results.Params.InitPower.As *= (
                        params_values_dict["sigma8"] ** 2 / sigma8**2
                    )
                    results.power_spectra_from_transfer()
            for product, collector in self.collectors.items():
                if collector:
                    state[product] = collector.method(
                        results, *collector.args, **collector.kwargs
                    )
                    if collector.post:
                        state[product] = collector.post(*state[product])
                else:
                    state[product] = results.copy()
        except self.camb.baseconfig.CAMBError as e:
            if self.stop_at_error:
                self.log.error(
                    "Computation error (see traceback below)! "
                    "Parameters sent to CAMB: %r and %r.\n"
                    "To ignore this kind of error, make 'stop_at_error: False'.",
                    dict(state["params"]),
                    dict(self.extra_args),
                )
                raise
            else:
                # Assumed to be a "parameter out of range" error.
                self.log.debug(
                    "Computation of cosmological products failed. "
                    "Assigning 0 likelihood and going on. "
                    "The output of the CAMB error was %s" % e
                )
                return False
            # Prepare derived parameters
        intermediates = CAMBOutputs(
            params, results, results.get_derived_params() if results else None
        )
        if want_derived:
            state["derived"] = self._get_derived_output(intermediates)
        # Prepare necessary extra derived parameters
        state["derived_extra"] = {
            p: self._get_derived(p, intermediates) for p in self.derived_extra
        }

    @staticmethod
    def _get_derived(p, intermediates):
        """
        General function to extract a single derived parameter.

        To get a parameter *from a likelihood* use `get_param` instead.
        """
        if intermediates.derived:
            derived = intermediates.derived.get(p, None)
            if derived is not None:
                return derived
        # Specific calls, if general ones fail:
        if p == "sigma8":
            return intermediates.results.get_sigma8_0()
        if p == "As":
            return intermediates.results.Params.InitPower.As
        try:
            return getattr(intermediates.camb_params, p)
        except AttributeError:
            try:
                return getattr(intermediates.results, p)
            except AttributeError:
                return getattr(intermediates.camb_params, "get_" + p, lambda: None)()

    def _get_derived_output(self, intermediates):
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
                    self.log,
                    "Derived param '%s' not implemented in the CAMB interface",
                    p,
                )
        return derived

    def _get_Cl(
        self, ell_factor=False, units="FIRASmuK2", lensed=True, scalar_only=False
    ):
        if scalar_only:
            assert lensed, "Only Implemented for lensed"
            which_key = "lensed_scal_Cl"
            which_result = "lensed_scalar"
        else:
            which_key = "Cl" if lensed else "unlensed_Cl"
            which_result = "total" if lensed else "unlensed_total"
        try:
            cl_camb = self.current_state[which_key][which_result].copy()
        except Exception:
            raise LoggedError(
                self.log,
                "No %s Cl's were computed. Are you sure that you have requested them?",
                "lensed" if lensed else "unlensed",
            )
        units_factor = self._cmb_unit_factor(
            units, self.current_state["derived_extra"]["TCMB"]
        )
        ls = np.arange(cl_camb.shape[0], dtype=np.int64)
        if not ell_factor:
            # unit conversion and ell_factor. CAMB output is *with* the factors already
            ells_factor = ls[1:] * (ls[1:] + 1)
            cl_camb[1:, :] /= ells_factor[..., np.newaxis]
            cl_camb[1:, :] *= (2 * np.pi) * units_factor**2
        elif units_factor != 1:
            cl_camb *= units_factor**2
        mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3, "et": 3}
        cls = {"ell": ls}
        for sp, i in mapping.items():
            cls[sp] = cl_camb[:, i]
        if lensed:
            cl_lens: np.ndarray | None = self.current_state["Cl"].get("lens_potential")
            if cl_lens is not None:
                cls["pp"] = cl_lens[:, 0].copy()
                if not ell_factor:
                    cls["pp"][1:] /= ells_factor**2 / (2 * np.pi)
                if self._needs_lensing_cross:
                    for i, cross in enumerate(["pt", "pe"]):
                        cls[cross] = cl_lens[:, i + 1].copy() * units_factor
                        if not ell_factor:
                            cls[cross][1:] /= ells_factor ** (3.0 / 2) / (2 * np.pi)
                        cls[cross[::-1]] = cls[cross]
        return cls

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl(ell_factor=ell_factor, units=units, lensed=True)

    def get_unlensed_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl(ell_factor=ell_factor, units=units, lensed=False)

    def get_lensed_scal_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl(
            ell_factor=ell_factor, units=units, lensed=True, scalar_only=True
        )

    def _get_z_dependent(self, quantity, z, _pool=None):
        # Partially reimplemented because of sigma8_z, etc., use different pool
        pool = None
        if quantity in ["sigma8_z", "fsigma8"]:
            pool = self.z_pool_for_perturbations
        return super()._get_z_dependent(quantity, z, pool=pool)

    def get_source_Cl(self):
        # get C_l^XX from the cosmological code
        try:
            cls = deepcopy(self.current_state["source_Cl"])
        except Exception:
            raise LoggedError(
                self.log,
                "No source Cl's were computed. "
                "Are you sure that you have requested some source?",
            )
        cls_dict: dict = dict()
        for term, cl in cls.items():
            term_tuple = tuple(
                (lambda x: x if x == "P" else list(self.sources)[int(x) - 1])(
                    _.strip("W")
                )
                for _ in term.split("x")
            )
            cls_dict[term_tuple] = cl
        cls_dict["ell"] = np.arange(cls[list(cls)[0]].shape[0])
        return cls_dict

    def get_CAMBdata(self):
        """
        Get the CAMB result object (must have been requested as a requirement).

        :return: CAMB's `CAMBdata <https://camb.readthedocs.io/en/latest/results.html>`_
                 result instance for the current parameters
        """
        return self.current_state["CAMBdata"]

    def get_can_provide_params(self):
        # possible derived parameters for derived_extra, excluding things that are
        # only input parameters.
        params_derived = list(get_class_methods(self.camb.CAMBparams))
        params_derived.remove("custom_source_names")
        fields = []
        for f, tp in self.camb.CAMBparams._fields_:
            if (
                tp is ctypes.c_double
                and "max_eta_k" not in f
                and f not in ["Alens", "num_nu_massless"]
            ):
                fields.append(f)
        fields += ["omega_de", "sigma8"]  # only parameters from CAMBdata
        if not self.external_primordial_pk:
            fields += ["As"]
        properties = get_properties(self.camb.CAMBparams)
        names = self.camb.model.derived_names + properties + fields + params_derived
        for name, mapped in self.renames.items():
            if mapped in names:
                names.append(name)
        # remove any parameters explicitly tagged as input requirements
        return set(names).difference(chain(self._transfer_requires, self.requires))

    def get_version(self):
        return self.camb.__version__

    def set(self, params_values_dict, state):
        # Prepare parameters to be passed: this is called from the CambTransfers instance
        args = {self.translate_param(p): v for p, v in params_values_dict.items()}
        # Generate and save
        self.log.debug("Setting parameters: %r and %r", args, self.extra_args)
        try:
            if not self._base_params:
                base_args = args.copy()
                base_args.update(self.extra_args)
                # Remove extra args that might
                # cause an error if the associated product is not requested
                if not self.extra_attrs["WantCls"]:
                    for not_needed in getfullargspec(
                        self.camb.CAMBparams.set_for_lmax
                    ).args[1:]:
                        base_args.pop(not_needed, None)
                self._reduced_extra_args = self.extra_args.copy()
                params = self.camb.set_params(**base_args)
                # pre-set the parameters that are not varying
                for non_param_func in ["set_classes", "set_matter_power", "set_for_lmax"]:
                    for fixed_param in getfullargspec(
                        getattr(self.camb.CAMBparams, non_param_func)
                    ).args[1:]:
                        if fixed_param in args:
                            raise LoggedError(
                                self.log,
                                "Trying to sample fixed theory parameter %s",
                                fixed_param,
                            )
                        self._reduced_extra_args.pop(fixed_param, None)
                if self.extra_attrs:
                    self.log.debug(
                        "Setting attributes of CAMBparams: %r", self.extra_attrs
                    )
                for attr, value in self.extra_attrs.items():
                    obj = params
                    if "." in attr:
                        parts = attr.split(".")
                        for p in parts[:-1]:
                            obj = getattr(obj, p)
                        par = parts[-1]
                    else:
                        par = attr
                    if hasattr(obj, par):
                        setattr(obj, par, value)
                    else:
                        raise LoggedError(
                            self.log,
                            "Some of the attributes to be set manually were not "
                            "recognized: %s=%s",
                            attr,
                            value,
                        )
                # Sources
                if source_dict := getattr(self, "sources", None):
                    self.log.debug("Setting sources: %r", self.sources)
                    sources = self.camb.sources
                    source_windows = []
                    for source, window in source_dict.items():
                        function = window.pop("function", None)
                        if function == "spline":
                            source_windows.append(sources.SplinedSourceWindow(**window))
                        elif function == "gaussian":
                            source_windows.append(sources.GaussianSourceWindow(**window))
                        else:
                            raise LoggedError(
                                self.log,
                                "Unknown source window function type %r",
                                function,
                            )
                        window["function"] = function
                    params.SourceWindows = source_windows
                    params.SourceTerms.limber_windows = self.limber
                self._base_params = params
            args.update(self._reduced_extra_args)
            return self.camb.set_params(self._base_params.copy(), **args)
        except self.camb.baseconfig.CAMBParamRangeError as e:
            if self.stop_at_error:
                raise LoggedError(
                    self.log, "%s\nOut of bound parameters: %r", e, params_values_dict
                )
            else:
                self.log.debug(
                    "%s;\n Out of bounds parameters. "
                    "Assigning 0 likelihood and going on.",
                    e,
                )
        except (self.camb.baseconfig.CAMBValueError, self.camb.baseconfig.CAMBError) as e:
            if self.stop_at_error:
                self.log.error(
                    "Error setting parameters (see traceback below)! "
                    "Parameters sent to CAMB: %r and %r.\n"
                    "To ignore this kind of error, make 'stop_at_error: False'.",
                    dict(state["params"]),
                    dict(self.extra_args),
                )
                raise
            else:
                self.log.debug("Error setting parameters: %s", e)
        except self.camb.baseconfig.CAMBUnknownArgumentError as e:
            raise LoggedError(
                self.log,
                "Some of the parameters passed to CAMB were not recognized: %s" % str(e),
            )
        return False

    def get_helper_theories(self):
        """
        Transfer functions are computed separately by camb.transfers, then this
        class uses the transfer functions to calculate power spectra (using A_s, n_s etc).
        """
        self._camb_transfers = CambTransfers(
            self,
            "camb.transfers",
            dict(stop_at_error=self.stop_at_error),
            timing=self.timer,
        )
        setattr(self._camb_transfers, "requires", self._transfer_requires)
        return {"camb.transfers": self._camb_transfers}

    def get_speed(self):
        if self._measured_speed:
            return self._measured_speed
        if not self.non_linear_sources:
            return self.speed * 10
        if {"omk", "omegak"}.intersection(set(self._camb_transfers.input_params)):
            return self.speed / 1.5
        return self.speed * 3

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(
            os.path.join(
                path, "code", cls._camb_repo_name[cls._camb_repo_name.find("/") + 1 :]
            )
        )

    @staticmethod
    def get_import_path(path):
        """
        Returns the ``camb`` module import path if there is a compiled version of CAMB in
        the given folder. Otherwise, raises ``FileNotFoundError``.
        """
        lib_fname = "cambdll.dll" if platform.system() == "Windows" else "camblib.so"
        if not os.path.isfile(os.path.realpath(os.path.join(path, "camb", lib_fname))):
            raise FileNotFoundError(
                f"Could not find compiled CAMB library {lib_fname} in {path}."
            )
        return path

    @classmethod
    def is_installed(cls, reload=False, **kwargs):
        if not kwargs.get("code", True):
            return True
        try:
            return bool(
                load_external_module(
                    "camb",
                    path=kwargs["path"],
                    get_import_path=cls.get_import_path,
                    min_version=cls._min_camb_version,
                    reload=reload,
                    logger=get_logger(cls.__name__),
                    not_installed_level="debug",
                )
            )
        except ComponentNotInstalledError:
            return False

    @classmethod
    def install(cls, path=None, code=True, no_progress_bars=False, **_kwargs):
        log = get_logger(cls.__name__)
        if not code:
            log.info("Code not requested. Nothing to do.")
            return True
        log.info("Installing pre-requisites...")
        exit_status = pip_install("wheel")
        if exit_status:
            log.error("Could not install pre-requisite: wheel")
            return False
        log.info("Downloading camb...")
        success = download_github_release(
            os.path.join(path, "code"),
            cls._camb_repo_name,
            cls._camb_repo_version,
            no_progress_bars=no_progress_bars,
            logger=log,
        )
        if not success:
            log.error("Could not download camb.")
            return False
        camb_path = cls.get_path(path)
        log.info("Compiling camb...")
        from subprocess import PIPE, Popen

        process_make = Popen(
            [sys.executable, "setup.py", "build_cluster"],
            cwd=camb_path,
            stdout=PIPE,
            stderr=PIPE,
        )
        out, err = process_make.communicate()
        if process_make.returncode:
            log.info(out.decode("utf-8"))
            log.info(err.decode("utf-8"))
            gcc_check = check_gcc_version(cls._camb_min_gcc_version, error_returns=False)
            if not gcc_check:
                cause = (
                    " Possible cause: it looks like `gcc` does not have the correct "
                    "version number (CAMB requires %s); and `ifort` is also "
                    "probably not available." % cls._camb_min_gcc_version
                )
            else:
                cause = ""
            log.error("Compilation failed!" + cause)
            return False
        return True


class CambTransfers(HelperTheory):
    """
    Helper theory class that calculates transfer functions only. The result is cached
    when only initial power spectrum or non-linear model parameters change
    """

    def __init__(self, cobaya_camb, name, info, timing=None):
        self.needs_perts = False
        self.non_linear_sources = False
        super().__init__(info, name, timing=timing)
        self.cobaya_camb = cobaya_camb
        self.camb = cobaya_camb.camb
        self.speed = self.cobaya_camb.speed * 1.5

    def get_can_support_params(self):
        supported_params = (
            self.camb.get_valid_numerical_params(
                transfer_only=True,
                dark_energy_model=self.cobaya_camb.extra_args.get("dark_energy_model"),
                recombination_model=self.cobaya_camb.extra_args.get(
                    "recombination_model"
                ),
            )
            - set(self.cobaya_camb.extra_args)
            - set(self.cobaya_camb.extra_attrs)
        )

        for name, mapped in self.cobaya_camb.renames.items():
            if mapped in supported_params:
                supported_params.add(name)
        return supported_params

    def get_allow_agnostic(self):
        return False

    def must_provide(self, **requirements):
        super().must_provide(**requirements)
        if opts := requirements.get("CAMB_transfers"):
            self.non_linear_sources = opts["non_linear"]
            self.needs_perts = opts["needs_perts"]
        self.cobaya_camb.check_no_repeated_input_extra()

    def get_CAMB_transfers(self):
        return self.current_state["results"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Set parameters
        camb_params = self.cobaya_camb.set(params_values_dict, state)
        # Failed to set parameters but no error raised
        # (e.g. out of computationally feasible range): lik=0
        if not camb_params:
            return False
        # Compute the transfer functions
        try:
            if self.non_linear_sources:
                # only need time sources if non-linear lensing or other non-linear
                # sources. Not needed just for non-linear PK.
                results = self.camb.get_transfer_functions(
                    camb_params, only_time_sources=True
                )
            else:
                results = (
                    self.camb.get_transfer_functions(camb_params)
                    if self.needs_perts
                    else self.camb.get_background(camb_params)
                )
            state["results"] = (camb_params, results)
        except self.camb.baseconfig.CAMBError as e:
            if self.stop_at_error:
                self.log.error(
                    "Computation error (see traceback below)! "
                    "Parameters sent to CAMB: %r and %r.\n"
                    "To ignore this kind of error, make 'stop_at_error: False'.",
                    dict(state["params"]),
                    dict(self.cobaya_camb.extra_args),
                )
                raise
            else:
                # Assumed to be a "parameter out of range" error.
                self.log.debug(
                    "Computation of cosmological products failed. "
                    "Assigning 0 likelihood and going on. "
                    "The output of the CAMB error was %s" % e
                )
                return False

    def initialize_with_params(self):
        if (
            len(set(self.input_params).intersection({"H0", "cosmomc_theta", "thetastar"}))
            > 1
        ):
            raise LoggedError(
                self.log, "Can't pass more than one of H0, theta, cosmomc_theta to CAMB."
            )
        if len(set(self.input_params).intersection({"tau", "zrei"})) > 1:
            raise LoggedError(
                self.log, "Can't pass more than one of tau and zrei to CAMB."
            )

        super().initialize_with_params()
