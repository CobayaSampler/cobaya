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
otherwise imported as a globally-installed Python package. If you want to force that
the global ``classy`` installation is used, pass ``path='global'``. Cobaya will print at
initialisation where CLASS was actually loaded from.


.. _classy_access:

Access to CLASS computation products
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can retrieve CLASS computation products within likelihoods (or other pipeline
components in general) or manually from a :class:`~model.Model` as long as you have added
them as requisites; see :doc:`cosmo_external_likelihood` or
:doc:`cosmo_external_likelihood_class` for the likelihood case, and :doc:`cosmo_model` for
the manual case.

The products that you can request and later retrieve are listed in
:func:`~theories.cosmo.BoltzmannBase.must_provide`.

For scalar parameters, you can add them as derived parameters in your input file. In
principle, you can add most of the parameters that you can retrieve manually in the CLASS
Python wrapper (the ones appearing inside the definition of the
``get_current_derived_parameters()`` function of the `python CLASS interface
<https://github.com/lesgourg/class_public/blob/master/python/classy.pyx>`__). If any of
them does not work (usually because it has been added to CLASS since Cobaya was last
updated), you can still add them as derived parameters in you input as long as you add
them also to the ``classy`` block as

.. code:: yaml

   theory:
     classy:
       [...]
       output_params: ["param1", "param2", ...]

If you would like to access a CLASS result that is not accessible in any of these ways,
you can access directly the return value of the `python CLASS interface
<https://github.com/lesgourg/class_public/blob/master/python/classy.pyx>`__ functions
``get_background()``, ``get_thermodynamics()``, ``get_primordial()``,
``get_perturbations()`` and  ``get_sources()``. To do that add to the requisites
``{get_CLASS_[...]: None}`` respectively, and retrieve it with
``provider.get_CLASS_[...]``.

In general, the use of these methods for direct access to CLASS results should be avoided
in public code, since it breaks compatibility with other Boltzmann codes at the likelihood
interface level. If you need a quantity for a public code that is not generally interfaced
in :func:`~theories.cosmo.BoltzmannBase.must_provide`, let us know if you think it makes
sense to add it.


String-vector parameters
^^^^^^^^^^^^^^^^^^^^^^^^

At the time of writing, the CLASS Python interface takes some vector-like parameters
as string in which different components are separater by a space. To be able to set priors
or fixed values on each components, see `this trick
<https://github.com/CobayaSampler/cobaya/issues/110#issuecomment-652333489>`_, and don't
forget the ``derived: False`` in the vector parameter (thanks to Lukas Hergt).

.. _classy_modify:

Modifying CLASS
^^^^^^^^^^^^^^^

If you modify CLASS and add new variables, make sure that the variables you create are
exposed in the Python interface
(`instructions here <https://github.com/lesgourg/class_public/wiki/Python-wrapper>`_).
If you follow those instructions you do not need to make any additional modification in
Cobaya.

If your modification involves new computed quantities, add the new quantities to the
return value of some of the direct-access methods listed in :ref:`classy_access`.

You can use the :doc:`model wrapper <cosmo_model>` to test your modification by
evaluating observables or getting derived quantities at known points in the parameter
space (set ``debug: True`` to get more detailed information of what exactly is passed to
CLASS).

In your CLASS modification, remember that you can raise a ``CosmoComputationError``
whenever the computation of any observable would fail, but you do not expect that
observable to be compatible with the data (e.g. at the fringes of the parameter
space). Whenever such an error is raised during sampling, the likelihood is assumed to be
zero, and the run is not interrupted.

.. note::

   If your modified CLASS has a lower version number than the minimum required by Cobaya,
   you will get an error at initialisation. You may still be able to use it by setting the
   option ``ignore_obsolete: True`` in the ``classy`` block (though you would be doing
   that at your own risk; ideally you should translate your modification to a newer CLASS
   version, in case there have been important fixes since the release of your baseline
   version).


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

You may prefer to install CLASS manually e.g. if you are planning to modify it.

.. note::

   *Pre-requisite*: CLASS' Python wrapper needs `Cython <https://cython.org/>`_, which you
   can install with

   .. code:: bash

      $ python -m pip install 'cython'

   In particular, if working with a modified CLASS version based on a version previous
   to v3.2.1, you need to change above ``'cython'`` by ``'cython<3'`` (see `this issue
   <https://github.com/lesgourg/class_public/issues/531>`_).

To download and install CLASS manually in a folder called ``CLASS`` under
``/path/to/cosmo``, simply do:

.. code:: bash

   $ cd /path/to/cosmo/
   $ git clone https://github.com/lesgourg/class_public.git CLASS --depth=1
   $ cd CLASS
   $ python setup.py build

If the **second** line produces an error (because you don't have ``git`` installed),
download the latest snapshot from `here
<https://github.com/lesgourg/class_public/archive/master.zip>`_, decompress it, rename the
resulting ``class_public-master`` folder to ``CLASS`` (optional) and run the
``python setup.py build`` command from there.

If the instructions above failed, follow those in the
`official CLASS web page <https://class-code.net/>`_.

Finally, in order for Cobaya to use this CLASS installation, you must specify the path to
it in the ``classy`` input block (otherwise a system-wide CLASS may be used instead):

.. code:: yaml

    theory:
      classy:
        path: /path/to/cosmo/CLASS
"""

import os
import platform
import sys
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any, NamedTuple

import numpy as np

from cobaya.component import ComponentNotInstalledError, load_external_module
from cobaya.install import check_gcc_version, download_github_release, pip_install
from cobaya.log import LoggedError, get_logger
from cobaya.theories.cosmo import BoltzmannBase
from cobaya.tools import (
    Pool1D,
    Pool2D,
    PoolND,
    VersionCheckError,
    combine_1d,
    get_compiled_import_path,
)


# Result collector
# NB: cannot use kwargs for the args, because the CLASS Python interface
#     is C-based, so args without default values are not named.
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: int | Sequence | None = None
    z_pool: PoolND | None = None
    post: Callable | None = None


# default non linear code -- same as CAMB
non_linear_default_code = "hmcode"
non_linear_null_value = "none"


class classy(BoltzmannBase):
    r"""
    CLASS cosmological Boltzmann code \cite{Blas:2011rf}.
    """

    # Name of the Class repo/folder and version to download
    _classy_repo_name = "lesgourg/class_public"
    _min_classy_version = "v3.3.3"
    _classy_min_gcc_version = "6.4"  # Lower ones are possible atm, but leak memory!
    _classy_repo_version = os.environ.get("CLASSY_REPO_VERSION", "master")

    classy_module: Any
    ignore_obsolete: bool

    def initialize(self):
        """Importing CLASS from the correct path, if given, and if not, globally."""
        install_path = None
        if self.packages_path is not None:
            install_path = self.get_path(self.packages_path)
        min_version = None if self.ignore_obsolete else self._min_classy_version
        try:
            try:
                self.classy_module = load_external_module(
                    "classy",
                    path=self.path,
                    install_path=install_path,
                    min_version=min_version,
                    get_import_path=self.get_import_path,
                    logger=self.log,
                    not_installed_level="debug",
                )
            # Regression introduced by CLASS v3.3 -- Deprecate CLASS <v3.3 eventually
            except (VersionCheckError, ComponentNotInstalledError) as ni_internal_excpt:
                try:
                    self.classy_module = load_external_module(
                        "classy",
                        path=self.path,
                        install_path=install_path,
                        min_version=min_version,
                        get_import_path=self.get_import_path_old,
                        logger=self.log,
                        not_installed_level="debug",
                    )
                    # Only runs if passed:
                    self.log.warning(
                        "Detected an old CLASS version (<3.3). "
                        "Please update: support for this will be deprecated soon."
                    )
                except ComponentNotInstalledError:
                    raise ni_internal_excpt
        except VersionCheckError as vc_excpt:
            raise VersionCheckError(
                str(vc_excpt) + " If you are using CLASS unmodified, upgrade with"
                "`cobaya-install classy --upgrade`. If you are using a modified CLASS, "
                "set the option `ignore_obsolete: True` for CLASS."
            ) from vc_excpt
        except ComponentNotInstalledError as ni_excpt:
            raise ComponentNotInstalledError(
                self.log,
                (
                    f"Could not find CLASS: {ni_excpt}. "
                    "To install it, run `cobaya-install classy`"
                ),
            ) from ni_excpt
        self.classy = self.classy_module.Class()
        super().initialize()
        # Add general CLASS stuff
        self.extra_args["output"] = self.extra_args.get("output", "")
        if "sBBN file" in self.extra_args:
            sbbn_dir, sbbn_file = os.path.split(self.extra_args["sBBN file"])
            if not os.path.isabs(sbbn_dir):
                # Discard dir, since it's standardized in the C code anyway.
                self.extra_args["sBBN file"] = os.path.join("/external/bbn", sbbn_file)
            # The "else" case (abs path) will fail in CLASS, and should be fixed.
        # Normalize `non_linear` vs `non linear`: prefer underscore
        # Keep this convention throughout the rest of this module!
        if "non linear" in self.extra_args:
            if "non_linear" in self.extra_args:
                raise LoggedError(
                    self.log,
                    (
                        "In `extra_args`, only one of `non_linear` or `non linear`"
                        " should be defined."
                    ),
                )
            self.extra_args["non_linear"] = self.extra_args.pop("non linear")
        # Normalize non_linear None|False --> "none"
        # Use default one if not specified
        if self.extra_args.get("non_linear", "dummy_string") in (None, False):
            self.extra_args["non_linear"] = non_linear_null_value
        elif "non_linear" not in self.extra_args or self.extra_args["non_linear"] is True:
            self.extra_args["non_linear"] = non_linear_default_code
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []

    def set_cl_reqs(self, reqs):
        """
        Sets some common settings for both lensend and unlensed Cl's.
        """
        if any(("t" in cl.lower()) for cl in reqs):
            self.extra_args["output"] += " tCl"
        if any((("e" in cl.lower()) or ("b" in cl.lower())) for cl in reqs):
            self.extra_args["output"] += " pCl"
        # For l_max_scalars, remember previous entries.
        self.extra_args["l_max_scalars"] = max(
            self.extra_args.get("l_max_scalars", 0), max(reqs.values())
        )
        if "T_cmb" not in self.derived_extra:
            self.derived_extra += ["T_cmb"]

    def must_provide(self, **requirements):
        # Computed quantities required by the likelihood
        super().must_provide(**requirements)
        for k, v in self._must_provide.items():
            # Products and other computations
            if k == "Cl":
                self.set_cl_reqs(v)
                # For modern experiments, always lensed Cl's!
                self.extra_args["output"] += " lCl"
                self.extra_args["lensing"] = "yes"
                self.collectors[k] = Collector(
                    method="lensed_cl", kwargs={"lmax": self.extra_args["l_max_scalars"]}
                )
            elif k == "unlensed_Cl":
                self.set_cl_reqs(v)
                self.collectors[k] = Collector(
                    method="raw_cl", kwargs={"lmax": self.extra_args["l_max_scalars"]}
                )
            elif k == "Hubble":
                self.set_collector_with_z_pool(
                    k, v["z"], "Hubble", args_names=["z"], arg_array=0
                )
            elif k in ["Omega_b", "Omega_cdm", "Omega_nu_massive"]:
                func_name = {
                    "Omega_b": "Om_b",
                    "Omega_cdm": "Om_cdm",
                    "Omega_nu_massive": "Om_ncdm",
                }[k]
                self.set_collector_with_z_pool(
                    k, v["z"], func_name, args_names=["z"], arg_array=0
                )
            elif k == "angular_diameter_distance":
                self.set_collector_with_z_pool(
                    k, v["z"], "angular_distance", args_names=["z"], arg_array=0
                )
            elif k == "comoving_radial_distance":
                self.set_collector_with_z_pool(
                    k,
                    v["z"],
                    "z_of_r",
                    args_names=["z"],
                    # returns r and dzdr!
                    post=(lambda r, dzdr: r),
                )
            elif k == "angular_diameter_distance_2":
                self.set_collector_with_z_pool(
                    k,
                    v["z_pairs"],
                    "angular_distance_from_to",
                    args_names=["z1", "z2"],
                    arg_array=[0, 1],
                    d=2,
                )
            elif isinstance(k, tuple) and k[0] == "Pk_grid":
                self.extra_args["output"] += " mPk"
                v = deepcopy(v)
                self.add_P_k_max(v.pop("k_max"), units="1/Mpc")
                # NB: Actually, only the max z is used, and the actual sampling in z
                # for computing P(k,z) is controlled by `perturb_sampling_stepsize`
                # (default: 0.1). But let's leave it like this in case this changes
                # in the future.
                self.add_z_for_matter_power(v.pop("z"))
                if v["nonlinear"]:
                    if "non_linear" not in self.extra_args:
                        # this is redundant with initialisation, but just in case
                        self.extra_args["non_linear"] = non_linear_default_code
                    elif self.extra_args["non_linear"] == non_linear_null_value:
                        raise LoggedError(
                            self.log,
                            (
                                "Non-linear Pk requested, but `non_linear: "
                                f"{non_linear_null_value}` imposed in "
                                "`extra_args`"
                            ),
                        )
                pair = k[2:]
                if pair == ("delta_tot", "delta_tot"):
                    v["only_clustering_species"] = False
                    self.collectors[k] = Collector(
                        method="get_pk_and_k_and_z",
                        kwargs=v,
                        post=(lambda P, kk, z: (kk, z, np.array(P).T)),
                    )
                elif pair == ("delta_nonu", "delta_nonu"):
                    v["only_clustering_species"] = True
                    self.collectors[k] = Collector(
                        method="get_pk_and_k_and_z",
                        kwargs=v,
                        post=(lambda P, kk, z: (kk, z, np.array(P).T)),
                    )
                elif pair == ("Weyl", "Weyl"):
                    self.extra_args["output"] += " mTk"
                    self.collectors[k] = Collector(
                        method="get_Weyl_pk_and_k_and_z",
                        kwargs=v,
                        post=(lambda P, kk, z: (kk, z, np.array(P).T)),
                    )
                else:
                    raise LoggedError(self.log, "NotImplemented in CLASS: %r", pair)
            elif k == "sigma8_z":
                self.add_z_for_matter_power(v["z"])
                self.set_collector_with_z_pool(
                    k,
                    v["z"],
                    "sigma",
                    args=[8],
                    args_names=["R", "z"],
                    kwargs={"h_units": True},
                    arg_array=1,
                )
            elif k == "fsigma8":
                self.add_z_for_matter_power(v["z"])
                z_step = 0.1  # left to CLASS default; increasing does not appear to help
                self.set_collector_with_z_pool(
                    k,
                    v["z"],
                    "effective_f_sigma8",
                    args=[z_step],
                    args_names=["z", "z_step"],
                    arg_array=0,
                )
            elif isinstance(k, tuple) and k[0] == "sigma_R":
                self.extra_args["output"] += " mPk"
                self.add_P_k_max(v.pop("k_max"), units="1/Mpc")
                # NB: See note about redshifts in Pk_grid
                self.add_z_for_matter_power(v["z"])
                pair = k[1:]
                try:
                    method = {
                        ("delta_tot", "delta_tot"): "sigma",
                        ("delta_nonu", "delta_nonu"): "sigma_cb",
                    }[pair]
                except KeyError as excpt:
                    raise LoggedError(
                        self.log, f"sigma(R,z) not implemented for {pair}"
                    ) from excpt
                self.collectors[k] = Collector(
                    method=method,
                    kwargs={"h_units": False},
                    args=[v["R"], v["z"]],
                    args_names=["R", "z"],
                    arg_array=[[0], [1]],
                    post=(lambda R, z, sigma: (z, R, sigma.T)),
                )
            elif k in [
                f"CLASS_{q}"
                for q in [
                    "background",
                    "thermodynamics",
                    "primordial",
                    "perturbations",
                    "sources",
                ]
            ]:
                # Get direct CLASS results
                self.collectors[k] = Collector(method=f"get_{k.lower()[len('CLASS_') :]}")
            elif v is None:
                k_translated = self.translate_param(k)
                if k_translated not in self.derived_extra:
                    self.derived_extra += [k_translated]
            else:
                raise LoggedError(self.log, "Requested product not known: %r", {k: v})
        # Derived parameters (if some need some additional computations)
        if any(("sigma8" in s) for s in set(self.output_params).union(requirements)):
            self.extra_args["output"] += " mPk"
            self.add_P_k_max(1, units="1/Mpc")
        # Adding tensor modes if requested
        if self.extra_args.get("r") or "r" in self.input_params:
            self.extra_args["modes"] = "s,t"
            # TEMPORARY: disable new limber scheme to avoid CLASS error (as of v3.3.3)
            self.extra_args["want_lcmb_full_limber"] = "no"
            self.log.warn(
                "Disabled finer Limber scheme ('want_lcmb_full_limber=no') because it is "
                "not implemented for tensor modes as of CLASS v3.3.3."
            )
        # If B spectrum with l>50, or lensing, recommend using a non-linear code
        cls = self._must_provide.get("Cl", {})
        has_BB_l_gt_50 = (
            any(("b" in cl.lower()) for cl in cls)
            and max(cls[cl] for cl in cls if "b" in cl.lower()) > 50
        )
        has_lensing = any(("p" in cl.lower()) for cl in cls)
        if (has_BB_l_gt_50 or has_lensing) and self.extra_args.get(
            "non_linear"
        ) == non_linear_null_value:
            self.log.warning(
                "Requesting BB for ell>50 or lensing Cl's: "
                "using a non-linear code is recommended (and you are not "
                "using any). To activate it, set "
                "'non_linear: halofit|hmcode|...' in classy's 'extra_args'."
            )
        # Cleanup of products string
        self.extra_args["output"] = " ".join(set(self.extra_args["output"].split()))
        self.check_no_repeated_input_extra()

    def add_z_for_matter_power(self, z):
        if getattr(self, "z_for_matter_power", None) is None:
            self.z_for_matter_power = np.empty(0)
        self.z_for_matter_power = np.flip(combine_1d(z, self.z_for_matter_power))
        self.extra_args["z_pk"] = " ".join(["%g" % zi for zi in self.z_for_matter_power])

    def set_collector_with_z_pool(
        self,
        k,
        zs,
        method,
        args=(),
        args_names=(),
        kwargs=None,
        arg_array=None,
        post=None,
        d=1,
    ):
        """
        Creates a collector for a z-dependent quantity, keeping track of the pool of z's.

        If ``z`` is an arg, i.e. it is in ``args_names``, then omit it in the ``args``,
        e.g. ``args_names=["a", "z", "b"]`` should be passed together with
        ``args=[a_value, b_value]``.
        """
        if k in self.collectors:
            z_pool = self.collectors[k].z_pool
            z_pool.update(zs)
        else:
            Pool = {1: Pool1D, 2: Pool2D}[d]
            z_pool = Pool(zs)
        # Insert z as arg or kwarg
        kwargs = kwargs or {}
        if d == 1 and "z" in kwargs:
            kwargs = deepcopy(kwargs)
            kwargs["z"] = z_pool.values
        elif d == 1 and "z" in args_names:
            args = deepcopy(args)
            i_z = args_names.index("z")
            args = list(args[:i_z]) + [z_pool.values] + list(args[i_z:])
        elif d == 2 and "z1" in args_names and "z2" in args_names:
            # z1 assumed appearing before z2!
            args = deepcopy(args)
            i_z1 = args_names.index("z1")
            i_z2 = args_names.index("z2")
            args = (
                list(args[:i_z1])
                + [z_pool.values[:, 0]]
                + list(args[i_z1:i_z2])
                + [z_pool.values[:, 1]]
                + list(args[i_z2:])
            )
        else:
            raise LoggedError(
                self.log,
                f"I do not know how to insert the redshift for collector method {method} "
                f"of requisite {k}",
            )
        self.collectors[k] = Collector(
            method=method,
            z_pool=z_pool,
            args=args,
            args_names=args_names,
            kwargs=kwargs,
            arg_array=arg_array,
            post=post,
        )

    def add_P_k_max(self, k_max, units):
        r"""
        Unifies treatment of :math:`k_\mathrm{max}` for matter power spectrum:
        ``P_k_max_[1|h]/Mpc``.

        Make ``units="1/Mpc"|"h/Mpc"``.
        """
        # Fiducial h conversion (high, though it may slow the computations)
        h_fid = 1
        if units == "h/Mpc":
            k_max *= h_fid
        # Take into account possible manual set of P_k_max_***h/Mpc*** through extra_args
        k_max_old = self.extra_args.pop(
            "P_k_max_1/Mpc", h_fid * self.extra_args.pop("P_k_max_h/Mpc", 0)
        )
        self.extra_args["P_k_max_1/Mpc"] = max(k_max, k_max_old)

    def set(self, params_values_dict):
        # If no output requested, remove arguments that produce an error
        # (e.g. complaints if halofit requested but no Cl's computed.) ?????
        # Needed for facilitating post-processing
        if not self.extra_args["output"]:
            for k in ["non_linear", "hmcode_version"]:
                self.extra_args.pop(k, None)
        # Prepare parameters to be passed: this-iteration + extra
        args = {self.translate_param(p): v for p, v in params_values_dict.items()}
        args.update(self.extra_args)
        # Generate and save
        self.param_dict_debug("Setting parameters: %r", args)
        self.classy.set(**args)

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Set parameters
        self.set(params_values_dict)
        # Compute!
        try:
            self.classy.compute()
        # "Valid" failure of CLASS: parameters too extreme -> log and report
        except self.classy_module.CosmoComputationError as e:
            if self.stop_at_error:
                self.log.error(
                    "Computation error (see traceback below)! "
                    "Parameters sent to CLASS: %r and %r.\n"
                    "To ignore this kind of error, make 'stop_at_error: False'.",
                    state["params"],
                    dict(self.extra_args),
                )
                raise
            else:
                self.log.debug(
                    "Computation of cosmological products failed. "
                    "Assigning 0 likelihood and going on. "
                    "The output of the CLASS error was %s",
                    e,
                )
            return False
        # CLASS not correctly initialized, or input parameters not correct
        except self.classy_module.CosmoSevereError:
            self.log.error(
                "Serious error setting parameters or computing results. "
                "The parameters passed were %r and %r. To see the original "
                "CLASS' error traceback, make 'debug: True'.",
                state["params"],
                self.extra_args,
            )
            raise  # No LoggedError, so that CLASS traceback gets printed
        # Gather products
        for product, collector in self.collectors.items():
            # Special case: sigma8 needs H0, which cannot be known beforehand:
            if "sigma8" in self.collectors:
                self.collectors["sigma8"].args[0] = 8 / self.classy.h()
            method = getattr(self.classy, collector.method)
            arg_array = self.collectors[product].arg_array
            if isinstance(arg_array, int):
                arg_array = np.atleast_1d(arg_array)
            if arg_array is None:
                state[product] = method(
                    *self.collectors[product].args, **self.collectors[product].kwargs
                )
            elif isinstance(arg_array, Sequence) or isinstance(arg_array, np.ndarray):
                arg_array = np.array(arg_array)
                if len(arg_array.shape) == 1:
                    # if more than one vectorised arg, assume all vectorised in parallel
                    n_values = len(self.collectors[product].args[arg_array[0]])
                    state[product] = np.zeros(n_values)
                    args = deepcopy(list(self.collectors[product].args))
                    for i in range(n_values):
                        for arg_arr_index in arg_array:
                            args[arg_arr_index] = self.collectors[product].args[
                                arg_arr_index
                            ][i]
                        state[product][i] = method(
                            *args, **self.collectors[product].kwargs
                        )
                elif len(arg_array.shape) == 2:
                    if len(arg_array) > 2:
                        raise NotImplementedError("Only 2 array expanded vars so far.")
                    # Create outer combinations
                    x_and_y = np.array(
                        np.meshgrid(
                            self.collectors[product].args[arg_array[0, 0]],
                            self.collectors[product].args[arg_array[1, 0]],
                        )
                    ).T
                    args = deepcopy(list(self.collectors[product].args))
                    result = np.empty(shape=x_and_y.shape[:2])
                    for i, row in enumerate(x_and_y):
                        for j, column_element in enumerate(x_and_y[i]):
                            args[arg_array[0, 0]] = column_element[0]
                            args[arg_array[1, 0]] = column_element[1]
                            result[i, j] = method(
                                *args, **self.collectors[product].kwargs
                            )
                    state[product] = (
                        self.collectors[product].args[arg_array[0, 0]],
                        self.collectors[product].args[arg_array[1, 0]],
                        result,
                    )
                else:
                    raise ValueError("arg_array not correctly formatted.")
            elif arg_array in self.collectors[product].kwargs:
                value = np.atleast_1d(self.collectors[product].kwargs[arg_array])
                state[product] = np.zeros(value.shape)
                for i, v in enumerate(value):
                    kwargs = deepcopy(self.collectors[product].kwargs)
                    kwargs[arg_array] = v
                    state[product][i] = method(*self.collectors[product].args, **kwargs)
            else:
                raise LoggedError(
                    self.log,
                    "Variable over which to do an array call "
                    f"not known: arg_array={arg_array}",
                )
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
        # TODO: fails with derived_requested=False
        # Put all parameters in CLASS nomenclature (self.derived_extra already is)
        requested = [
            self.translate_param(p)
            for p in (self.output_params if derived_requested else [])
        ]
        requested_and_extra = dict.fromkeys(set(requested).union(self.derived_extra))
        # Parameters with their own getters or different CLASS internal names
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
                [p for p, v in requested_and_extra.items() if v is None]
            )
        )
        # Separate the parameters before returning
        # Remember: self.output_params is in sampler nomenclature,
        # but self.derived_extra is in CLASS
        derived = {
            p: requested_and_extra[self.translate_param(p)] for p in self.output_params
        }
        derived_extra = {p: requested_and_extra[p] for p in self.derived_extra}
        return derived, derived_extra

    def _get_Cl(self, ell_factor=False, units="FIRASmuK2", lensed=True):
        which_key = "Cl" if lensed else "unlensed_Cl"
        which_error = "lensed" if lensed else "unlensed"
        try:
            cls = deepcopy(self.current_state[which_key])
        except Exception as excpt:
            raise LoggedError(
                self.log,
                "No %s Cl's were computed. Are you sure that you have requested them?",
                which_error,
            ) from excpt
        # unit conversion and ell_factor
        ells_factor = (
            ((cls["ell"] + 1) * cls["ell"] / (2 * np.pi))[2:] if ell_factor else 1
        )
        units_factor = self._cmb_unit_factor(
            units, self.current_state["derived_extra"]["T_cmb"]
        )
        for cl in cls:
            if cl == "ell":
                continue
            units_power = float(sum(cl.count(p) for p in ["t", "e", "b"]))
            cls[cl][2:] *= units_factor**units_power
            if ell_factor:
                if "p" not in cl:
                    cls[cl][2:] *= ells_factor
                elif cl == "pp" and lensed:
                    cls[cl][2:] *= ells_factor**2 * (2 * np.pi)
                elif "p" in cl and lensed:
                    cls[cl][2:] *= ells_factor ** (3 / 2) * np.sqrt(2 * np.pi)
        return cls

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl(ell_factor=ell_factor, units=units, lensed=True)

    def get_unlensed_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl(ell_factor=ell_factor, units=units, lensed=False)

    def get_CLASS_background(self):
        """Direct access to ``get_background`` from the CLASS python interface."""
        return self.current_state["CLASS_background"]

    def get_CLASS_thermodynamics(self):
        """Direct access to ``get_thermodynamics`` from the CLASS python interface."""
        return self.current_state["CLASS_thermodynamics"]

    def get_CLASS_primordial(self):
        """Direct access to ``get_primordial`` from the CLASS python interface."""
        return self.current_state["CLASS_primordial"]

    def get_CLASS_perturbations(self):
        """Direct access to ``get_perturbations`` from the CLASS python interface."""
        return self.current_state["CLASS_perturbations"]

    def get_CLASS_sources(self):
        """Direct access to ``get_sources`` from the CLASS python interface."""
        return self.current_state["CLASS_sources"]

    def close(self):
        self.classy.empty()

    def get_can_provide_params(self):
        names = [
            "h",
            "H0",
            "Omega_Lambda",
            "Omega_cdm",
            "Omega_b",
            "Omega_m",
            "Omega_k",
            "rs_drag",
            "tau_reio",
            "z_reio",
            "z_rec",
            "tau_rec",
            "m_ncdm_tot",
            "Neff",
            "YHe",
            "age",
            "conformal_age",
            "sigma8",
            "sigma8_cb",
            "theta_s_100",
        ]
        for name, mapped in self.renames.items():
            if mapped in names:
                names.append(name)
        return names

    def get_can_support_params(self):
        # non-exhaustive list of supported input parameters that will be assigned to
        # classy if they are varied
        return ["H0"]

    def get_version(self):
        return getattr(self.classy_module, "__version__", None)

    # Installation routines

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(os.path.join(path, "code", cls.__name__))

    @staticmethod
    def get_import_path(path):
        return get_compiled_import_path(path)

    @staticmethod
    def get_import_path_old(path):
        return get_compiled_import_path(os.path.join(path, "python"))

    @classmethod
    def is_compatible(cls):
        if platform.system() == "Windows":
            return False
        return True

    @classmethod
    def is_installed(cls, reload=False, **kwargs):
        if not kwargs.get("code", True):
            return True
        try:
            return bool(
                load_external_module(
                    "classy",
                    path=kwargs["path"],
                    get_import_path=cls.get_import_path,
                    min_version=cls._min_classy_version,
                    reload=reload,
                    logger=get_logger(cls.__name__),
                    not_installed_level="debug",
                )
            )
        # Regression introduced by CLASS v3.3 -- Deprecate CLASS <v3.3 eventually
        except ComponentNotInstalledError:
            try:
                success = bool(
                    load_external_module(
                        "classy",
                        path=kwargs["path"],
                        get_import_path=cls.get_import_path_old,
                        min_version=cls._min_classy_version,
                        reload=reload,
                        logger=get_logger(cls.__name__),
                        not_installed_level="debug",
                    )
                )
                # Only runs if passed:
                get_logger(cls.__name__).warning(
                    "Detected an old CLASS version (<3.3). "
                    "Please update: support for this will be deprecated soon."
                )
                return success
            except ComponentNotInstalledError:
                return False

    @classmethod
    def install(cls, path=None, code=True, no_progress_bars=False, **_kwargs):
        log = get_logger(cls.__name__)
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
            os.path.join(path, "code"),
            cls._classy_repo_name,
            cls._classy_repo_version,
            directory=cls.__name__,
            no_progress_bars=no_progress_bars,
            logger=log,
        )
        if not success:
            log.error("Could not download classy.")
            return False
        # Compilation
        # gcc check after downloading, in case the user wants to change the compiler by
        # hand in the Makefile
        classy_path = cls.get_path(path)
        if not check_gcc_version(cls._classy_min_gcc_version, error_returns=False):
            log.error(
                "Your gcc version is too low! CLASS would probably compile, "
                "but it would leak memory when running a chain. Please use a "
                "gcc version newer than %s. You can still compile CLASS by hand, "
                "maybe changing the compiler in the Makefile. CLASS has been "
                "downloaded into %r",
                cls._classy_min_gcc_version,
                classy_path,
            )
            return False
        log.info("Compiling classy...")
        from subprocess import PIPE, Popen

        env = deepcopy(os.environ)
        env.update({"PYTHON": sys.executable})
        process_make = Popen(
            [sys.executable, "setup.py", "build"],
            cwd=classy_path,
            stdout=PIPE,
            stderr=PIPE,
            env=env,
        )
        out, err = process_make.communicate()
        if process_make.returncode:
            log.info(out.decode("utf-8"))
            log.info(err.decode("utf-8"))
            log.error("Compilation failed!")
            return False
        return True
