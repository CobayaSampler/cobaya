"""
.. module:: BoltzmannBase

:Synopsis: Template for Cosmological theory codes.
           Mostly here to document how to compute and get observables.
:Author: Jesus Torrado

"""

from collections.abc import Iterable, Mapping

import numpy as np
from scipy.interpolate import RectBivariateSpline

from cobaya.conventions import Const
from cobaya.log import LoggedError, abstract, get_logger
from cobaya.theory import Theory
from cobaya.tools import check_2d, combine_1d, combine_2d, deepcopy_where_possible
from cobaya.typing import InfoDict, empty_dict

H_units_conv_factor = {"1/Mpc": 1, "km/s/Mpc": Const.c_km_s}


class BoltzmannBase(Theory):
    _is_abstract = True
    renames: Mapping[str, str] = empty_dict
    extra_args: InfoDict
    _must_provide: dict
    path: str

    def initialize(self):
        # Dict of named tuples to collect requirements and computation methods
        self.collectors = {}
        # Additional input parameters (e.g. to pass to setter function, to set as attr...)
        self.extra_args = deepcopy_where_possible(self.extra_args) or {}
        self._must_provide = {}

    def initialize_with_params(self):
        self.check_no_repeated_input_extra()

    def get_allow_agnostic(self):
        return True

    def translate_param(self, p):
        return self.renames.get(p, p)

    def get_param(self, p):
        translated = self.translate_param(p)
        for pool in ["params", "derived", "derived_extra"]:
            value = (self.current_state[pool] or {}).get(translated, None)
            if value is not None:
                return value

        raise LoggedError(self.log, "Parameter not known: '%s'", p)

    def _norm_vars_pairs(self, vars_pairs, name):
        # Empty list: default to *total matter*: CMB + Baryon + MassiveNu
        vars_pairs = vars_pairs or [2 * ["delta_tot"]]
        if not isinstance(vars_pairs, Iterable):
            raise LoggedError(
                self.log,
                "vars_pairs must be an iterable of pairs "
                "of variable names: got '%r' for %s",
                vars_pairs,
                name,
            )
        if isinstance(list(vars_pairs)[0], str):
            vars_pairs = [vars_pairs]
        pairs = set()
        for pair in vars_pairs:
            if len(list(pair)) != 2 or not all(isinstance(x, str) for x in pair):
                raise LoggedError(
                    self.log, "Cannot understand vars_pairs '%r' for %s", vars_pairs, name
                )
            pairs.add(tuple(sorted(pair)))
        return pairs

    def _check_args(self, req, value, vals):
        for par in vals:
            if par not in value:
                raise LoggedError(
                    self.log, "%s must specify %s in requirements", req, par
                )

    def must_provide(self, **requirements):
        r"""
        Specifies the quantities that this Boltzmann code is requested to compute.

        Typical requisites in Cosmology (as keywords, case-insensitive):

        - ``Cl={...}``: CMB lensed power spectra, as a dictionary ``{[spectrum]:
          l_max}``, where the possible spectra are combinations of ``"t"``, ``"e"``,
          ``"b"`` and ``"p"`` (lensing potential). Get with :func:`~BoltzmannBase.get_Cl`.
        - ``unlensed_Cl={...}``: CMB unlensed power spectra, as a dictionary
          ``{[spectrum]: l_max}``, where the possible spectra are combinations of
          ``"t"``, ``"e"``, ``"b"``. Get with :func:`~BoltzmannBase.get_unlensed_Cl`.
        - **[BETA: CAMB only; notation may change!]** ``source_Cl={...}``:
          :math:`C_\ell` of given sources with given windows, e.g.:
          ``source_name: {"function": "spline"|"gaussian", [source_args]``;
          for now, ``[source_args]`` follows the notation of `CAMBSources
          <https://camb.readthedocs.io/en/latest/sources.html>`_.
          It can also take ``"lmax": [int]``, ``"limber": True`` if Limber approximation
          desired, and ``"non_linear": True`` if non-linear contributions requested.
          Get with :func:`~BoltzmannBase.get_source_Cl`.
        - ``Pk_interpolator={...}``: Matter power spectrum interpolator in :math:`(z, k)`.
          Takes ``"z": [list_of_evaluated_redshifts]``, ``"k_max": [k_max]``,
          ``"nonlinear": [True|False]``,
          ``"vars_pairs": [["delta_tot", "delta_tot"], ["Weyl", "Weyl"], [...]]}``.
          Notice that ``k_min`` cannot be specified. To reach a lower one, use
          ``extra_args`` in CAMB to increase ``accuracyboost`` and ``TimeStepBoost``, or
          in CLASS to decrease ``k_min_tau0``. The method :func:`~get_Pk_interpolator` to
          retrieve the interpolator admits extrapolation limits ``extrap_[kmax|kmin]``.
          It is recommended to use ``extrap_kmin`` to reach the desired ``k_min``, and
          increase precision parameters only as much as necessary to improve over the
          interpolation. All :math:`k` values should be in units of
          :math:`1/\mathrm{Mpc}`.

          Non-linear contributions are included by default. Note that the non-linear
          setting determines whether non-linear corrections are calculated; the
          :func:`~get_Pk_interpolator` method also has a non-linear argument to specify if
          you want the linear or non-linear spectrum returned (to have both linear and
          non-linear spectra available request a tuple ``(False,True)`` for the non-linear
          argument).
        - ``Pk_grid={...}``: similar to ``Pk_interpolator`` except that rather than
          returning a bicubic spline object it returns the raw power spectrum grid as
          a ``(k, z, P(z,k))`` set of arrays. Get with :func:`~BoltzmannBase.get_Pk_grid`.
        - ``sigma_R={...}``: RMS linear fluctuation in spheres of radius :math:`R` at
          redshifts :math:`z`. Takes ``"z": [list_of_evaluated_redshifts]``,
          ``"k_max": [k_max]``, ``"vars_pairs": [["delta_tot", "delta_tot"],  [...]]``,
          ``"R": [list_of_evaluated_R]``. Note that :math:`R` is in :math:`\mathrm{Mpc}`,
          not :math:`h^{-1}\,\mathrm{Mpc}`. Get with :func:`~BoltzmannBase.get_sigma_R`.
        - ``Hubble={'z': [z_1, ...]}``: Hubble rates at the requested redshifts.
          Get it with :func:`~BoltzmannBase.get_Hubble`.
        - ``Omega_b={'z': [z_1, ...]}``: Baryon density parameter at the requested
          redshifts. Get it with :func:`~BoltzmannBase.get_Omega_b`.
        - ``Omega_cdm={'z': [z_1, ...]}``: Cold dark matter density parameter at the
          requested redshifts. Get it with :func:`~BoltzmannBase.get_Omega_cdm`.
        - ``Omega_nu_massive={'z': [z_1, ...]}``: Massive neutrinos' density parameter at
          the requested redshifts. Get it with
          :func:`~BoltzmannBase.get_Omega_nu_massive`.
        - ``angular_diameter_distance={'z': [z_1, ...]}``: Physical angular
          diameter distance to the redshifts requested. Get it with
          :func:`~BoltzmannBase.get_angular_diameter_distance`.
        - ``angular_diameter_distance_2={'z_pairs': [(z_1a, z_1b), (z_2a, z_2b)...]}``:
          Physical angular diameter distance between the pairs of redshifts requested.
          If a 1d array of redshifts is passed as `z_pairs`, all possible combinations
          of two are computed and stored (not recommended if only a subset is needed).
          Get it with :func:`~BoltzmannBase.get_angular_diameter_distance_2`.
        - ``comoving_radial_distance={'z': [z_1, ...]}``: Comoving radial distance
          from us to the redshifts requested. Get it with
          :func:`~BoltzmannBase.get_comoving_radial_distance`.
        - ``sigma8_z={'z': [z_1, ...]}``: Amplitude of rms fluctuations
          :math:`\sigma_8` at the redshifts requested. Get it with
          :func:`~BoltzmannBase.get_sigma8`.
        - ``fsigma8={'z': [z_1, ...]}``: Structure growth rate
          :math:`f\sigma_8` at the redshifts requested. Get it with
          :func:`~BoltzmannBase.get_fsigma8`.
        - **Other derived parameters** that are not included in the input but whose
          value the likelihood may need.

        """
        super().must_provide(**requirements)
        self._must_provide = self._must_provide or dict.fromkeys(self.output_params or [])
        # Accumulate the requirements across several calls in a safe way;
        # e.g. take maximum of all values of a requested precision parameter
        for k, v in requirements.items():
            # Products and other computations
            if k == "Cl" or k == "lensed_scal_Cl" or k == "unlensed_Cl":
                current = self._must_provide.get(k, {})
                v = {cl.lower(): v[cl] for cl in v}  # to lowercase
                self._must_provide[k] = {
                    cl.lower(): max(current.get(cl.lower(), 0), v.get(cl, 0))
                    for cl in set(current).union(v)
                }
            elif k == "sigma_R":
                self._check_args(k, v, ("z", "R"))
                for pair in self._norm_vars_pairs(v.pop("vars_pairs", []), k):
                    k = ("sigma_R",) + pair
                    current = self._must_provide.get(k, {})
                    self._must_provide[k] = {
                        "R": combine_1d(v["R"], current.get("R")),
                        "z": combine_1d(v["z"], current.get("z")),
                        "k_max": max(
                            current.get("k_max", 0), v.get("k_max", 2 / np.min(v["R"]))
                        ),
                    }
            elif k in ("Pk_interpolator", "Pk_grid"):
                # arguments are all identical, collect all in Pk_grid
                self._check_args(k, v, ("z", "k_max"))
                redshifts = v.pop("z")
                k_max = v.pop("k_max")
                nonlin = v.pop("nonlinear", True)
                if not isinstance(nonlin, Iterable):
                    nonlin = [nonlin]
                for var_pair in self._norm_vars_pairs(v.pop("vars_pairs", []), k):
                    for nonlinear in nonlin:
                        k = ("Pk_grid", bool(nonlinear)) + var_pair
                        current = self._must_provide.get(k, {})
                        self._must_provide[k] = dict(
                            nonlinear=nonlinear,
                            z=combine_1d(redshifts, current.get("z")),
                            k_max=max(current.get("k_max", 0), k_max),
                            **v,
                        )
            elif k == "source_Cl":
                if k not in self._must_provide:
                    self._must_provide[k] = {}
                if "sources" not in v:
                    raise LoggedError(
                        self.log,
                        "Needs a 'sources' key, containing a dict with every "
                        "source name and definition",
                    )
                # Check that no two sources with equal name but diff specification
                # for source, window in v["sources"].items():
                #     if source in (getattr(self, "sources", {}) or {}):
                #         # TODO: improve this test!!!
                #         # (2 z-vectors that fulfill np.allclose would fail a == test)
                #         if window != self.sources[source]:
                #             raise LoggedError(
                #                 self.log,
                #                 "Source %r requested twice with different specification"
                #                 ": %r vs %r.", source, window, self.sources[source])
                self._must_provide[k].update(v)
            elif k in {
                "Hubble",
                "Omega_b",
                "Omega_cdm",
                "Omega_nu_massive",
                "angular_diameter_distance",
                "comoving_radial_distance",
                "sigma8_z",
                "fsigma8",
            }:
                if k not in self._must_provide:
                    self._must_provide[k] = {}
                if not isinstance(v, Mapping) or "z" not in v:
                    raise LoggedError(
                        self.log,
                        f"The value in the dictionary of requisites {k} must be a "
                        "dictionary containing the key 'z' with a list of redshifts "
                        f"(got instead {{{k}: {v}}})",
                    )
                self._must_provide[k]["z"] = combine_1d(
                    v["z"], self._must_provide[k].get("z")
                )
            elif k == "angular_diameter_distance_2":
                if k not in self._must_provide:
                    self._must_provide[k] = {}
                zs = v.pop("z_pairs", None)
                # Combine pairs with previous ones and uniquify
                try:
                    self._must_provide[k]["z_pairs"] = combine_2d(
                        zs, self._must_provide[k].get("z_pairs")
                    )
                except ValueError:
                    raise LoggedError(
                        self.log,
                        "For requisite `angular_diameter_distance2`, `z_pairs` "
                        "must be a list of pairs (z1, z2), but is "
                        f"{v['z_pairs']}",
                    )
            # Extra derived parameters and other unknown stuff (keep capitalization)
            elif v is None:
                self._must_provide[k] = None
            else:
                raise LoggedError(self.log, "Unknown required product: '%s'.", k)

    def requested(self):
        """
        Returns the full set of requested cosmological products and parameters.
        """
        return self._must_provide

    def check_no_repeated_input_extra(self):
        """
        Checks that there are no repeated parameters between input and extra.

        Should be called at initialisation, and at the end of every call to must_provide()
        """
        if common := set(self.input_params).intersection(self.extra_args):
            raise LoggedError(
                self.log,
                "The following parameters appear both as input parameters and "
                "as extra arguments: %s. Please, remove one of the definitions "
                "of each.",
                common,
            )

    def _get_z_dependent(self, quantity, z, pool=None):
        if pool is None:
            pool = self.collectors[quantity].z_pool
        try:
            i_kwarg_z = pool.find_indices(z)
        except ValueError:
            raise LoggedError(
                self.log,
                f"{quantity} not computed for all z requested. "
                f"Requested z are {z}, but computed ones are "
                f"{pool.values}.",
            )
        return np.array(self.current_state[quantity], copy=True)[i_kwarg_z]

    def _get_z_pair_dependent(self, quantity, z_pairs, inv_value=0):
        """
        ``inv_value`` (default=0) is assigned to pairs for which ``z1 > z2``.
        """
        try:
            check_2d(z_pairs, allow_1d=False)
        except ValueError:
            raise LoggedError(
                self.log,
                f"z_pairs={z_pairs} not correctly formatted for "
                f"{quantity}. It should be a list of pairs.",
            )
        # Only recover for correctly sorted pairs
        z_pairs_arr = np.array(z_pairs)
        i_right = z_pairs_arr[:, 0] <= z_pairs_arr[:, 1]
        pool = self.collectors[quantity].z_pool
        try:
            i_z_pair = pool.find_indices(z_pairs_arr[i_right])
        except ValueError:
            raise LoggedError(
                self.log,
                f"{quantity} not computed for all z pairs requested. "
                f"Requested z are {z_pairs}, but computed ones are "
                f"{pool.values}.",
            )
        result = np.full(len(z_pairs), inv_value, dtype=float)
        result[i_right] = np.array(self.current_state[quantity], copy=True)[i_z_pair]
        return result

    def _cmb_unit_factor(self, units, T_cmb):
        units_factors = {
            "1": 1,
            "muK2": T_cmb * 1.0e6,
            "K2": T_cmb,
            "FIRASmuK2": 2.7255e6,
            "FIRASK2": 2.7255,
        }
        try:
            return units_factors[units]
        except KeyError:
            raise LoggedError(
                self.log,
                "Units '%s' not recognized. Use one of %s.",
                units,
                list(units_factors),
            )

    @abstract
    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        r"""
        Returns a dictionary of lensed total CMB power spectra and the
        lensing potential ``pp`` power spectrum.

        Set the units with the keyword ``units=number|'muK2'|'K2'|'FIRASmuK2'|'FIRASK2'``.
        The default is ``FIRASmuK2``, which returns CMB :math:`C_\ell`'s in
        FIRAS-calibrated :math:`\mu K^2`, i.e. scaled by a fixed factor of
        :math:`(2.7255\cdot 10^6)^2` (except for the lensing potential power spectrum,
        which is always unitless).
        The ``muK2`` and ``K2`` options use the model's CMB temperature.

        If ``ell_factor=True`` (default: ``False``), multiplies the spectra by
        :math:`\ell(\ell+1)/(2\pi)` (or by :math:`[\ell(\ell+1)]^2/(2\pi)` in the case of
        the lensing potential ``pp`` spectrum, and :math:`[\ell(\ell+1)]^{3/2}/(2\pi)` for
        the cross spectra ``tp`` and ``ep``).
        """

    @abstract
    def get_unlensed_Cl(self, ell_factor=False, units="FIRASmuK2"):
        r"""
        Returns a dictionary of unlensed CMB power spectra.

        For ``units`` options, see :func:`~BoltzmannBase.get_Cl`.

        If ``ell_factor=True`` (default: ``False``), multiplies the spectra by
        :math:`\ell(\ell+1)/(2\pi)`.
        """

    @abstract
    def get_lensed_scal_Cl(self, ell_factor=False, units="FIRASmuK2"):
        r"""
        Returns a dictionary of lensed scalar CMB power spectra and the lensing
        potential ``pp`` power spectrum.

        For ``units`` options, see :func:`~BoltzmannBase.get_Cl`.

        If ``ell_factor=True`` (default: ``False``), multiplies the spectra by
        :math:`\ell(\ell+1)/(2\pi)`.
        """

    def get_Hubble(self, z, units="km/s/Mpc"):
        r"""
        Returns the Hubble rate at the given redshift(s) ``z``.

        The redshifts must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.
        The available units are ``"km/s/Mpc"`` (i.e. :math:`cH(\mathrm(Mpc)^{-1})`) and
        ``1/Mpc``.
        """
        try:
            return self._get_z_dependent("Hubble", z) * H_units_conv_factor[units]
        except KeyError:
            raise LoggedError(
                self.log,
                "Units not known for H: '%s'. Try instead one of %r.",
                units,
                list(H_units_conv_factor),
            )

    def get_Omega_b(self, z):
        r"""
        Returns the Baryon density parameter at the given redshift(s) ``z``.

        The redshifts must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.
        """
        return self._get_z_dependent("Omega_b", z)

    def get_Omega_cdm(self, z):
        r"""
        Returns the Cold Dark Matter density parameter at the given redshift(s) ``z``.

        The redshifts must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.
        """
        return self._get_z_dependent("Omega_cdm", z)

    def get_Omega_nu_massive(self, z):
        r"""
        Returns the Massive neutrinos' density parameter at the given redshift(s) ``z``.

        The redshifts must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.
        """
        return self._get_z_dependent("Omega_nu_massive", z)

    def get_angular_diameter_distance(self, z):
        r"""
        Returns the physical angular diameter distance in :math:`\mathrm{Mpc}` to the
        given redshift(s) ``z``.

        The redshifts must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.
        """
        return self._get_z_dependent("angular_diameter_distance", z)

    def get_angular_diameter_distance_2(self, z_pairs):
        r"""
        Returns the physical angular diameter distance between pairs of redshifts
        `z_pairs` in :math:`\mathrm{Mpc}`.

        The redshift pairs must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.

        Return zero for pairs in which ``z1 > z2``.
        """
        return self._get_z_pair_dependent(
            "angular_diameter_distance_2", z_pairs, inv_value=0
        )

    def get_comoving_radial_distance(self, z):
        r"""
        Returns the comoving radial distance in :math:`\mathrm{Mpc}` to the given
        redshift(s) ``z``.

        The redshifts must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.
        """
        return self._get_z_dependent("comoving_radial_distance", z)

    def get_Pk_interpolator(
        self,
        var_pair=("delta_tot", "delta_tot"),
        nonlinear=True,
        extrap_kmin=None,
        extrap_kmax=None,
    ):
        r"""
        Get a :math:`P(z,k)` bicubic interpolation object
        (:class:`PowerSpectrumInterpolator`).

        In the interpolator returned, the input :math:`k` and resulting
        :math:`P(z,k)` are in units of :math:`1/\mathrm{Mpc}` and
        :math:`\mathrm{Mpc}^3` respectively (not in :math:`h^{-1}` units).

        :param var_pair: variable pair for power spectrum
        :param nonlinear: non-linear spectrum (default True)
        :param extrap_kmin: use log linear extrapolation from ``extrap_kmin`` up to min
                            :math:`k`.
        :param extrap_kmax: use log linear extrapolation beyond max :math:`k` computed up
                            to ``extrap_kmax``.
        :return: :class:`PowerSpectrumInterpolator` instance.
        """
        nonlinear = bool(nonlinear)
        key = ("Pk_interpolator", nonlinear, extrap_kmin, extrap_kmax) + tuple(
            sorted(var_pair)
        )
        if key in self.current_state:
            return self.current_state[key]
        k, z, pk = self.get_Pk_grid(var_pair=var_pair, nonlinear=nonlinear)
        log_p = True
        sign = 1
        if np.any(pk < 0):
            if np.all(pk < 0):
                sign = -1
            else:
                log_p = False
        extrapolating = (extrap_kmax and extrap_kmax > k[-1]) or (
            extrap_kmin and extrap_kmin < k[0]
        )
        if log_p:
            pk = np.log(sign * pk)
        elif extrapolating:
            raise LoggedError(
                self.log,
                "Cannot do log extrapolation with zero-crossing pk for %s, %s" % var_pair,
            )
        result = PowerSpectrumInterpolator(
            z,
            k,
            pk,
            logP=log_p,
            logsign=sign,
            extrap_kmin=extrap_kmin,
            extrap_kmax=extrap_kmax,
        )
        self.current_state[key] = result
        return result

    def get_Pk_grid(self, var_pair=("delta_tot", "delta_tot"), nonlinear=True):
        r"""
        Get  matter power spectrum, e.g. suitable for splining.
        Returned arrays may be bigger or more densely sampled than requested, but will
        include required values.

        In the grid returned, :math:`k` and :math:`P(z,k)` are in units of
        :math:`1/\mathrm{Mpc}` and :math:`\mathrm{Mpc}^3` respectively
        (not in :math:`h^{-1}` units), and :math:`z` and :math:`k` are in
        **ascending** order.

        :param nonlinear: whether the linear or non-linear spectrum
        :param var_pair: which power spectrum
        :return: ``k``, ``z``, ``Pk``, where ``k`` and ``z`` are 1-d arrays,
                 and the 2-d array ``Pk[i,j]`` is the value of :math:`P(z,k)` at ``z[i]``,
                 ``k[j]``.
        """
        try:
            return self.current_state[
                ("Pk_grid", bool(nonlinear)) + tuple(sorted(var_pair))
            ]
        except KeyError:
            if ("Pk_grid", False) + tuple(sorted(var_pair)) in self.current_state:
                raise LoggedError(
                    self.log,
                    "Getting non-linear matter power but 'nonlinear' "
                    "not specified in requirements",
                )
            raise LoggedError(self.log, "Matter power %s, %s not computed" % var_pair)

    def get_sigma_R(self, var_pair=("delta_tot", "delta_tot")):
        r"""
        Get :math:`\sigma_R(z)`, the RMS power in a sphere of radius :math:`R` at
        redshift :math:`z`.

        Note that :math:`R` is in :math:`\mathrm{Mpc}`, not :math:`h^{-1}\,\mathrm{Mpc}`,
        and :math:`z` and :math:`R` are returned in **ascending** order.

        You may get back more values than originally requested, but the requested
        :math:`R` and :math:`z` should be in the returned arrays.

        :param var_pair: which two fields to use for the RMS power
        :return: ``z``, ``R``, ``sigma_R``, where ``z`` and ``R`` are arrays of computed
                 values, and ``sigma_R[i,j]`` is the value :math:`\sigma_R(z)` for
                 ``z[i]``, ``R[j]``.
        """
        try:
            return self.current_state[("sigma_R",) + tuple(sorted(var_pair))]
        except KeyError:
            raise LoggedError(self.log, "sigmaR %s not computed" % var_pair)

    @abstract
    def get_source_Cl(self):
        r"""
        Returns a dict of power spectra of for the computed sources, with keys a tuple of
        sources ``([source1], [source2])``, and an additional key ``ell`` containing the
        multipoles.
        """

    def get_sigma8_z(self, z):
        r"""
        Present day linear theory root-mean-square amplitude of the matter
        fluctuation spectrum averaged in spheres of radius 8 h^{−1} Mpc.

        The redshifts must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.
        """
        return self._get_z_dependent("sigma8_z", z)

    def get_fsigma8(self, z):
        r"""
        Structure growth rate :math:`f\sigma_8`, as defined in eq. 33 of
        `Planck 2015 results. XIII.
        Cosmological parameters <https://arxiv.org/pdf/1502.01589.pdf>`_,
        at the given redshift(s) ``z``.

        The redshifts must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.
        """
        return self._get_z_dependent("fsigma8", z)

    def get_auto_covmat(self, params_info, likes_info):
        r"""
        Tries to get match to a database of existing covariance matrix files for the
        current model and data.

        ``params_info`` should contain preferably the slow parameters only.
        """
        from cobaya.cosmo_input import get_best_covmat_ext, get_covmat_package_folders

        return get_best_covmat_ext(
            get_covmat_package_folders(self.packages_path), params_info, likes_info
        )


class PowerSpectrumInterpolator(RectBivariateSpline):
    r"""
    2D spline interpolation object (scipy.interpolate.RectBivariateSpline)
    to evaluate matter power spectrum as function of z and k.

    *This class is adapted from CAMB's own P(k) interpolator, by Antony Lewis;
    it's mostly interface-compatible with the original.*

    :param z: values of z for which the power spectrum was evaluated.
    :param k: values of k for which the power spectrum was evaluated.
    :param P_or_logP: Values of the power spectrum (or log-values, if logP=True).
    :param logP: if True (default: False), log of power spectrum are given and used
        for the underlying interpolator.
    :param logsign: if logP is True, P_or_logP is log(logsign*Pk)
    :param extrap_kmax: if set, use power law extrapolation beyond kmax up to
        extrap_kmax; useful for tails of integrals.
    """

    def __init__(
        self, z, k, P_or_logP, extrap_kmin=None, extrap_kmax=None, logP=False, logsign=1
    ):
        self.islog = logP
        #  Check order
        z, k = (np.atleast_1d(x) for x in [z, k])
        if len(z) < 4:
            raise ValueError(
                "Require at least four redshifts for Pk interpolation."
                "Consider using Pk_grid if you just need a small number"
                "of specific redshifts (doing 1D splines in k yourself)."
            )
        z, k, P_or_logP = np.array(z), np.array(k), np.array(P_or_logP)
        i_z = np.argsort(z)
        i_k = np.argsort(k)
        self.logsign = logsign
        self.z, self.k, P_or_logP = z[i_z], k[i_k], P_or_logP[i_z, :][:, i_k]
        self.zmin, self.zmax = self.z[0], self.z[-1]
        self.extrap_kmin, self.extrap_kmax = extrap_kmin, extrap_kmax
        logk = np.log(self.k)
        # Start from extrap_kmin using a (log,log)-linear extrapolation
        if extrap_kmin and extrap_kmin < self.input_kmin:
            if not logP:
                raise ValueError("extrap_kmin must use logP")
            logk = np.hstack(
                [
                    np.log(extrap_kmin),
                    np.log(self.input_kmin) * 0.1 + np.log(extrap_kmin) * 0.9,
                    logk,
                ]
            )
            logPnew = np.empty((P_or_logP.shape[0], P_or_logP.shape[1] + 2))
            logPnew[:, 2:] = P_or_logP
            diff = (logPnew[:, 3] - logPnew[:, 2]) / (logk[3] - logk[2])
            delta = diff * (logk[2] - logk[0])
            logPnew[:, 0] = logPnew[:, 2] - delta
            logPnew[:, 1] = logPnew[:, 2] - delta * 0.9
            P_or_logP = logPnew
        # Continue until extrap_kmax using a (log,log)-linear extrapolation
        if extrap_kmax and extrap_kmax > self.input_kmax:
            if not logP:
                raise ValueError("extrap_kmax must use logP")
            logk = np.hstack(
                [
                    logk,
                    np.log(self.input_kmax) * 0.1 + np.log(extrap_kmax) * 0.9,
                    np.log(extrap_kmax),
                ]
            )
            logPnew = np.empty((P_or_logP.shape[0], P_or_logP.shape[1] + 2))
            logPnew[:, :-2] = P_or_logP
            diff = (logPnew[:, -3] - logPnew[:, -4]) / (logk[-3] - logk[-4])
            delta = diff * (logk[-1] - logk[-3])
            logPnew[:, -1] = logPnew[:, -3] + delta
            logPnew[:, -2] = logPnew[:, -3] + delta * 0.9
            P_or_logP = logPnew
        super().__init__(self.z, logk, P_or_logP)

    @property
    def input_kmin(self):
        """Minimum k for the interpolation (not incl. extrapolation range)."""
        return self.k[0]

    @property
    def input_kmax(self):
        """Maximum k for the interpolation (not incl. extrapolation range)."""
        return self.k[-1]

    @property
    def kmin(self):
        """Minimum k of the interpolator (incl. extrapolation range)."""
        if self.extrap_kmin is None:
            return self.input_kmin
        return self.extrap_kmin

    @property
    def kmax(self):
        """Maximum k of the interpolator (incl. extrapolation range)."""
        if self.extrap_kmax is None:
            return self.input_kmax
        return self.extrap_kmax

    def check_ranges(self, z, k):
        """Checks that we are not trying to extrapolate beyond the interpolator limits."""
        z = np.atleast_1d(z).flatten()
        min_z, max_z = min(z), max(z)
        if min_z < self.zmin and not np.allclose(min_z, self.zmin):
            raise LoggedError(
                get_logger(self.__class__.__name__),
                f"Not possible to extrapolate to z={min(z)} "
                f"(minimum z computed is {self.zmin}).",
            )
        if max_z > self.zmax and not np.allclose(max_z, self.zmax):
            raise LoggedError(
                get_logger(self.__class__.__name__),
                f"Not possible to extrapolate to z={max(z)} "
                f"(maximum z computed is {self.zmax}).",
            )
        k = np.atleast_1d(k).flatten()
        min_k, max_k = min(k), max(k)
        if min_k < self.kmin and not np.allclose(min_k, self.kmin):
            raise LoggedError(
                get_logger(self.__class__.__name__),
                f"Not possible to extrapolate to k={min(k)} 1/Mpc "
                f"(minimum k possible is {self.kmin} 1/Mpc).",
            )
        if max_k > self.kmax and not np.allclose(max_k, self.kmax):
            raise LoggedError(
                get_logger(self.__class__.__name__),
                f"Not possible to extrapolate to k={max(k)} 1/Mpc "
                f"(maximum k possible is {self.kmax} 1/Mpc).",
            )

    def P(self, z, k, grid=None):
        """
        Get the power spectrum at (z,k).
        """
        self.check_ranges(z, k)
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self.logsign * np.exp(self(z, np.log(k), grid=grid, warn=False))
        else:
            return self(z, np.log(k), grid=grid, warn=False)

    def logP(self, z, k, grid=None):
        """
        Get the log power spectrum at (z,k). (or minus log power spectrum if
        islog and logsign=-1)
        """
        self.check_ranges(z, k)
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self(z, np.log(k), grid=grid, warn=False)
        else:
            return np.log(self(z, np.log(k), grid=grid, warn=False))

    def __call__(self, *args, warn=True, **kwargs):
        if warn:
            get_logger(self.__class__.__name__).warning(
                "Do not call the instance directly. Use instead methods P(z, k) or "
                "logP(z, k) to get the (log)power spectrum. (If you know what you are "
                "doing, pass warn=False)"
            )
        return super().__call__(*args, **kwargs)
