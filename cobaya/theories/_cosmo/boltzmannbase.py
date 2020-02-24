"""
.. module:: BoltzmannBase

:Synopsis: Template for Cosmological theory codes.
           Mostly here to document how to compute and get observables.
:Author: Jesus Torrado

"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from itertools import chain
from typing import Mapping

# Local
from cobaya.theory import Theory
from cobaya.tools import deepcopy_where_possible
from cobaya.log import LoggedError
from cobaya.conventions import _c_km_s, empty_dict

H_units_conv_factor = {"1/Mpc": 1, "km/s/Mpc": _c_km_s}


class BoltzmannBase(Theory):
    _get_z_dependent: callable  # defined by inheriting classes
    renames: Mapping[str, str] = empty_dict

    def initialize(self):

        # Dict of named tuples to collect requirements and computation methods
        self.collectors = {}
        # Additional input parameters to pass to CAMB, and attributes to set_ manually
        self.extra_args = deepcopy_where_possible(self.extra_args) or {}
        self._needs = None

    def get_allow_agnostic(self):
        return True

    def translate_param(self, p):
        return self.renames.get(p, p)

    def get_param(self, p):
        translated = self.translate_param(p)
        for pool in ["params", "derived", "derived_extra"]:
            value = (self._current_state[pool] or {}).get(translated, None)
            if value is not None:
                return value

        raise LoggedError(self.log, "Parameter not known: '%s'", p)

    def needs(self, **requirements):
        r"""
        Specifies the quantities that each likelihood needs from the Cosmology code.

        Typical requisites in Cosmology (as keywords, case insensitive):

        - ``Cl={...}``: CMB lensed power spectra, as a dictionary ``{spectrum:l_max}``,
          where the possible spectra are combinations of "t", "e", "b" and "p"
          (lensing potential). Get with :func:`~BoltzmannBase.get_Cl`.
        - **[BETA: CAMB only; notation may change!]** ``source_Cl={...}``:
          :math:`C_\ell` of given sources with given windows, e.g.:
          ``source_name: {"function": "spline"|"gaussian", [source_args]``;
          for now, ``[source_args]`` follow the notation of ``CAMBSources``.
          If can also take ``lmax: [int]``, ``limber: True`` if Limber approximation
          desired, and ``non_linear: True`` if non-linear contributions requested.
          Get with :func:`~BoltzmannBase.get_source_Cl`.
        - ``Pk_interpolator={...}``: Matter power spectrum interpolator in :math:`(z, k)`.
          Takes ``"z": [list_of_evaluated_redshifts]``, ``"k_max": [k_max]``,
          ``"extrap_kmax": [max_k_max_extrapolated]``, ``"nonlinear": [True|False]``,
          ``"vars_pairs": [["delta_tot", "delta_tot"], ["Weyl", "Weyl"], [...]]}``.
          Non-linear contributions are included by default.
          All ``k`` values should be in units of ``1/Mpc``.
        - ``Pk_grid={...}``: similar to Pk_interpolator except that rather than returning
          a bicuplic spline object it returns the raw power spectrum grid as a (k, z, PK)
          set of arrays.
        - ``sigma_R{...}``: RMS linear fluctuation in spheres of radius R at redshifts z.
          Takes ``"z": [list_of_evaluated_redshifts]``, ``"k_max": [k_max]``,
          ``"vars_pairs": [["delta_tot", "delta_tot"],  [...]]}``,
          ``"R": [list_of_evaluated_R]``.
          Note that R is in Mpc, not h^{-1} Mpc.
        - ``Hubble={'z': [z_1, ...], 'units': '1/Mpc' or 'km/s/Mpc'}``: Hubble
          rate at the redshifts requested, in the given units. Get it with
          :func:`~BoltzmannBase.get_Hubble`.
        - ``angular_diameter_distance={'z': [z_1, ...]}``: Physical angular
          diameter distance to the redshifts requested. Get it with
          :func:`~BoltzmannBase.get_angular_diameter_distance`.
        - ``comoving_radial_distance={'z': [z_1, ...]}``: Comoving radial distance
          from us to the redshifts requested. Get it with
          :func:`~BoltzmannBase.get_comoving_radial_distance`.
        - ``fsigma8={'z': [z_1, ...]}``: Structure growth rate
          :math:`f\sigma_8` at the redshifts requested. Get it with
          :func:`~BoltzmannBase.get_fsigma8`.
        - ``k_max=[...]``: Fixes the maximum comoving wavenumber considered.
        - **Other derived parameters** that are not included in the input but whose
          value the likelihood may need.
        """

        super().needs(**requirements)

        self._needs = self._needs or {p: None for p in self.output_params}
        # TO BE DEPRECATED IN >=1.3
        for product, capitalization in {
            "cl": "Cl", "pk_interpolator": "Pk_interpolator"}.items():
            if product in requirements:
                raise LoggedError(
                    self.log, "You requested product '%s', which from now on should be "
                              "capitalized as '%s'.", product, capitalization)
        # Accumulate the requirements across several calls in a safe way;
        # e.g. take maximum of all values of a requested precision parameter
        for k, v in requirements.items():
            # Products and other computations
            if k == "Cl":
                self._needs["Cl"] = {
                    cl: max(self._needs.get("Cl", {}).get(cl, 0), v.get(cl, 0))
                    for cl in set(self._needs.get("Cl", {})).union(v)}
            elif k in ["Pk_interpolator", "Pk_grid", "sigma_R"]:
                # TODO: could make Pk_interpolator to Pk_grid here, then only PK_grid in
                # camb/classy?
                # Make sure vars_pairs is a list of [iterable of 2 vars pairs]
                vars_pairs = v.pop("vars_pairs", [])
                try:
                    if isinstance(vars_pairs[0], str):
                        vars_pairs = [vars_pairs]
                except IndexError:
                    # Empty list: default to *total matter*: CMB + Baryon + MassiveNu
                    vars_pairs = [2 * ["delta_tot"]]
                except:
                    raise LoggedError(self.log,
                                      "Cannot understand vars_pairs '%r' for %s",
                                      vars_pairs, k)
                vars_pairs = set(tuple(sorted(pair)) for pair in chain(
                    self._needs.get(k, {}).get("vars_pairs", []), vars_pairs))
                kmax = self._needs.get(k, {}).get("k_max", 0)
                if k == "sigma_R":
                    v["R"] = np.sort(np.unique(np.concatenate(
                        (self._needs.get(k, {}).get("R", []), np.atleast_1d(v["R"])))))
                    kmax = max(kmax, v.get("k_max", 2 / np.min(v["R"])))
                else:
                    v["nonlinear"] = bool(v.get("nonlinear", True))
                    kmax = max(kmax, v["k_max"])
                self._needs[k] = {
                    "z": np.unique(np.concatenate(
                        (self._needs.get(k, {}).get("z", []),
                         np.atleast_1d(v["z"])))),
                    "k_max": kmax,
                    "vars_pairs": vars_pairs}
                self._needs[k].update(v)
            elif k == "source_Cl":
                if k not in self._needs:
                    self._needs[k] = {}
                if "sources" not in v:
                    raise LoggedError(
                        self.log, "Needs a 'sources' key, containing a dict with every "
                                  "source name and definition")
                # Check that no two sources with equal name but diff specification
                # TODO: commented, what is self.sources?
                # for source, window in v["sources"].items():
                #     if source in (getattr(self, "sources", {}) or {}):
                #         # TODO: improve this test!!!
                #         # (e.g. 2 z-vectors that fulfill np.allclose would fail a == test)
                #         if window != self.sources[source]:
                #             raise LoggedError(
                #                 self.log,
                #                 "Source %r requested twice with different specification: "
                #                 "%r vs %r.", window, self.sources[source])
                self._needs[k].update(v)
            elif k in ["Hubble", "angular_diameter_distance",
                       "comoving_radial_distance", "fsigma8"]:
                if k not in self._needs:
                    self._needs[k] = {}
                self._needs[k]["z"] = np.unique(np.concatenate(
                    (self._needs[k].get("z", []), v["z"])))
            # Extra derived parameters and other unknown stuff (keep capitalization)
            elif v is None:
                self._needs[k] = None
            else:
                raise LoggedError(self.log, "Unknown required product: '%s'.", k)

    def requested(self):
        """
        Returns the full set of requested cosmological products and parameters.
        """
        return self._needs

    def get_Cl(self, ell_factor=False, units="muK2"):
        r"""
        Returns a dictionary of lensed CMB power spectra and the lensing potential ``pp``
        power spectrum.

        Set the units with the keyword ``units='1'|'muK2'|'K2'`` (default: 'muK2',
        except for the lensing potential power spectrum, which is always unitless).

        If ``ell_factor=True`` (default: False), multiplies the spectra by
        :math:`\ell(\ell+1)/(2\pi)` (or by :math:`\ell^2(\ell+1)^2/(2\pi)` in the case of
        the lensing potential ``pp`` spectrum).
        """
        pass

    def get_Hubble(self, z, units="km/s/Mpc"):
        r"""
        Returns the Hubble rate at the given redshifts.

        The redshifts must be a subset of those requested when :func:`~BoltzmannBase.needs`
        was called.

        The available units are ``km/s/Mpc`` (i.e. ``c*H(Mpc^-1)``) and ``1/Mpc``.
        """
        try:
            return self._get_z_dependent("Hubble", z) * H_units_conv_factor[units]
        except KeyError:
            raise LoggedError(
                self.log, "Units not known for H: '%s'. Try instead one of %r.",
                units, list(H_units_conv_factor))

    def get_angular_diameter_distance(self, z):
        r"""
        Returns the physical angular diameter distance to the given redshifts in Mpc.

        The redshifts must be a subset of those requested when :func:`~BoltzmannBase.needs`
        was called.
        """
        return self._get_z_dependent("angular_diameter_distance", z)

    def get_comoving_radial_distance(self, z):
        r"""
        Returns the comoving radial distance to the given redshifts in Mpc.

        The redshifts must be a subset of those requested when :func:`~BoltzmannBase.needs`
        was called.
        """
        return self._get_z_dependent("comoving_radial_distance", z)

    def get_Pk_grid(self, var_pair=("delta_tot", "delta_tot"), nonlinear=True):
        """
        Get  matter power spectrum, e.g. suitable for splining.
        Returned arrays may be bigger or more densely sampled than requested, but will
        include required values. Neither k nor PK are in h^{-1} units.
        z and k are in ascending order.

        :param nonlinear: whether the linear or nonlinear spectrum
        :param var_pair: which power spectrum
        :return: k, z, PK, where k and z are arrays,
                 and PK[i,j] is the value at z[i], k[j]
        """
        try:
            return self._current_state[
                ("Pk_grid", bool(nonlinear)) + tuple(sorted(var_pair))]
        except KeyError:
            if ("Pk_grid", False) + tuple(sorted(var_pair)) in self._current_state:
                raise LoggedError(self.log,
                                  "Getting non-linear matter power but nonlinear "
                                  "not specified in requirements")
            raise LoggedError(self.log, "Matter power %s, %s not computed" % var_pair)

    def get_Pk_interpolator(self, var_pair=("delta_tot", "delta_tot"), nonlinear=True,
                            extrap_kmax=None):
        """
        Get P(z,k) bicubic interpolation object (:class:`PowerSpectrumInterpolator`).
        Neither k nor PK are in h^{-1} units.

        :param var_pair: variable pair for power spectrum
        :param nonlinear: non-linear spectrum (default True)
        :param extrap_kmax: use log linear extrapolation beyond max k computed up to
                            extrap_kmax
        :return: :class:`PowerSpectrumInterpolator` instance.
        """
        nonlinear = bool(nonlinear)
        key = ("Pk_interpolator", nonlinear, extrap_kmax) + tuple(sorted(var_pair))
        if key in self._current_state:
            return self._current_state[key]
        k, z, pk = self.get_Pk_grid(var_pair=var_pair, nonlinear=nonlinear)
        log_p = True
        sign = 1
        if np.any(pk < 0):
            if np.all(pk < 0):
                sign = -1
            else:
                log_p = False
        if log_p:
            pk = np.log(sign * pk)
        elif extrap_kmax > k[-1]:
            raise LoggedError(self.log,
                              'Cannot do log extrapolation with zero-crossing pk '
                              'for %s, %s' % var_pair)
        result = PowerSpectrumInterpolator(z, k, pk, logP=log_p, logsign=sign,
                                           extrap_kmax=extrap_kmax)
        self._current_state[key] = result
        return result

    def get_sigma_R(self, var_pair=("delta_tot", "delta_tot")):
        """
        Get sigma(R), the RMS power in an sphere of radius R
        Note R is in Mpc not h^{-1}Mpc units and z and R are returned in descending and
        ascending order respectively.

        You may get back more values than originally requested, but requested R and z
        should in the returned arrays.

        :param var_pair: which two fields to use for the RMS power
        :return: R, z, sigma_R, where R and z are arrays of computed values,
                 and sigma_R[i,j] is the value for z[i], R[j]
        """
        try:
            return self._current_state[("sigma_R",) + tuple(sorted(var_pair))]
        except KeyError:
            raise LoggedError(self.log, "sigmaR %s not computed" % var_pair)

    def get_source_Cl(self):
        r"""
        Returns a dict of power spectra of for the computed sources, with keys a tuple of
        sources ``([source1], [source2])``, and an additional key ``ell`` containing the
        multipoles.
        """

    def get_fsigma8(self, z):
        r"""
        Structure growth rate :math:`f\sigma_8`, as defined in eq. 33 of
        `Planck 2015 results. XIII. Cosmological parameters <https://arxiv.org/pdf/1502.01589.pdf>`_,
        at the given redshifts.

        The redshifts must be a subset of those requested when :func:`~BoltzmannBase.needs`
        was called.
        """
        pass

    def get_auto_covmat(self, params_info, likes_info):
        r"""
        Tries to get match to a database of existing covariance matrix files for the
        current model and data.

        ``params_info`` should contain preferably the slow parameters only.
        """
        from cobaya.cosmo_input import get_best_covmat
        return get_best_covmat(self.path_install, params_info, likes_info)


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

    def __init__(self, z, k, P_or_logP, extrap_kmax=None, logP=False, logsign=1):
        self.islog = logP
        #  Check order
        z, k = (np.atleast_1d(x) for x in [z, k])
        if len(z) < 3:
            raise ValueError('Require at least three redshifts for interpolation')
        i_z = np.argsort(z)
        i_k = np.argsort(k)
        self.logsign = logsign
        self.z, self.k, P_or_logP = z[i_z], k[i_k], P_or_logP[i_z, :][:, i_k]
        self.zmin, self.zmax = self.z[0], self.z[-1]
        self.kmin, self.kmax = self.k[0], self.k[-1]
        logk = np.log(self.k)
        # Continue until extrap_kmax using a (log,log)-linear extrapolation
        if extrap_kmax and extrap_kmax > self.kmax:
            if not logP:
                raise ValueError('extrap_kmax must use logP')
            logk = np.hstack(
                [logk, np.log(self.kmax) * 0.1 + np.log(extrap_kmax) * 0.9,
                 np.log(extrap_kmax)])
            logPnew = np.empty((P_or_logP.shape[0], P_or_logP.shape[1] + 2))
            logPnew[:, :-2] = P_or_logP
            diff = (logPnew[:, -3] - logPnew[:, -4]) / (logk[-3] - logk[-4])
            delta = diff * (logk[-1] - logk[-3])
            logPnew[:, -1] = logPnew[:, -3] + delta
            logPnew[:, -2] = logPnew[:, -3] + delta * 0.9
            self.kmax = extrap_kmax  # Added for consistency with CAMB

            P_or_logP = logPnew

        super().__init__(self.z, logk, P_or_logP)

    def P(self, z, k, grid=None):
        """
        Get the power spectrum at (z,k).
        """
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self.logsign * np.exp(self(z, np.log(k), grid=grid))
        else:
            return self(z, np.log(k), grid=grid)

    def logP(self, z, k, grid=None):
        """
        Get the log power spectrum at (z,k). (or minus log power spectrum if
        islog and logsign=-1)
        """
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self(z, np.log(k), grid=grid)
        else:
            return np.log(self(z, np.log(k), grid=grid))
