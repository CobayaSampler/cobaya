"""
.. module:: BoltzmannBase

:Synopsis: Template for Cosmological theory codes.
           Mostly here to document how to compute and get observables.
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import
import numpy as np
from scipy.interpolate import RectBivariateSpline
from six import string_types
from itertools import chain

# Local
from cobaya.theory import Theory
from cobaya.tools import fuzzy_match, create_banner, deepcopy_where_possible
from cobaya.log import LoggedError


class BoltzmannBase(Theory):

    def initialize(self):
        # Generate states, to avoid recomputing
        self._n_states = 3
        self._states = [
            {"params": None, "derived": None, "derived_extra": None, "last": 0}
            for _ in range(self._n_states)]
        # Dict of named tuples to collect requirements and computation methods
        self.collectors = {}
        # Additional input parameters to pass to CAMB, and attributes to set_ manually
        self.extra_args = deepcopy_where_possible(self.extra_args) or {}

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
        - ``H={'z': [z_1, ...], 'units': '1/Mpc' or 'km/s/Mpc'}``: Hubble
          rate at the redshifts requested, in the given units. Get it with
          :func:`~BoltzmannBase.get_H`.
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
        if not getattr(self, "_needs", None):
            self._needs = dict([(p, None) for p in self.output_params])
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
            elif k == "Pk_interpolator":
                # Make sure vars_pairs is a list of [list of 2 vars pairs]
                vars_pairs = v.pop("vars_pairs", [])
                try:
                    if isinstance(vars_pairs[0], string_types):
                        vars_pairs = [vars_pairs]
                except IndexError:
                    # Empty list: by default [delta_tot, delta_tot]
                    vars_pairs = [2 * ["delta_tot"]]
                except:
                    raise LoggedError(
                        self.log,
                        "Cannot understands vars_pairs '%r' for P(k) interpolator",
                        vars_pairs)
                vars_pairs = set([tuple(pair) for pair in chain(
                    self._needs.get(k, {}).get("vars_pairs", []), vars_pairs)])
                self._needs[k] = {
                    "z": np.unique(np.concatenate(
                        (self._needs.get(k, {}).get("z", []),
                         np.atleast_1d(v["z"])))),
                    "k_max": max(
                        self._needs.get(k, {}).get("k_max", 0), v["k_max"]),
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
                for source, window in v["sources"].items():
                    if source in (getattr(self, "sources", {}) or {}):
                        # TODO: improve this test!!!
                        # (e.g. 2 z-vectors that fulfill np.allclose would fail a == test)
                        if window != self.sources[source]:
                            raise LoggedError(
                                self.log,
                                "Source %r requested twice with different specification: "
                                "%r vs %r.", window, self.sources[source])
                self._needs[k].update(v)
            elif k in ["H", "angular_diameter_distance",
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
            if not self.run_calculation(_derived, i_state, **params_values_dict):
                return 0
            if self.timer:
                self.timer.increment(self.log)

        # make this one the current one by decreasing the antiquity of the rest
        for i in range(self._n_states):
            self._states[i]["last"] -= max(lasts)
        self._states[i_state]["last"] = 1
        return 1 if reused_state else 2

    def requested(self):
        """
        Returns the full set of cosmological products and parameters requested from
        anywhere.
        """
        return self._needs

    def get_param(self, p):
        """
        Interface function for likelihoods to get sampled and derived parameters.

        Always use this one; don't try to access theory code attributes directly!
        """
        pass

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

    def get_H(self, z, units="km/s/Mpc"):
        r"""
        Returns the Hubble rate at the given redshifts.

        The redshifts must be a subset of those requested when :func:`~BoltzmannBase.needs`
        was called.

        The available units are ``km/s/Mpc`` (i.e. ``c*H(Mpc^-1)``) and ``1/Mpc``.
        """
        try:
            return self._get_z_dependent("H", z) * self.H_units_conv_factor[units]
        except KeyError:
            raise LoggedError(
                self.log, "Units not known for H: '%s'. Try instead one of %r.",
                units, list(self.H_units_conv_factor))

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

    def get_matter_power(self, var_pair=("delta_tot", "delta_tot"), nonlinear=True,
                         _state=None):
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
        return None, None, None

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

        current_state = self.current_state()
        nonlinear = bool(nonlinear)
        key = ("Pk_interpolator", nonlinear, extrap_kmax) + tuple(var_pair)
        if key in current_state:
            return current_state[key]

        k, z, pk = self.get_matter_power(var_pair=var_pair, nonlinear=nonlinear,
                                         _state=current_state)
        log_p = np.all(pk > 0)
        if log_p:
            pk = np.log(pk)
        elif extrap_kmax > k[-1]:
            raise ValueError('cannot do log extrapolation with negative pk for %s, %s'
                             % var_pair)

        result = PowerSpectrumInterpolator(z, k, pk, logP=log_p, extrap_kmax=extrap_kmax)
        current_state[key] = result
        return result

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

    def current_state(self):
        lasts = [self._states[i]["last"] for i in range(self._n_states)]
        return self._states[lasts.index(max(lasts))]

    def __getattr__(self, method):
        try:
            object.__getattr__(self, method)
        except AttributeError:
            if method.startswith("get"):
                # Deprecated method names
                # -- this will be deprecated in favour of the error below
                new_names = {"get_cl": "get_Cl"}
                if method in new_names:
                    msg = create_banner(
                        "Method '%s' has been re-capitalized to '%s'.\n"
                        "Overriding for now, but please change it: "
                        "this will produce an error in the future." % (
                            method, new_names[method]))
                    for line in msg.split("\n"):
                        self.log.warning(line)
                    return getattr(self, new_names[method])
                # End of deprecation block ------------------------------
                raise LoggedError(
                    self.log, "Getter method for cosmology product %r is not known. "
                              "Maybe you meant any of %r?",
                    method, fuzzy_match(method, dir(self), n=3))


class PowerSpectrumInterpolator(RectBivariateSpline):
    r"""
    2D spline interpolation object (scipy.interpolate.RectBivariateSpline)
    to evaluate matter power spectrum as function of z and k.

    *This class is adapted from CAMB's own P(k) interpolator, by Antony Lewis;
    it's mostly interface-compatible with the original.*

    :param z: values of z for which the power spectrum was evaluated.
    :param k: values of k for which the power spectrum was evaluated.
    :param P_or_logP: Values of the power spectrum (or log-values, if logP=True).
    :param logk: if True (default: False), assumes that k's are log-spaced.
    :param logP: if True (default: False), log of power spectrum are given and used
        for the underlying interpolator.
    :param extrap_kmax: if set, use power law extrapolation beyond kmax up to
        extrap_kmax; useful for tails of integrals.
    """

    def __init__(self, z, k, P_or_logP, extrap_kmax=None, logk=False, logP=False):
        # TODO: here assuming at least 3 redshifts?
        # AL I renamed self.logP here to islog since was overriding logP() function
        self.logk, self.islog = logk, logP
        #  Check order
        z, k = (np.atleast_1d(x) for x in [z, k])
        i_z = np.argsort(z)
        i_k = np.argsort(k)
        self.z, self.k, P_or_logP = z[i_z], k[i_k], P_or_logP[i_z, :][:, i_k]
        self.zmin, self.zmax = np.min(self.z), np.max(self.z)
        self._fk = (lambda k: k if logk else np.log(k))
        # TODO: _finvk looks redundant
        self._finvk = (lambda k: np.exp(k) if logk else k)
        self.kmin, self.kmax = np.min(self.k), np.max(self.k)
        # Continue until extrap_kmax using a (log,log)-linear extrapolation
        if extrap_kmax and extrap_kmax > self.kmax:
            # TODO: here assuming k is k, but _fk seems to assume k is log(k) if
            #  logk=True. (doc string only refers to spacing, not actually being log(k))
            #  just remove logk option, avoiding doing log(exp(log(k)))?
            #  (depending on what CLASS is doing)
            assert not logk  # only trying to make for for False
            logknew = np.hstack(
                [np.log(self.k), np.log(self.kmax) * 0.1 + np.log(extrap_kmax) * 0.9,
                 np.log(extrap_kmax)])
            logPnew = np.empty((P_or_logP.shape[0], P_or_logP.shape[1] + 2))
            logPnew[:, :-2] = P_or_logP if self.islog else np.log(P_or_logP)
            diff = (logPnew[:, -3] - logPnew[:, -4]) / (logknew[-3] - logknew[-4])
            delta = diff * (logknew[-1] - logknew[-3])
            logPnew[:, -1] = logPnew[:, -3] + delta
            logPnew[:, -2] = logPnew[:, -3] + delta * 0.9
            self.kmax = extrap_kmax  # Added for consistency with CAMB

            P_or_logP = logPnew if self.islog else np.exp(logPnew)
            super(self.__class__, self).__init__(self.z, logknew, P_or_logP)

        else:
            super(self.__class__, self).__init__(self.z, self._fk(self.k), P_or_logP)

    def P(self, z, k, grid=None):
        """
        Get the power spectrum at (z,k).
        """
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return np.exp(self.logP(z, k, grid=grid))
        else:
            return self(z, self._fk(k), grid=grid)

    def logP(self, z, k, grid=None):
        """
        Get the log power spectrum at (z,k).
        """
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self(z, self._fk(k), grid=grid)
        else:
            return self.P(z, k, grid=grid)
