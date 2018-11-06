"""
.. module:: _cosmo

:Synopsis: Template for Cosmological theory codes.
           Mostly here to document how to compute and get observables.
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import
import numpy as np
from copy import deepcopy
from scipy.interpolate import RectBivariateSpline
from six import string_types
from itertools import chain

# Local
from cobaya.theory import Theory
from cobaya.log import HandledException


class _cosmo(Theory):

    def needs(self, **requirements):
        """
        Specifies the quantities that each likelihood needs from the Cosmology code.

        Typical requisites in Cosmology (as keywords, case insensitive):

        - ``cl={[...]}``: CMB lensed power spectra, as a dictionary ``{spectrum:l_max}``,
          where the possible spectra are combinations of "t", "e", "b" and "p"
          (lensing potential). Get with :func:`~_cosmo.get_cl`.
        - ``H={'z': [z_1, ...], 'units': '1/Mpc' or 'km/s/Mpc'}``: Hubble
          rate at the redshifts requested, in the given units. Get it with
          :func:`~_cosmo.get_H`.
        - ``angular_diameter_distance={'z': [z_1, ...]}``: Physical angular
          diameter distance to the redshifts requested. Get it with
          :func:`~_cosmo.get_angular_diameter_distance`.
        - ``comoving_radial_distance={'z': [z_1, ...]}``: Comoving radial distance
          from us to the redshifts requested. Get it with
          :func:`~_cosmo.get_comoving_radial_distance`.
        - ``fsigma8={'z': [z_1, ...]}``: Structure growth rate
          :math:`f\sigma_8` at the redshifts requested. Get it with
          :func:`~_cosmo.get_fsigma8`.
        - ``k_max=[...]``: Fixes the maximum comoving wavenumber considered.
        - **Other derived parameters** that are not included in the input but whose
          value the likelihood may need.
        """
        if not getattr(self, "_needs", None):
            self._needs = deepcopy(self.output_params)
        # Accumulate the requirements across several calls in a safe way;
        # e.g. take maximum of all values of a requested precision paramater
        for k, v in requirements.items():
            # Products and other computations
            if k.lower() == "cl":
                self._needs["cl"] = {
                    cl: max(self._needs.get("cl", {}).get(cl, 0), v.get(cl, 0))
                    for cl in set(self._needs.get("cl", {})).union(v)}
            elif k.lower() == "pk_interpolator":
                # Make sure vars_pairs is a list of [list of 2 vars pairs]
                vars_pairs = v.pop("vars_pairs", [])
                try:
                    if isinstance(vars_pairs[0], string_types):
                        vars_pairs = [vars_pairs]
                except IndexError:
                    # Empty list: by default [delta_tot, delta_tot]
                    vars_pairs = [2*["delta_tot"]]
                except:
                    self.log("Cannot understands vars_pairs '%r' for P(k) interpolator",
                             vars_pairs)
                    raise HandledException
                vars_pairs = set([tuple(pair) for pair in chain(
                    self._needs.get(k.lower(), {}).get("vars_pairs", []), vars_pairs)])
                self._needs[k.lower()] = {
                    "z": np.unique(np.concatenate(
                        (self._needs.get(k.lower(), {}).get("z", []),
                         np.atleast_1d(v["z"])))),
                    "k_max": max(
                        self._needs.get(k.lower(), {}).get("k_max", 0), v["k_max"]),
                    "vars_pairs": vars_pairs}
                self._needs[k.lower()].update(v)
            elif k.lower() in ["h", "angular_diameter_distance",
                               "comoving_radial_distance", "fsigma8"]:
                if not k.lower() in self._needs:
                    self._needs[k.lower()] = {}
                self._needs[k.lower()]["z"] = np.unique(np.concatenate(
                    (self._needs[k.lower()].get("z", []), v["z"])))
            # Extra derived paramaters and other unknown stuff (keep capitalization)
            elif v is None:
                self._needs[k] = None
            else:
                self.log.error("Unknown required product: '%s:%s'.", k, v)
                raise HandledException

    def requested(self):
        """
        Returns the full set of cosmological products and parameters requested by the
        likelihoods.
        """
        return self._needs

    def get_param(self, p):
        """
        Interface function for likelihoods to get sampled and derived parameters.

        Always use this one; don't try to access theory code attributes directly!
        """
        pass

    def get_cl(self, ell_factor=False, units="muK2"):
        """
        Returns a dictionary of lensed CMB power spectra and the lensing potential ``pp`` power spectrum.

        Set the units with the keyword ``units='1'|'muK2'|'K2'`` (default: 'muK2',
        except for the lensing potential power spectrum, which is always unitless).

        If ``ell_factor=True`` (default: False), multiplies the spectra by
        :math:`\ell(\ell+1)/(2\pi)` (or by :math:`\ell^2(\ell+1)^2/(2\pi)` in the case of
        the lensing potential ``pp`` spectrum).
        """
        pass

    def get_H(self, z, units="km/s/Mpc"):
        """
        Returns the Hubble rate at the given redshifts.

        The redshifts must be a subset of those requested when :func:`~_cosmo.needs`
        was called.

        The available units are ``km/s/Mpc`` (i.e. ``c*H(Mpc^-1)``) and ``1/Mpc``.
        """
        pass

    def get_angular_diameter_distance(self, z):
        """
        Returns the physical angular diameter distance to the given redshifts.

        The redshifts must be a subset of those requested when :func:`~_cosmo.needs`
        was called.
        """
        pass

    def get_Pk_interpolator(self, z):
        """
        Returns a (dict of) power spectrum interpolator(s)
        :class:`PowerSpectrumInterpolator`.
        """
        pass

    def get_fsigma8(self, z):
        """
        Structure growth rate :math:`f\sigma_8`, as defined in eq. 33 of
        `Planck 2015 results. XIII. Cosmological parameters <https://arxiv.org/pdf/1502.01589.pdf>`_,
        at the given redshifts.

        The redshifts must be a subset of those requested when :func:`~_cosmo.needs`
        was called.
        """
        pass

    def get_fsigma8(self, z):
        """
        Structure growth rate :math:`f\sigma_8`, as defined in eq. 33 of
        `Planck 2015 results. XIII. Cosmological parameters <https://arxiv.org/pdf/1502.01589.pdf>`_,
        at the given redshifts.

        The redshifts must be a subset of those requested when :func:`~_cosmo.needs`
        was called.
        """
        pass

    def get_auto_covmat(self, params_info, likes_info):
        """
        Tries to get match to a database of existing covariance matrix files for the current model and data.

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
    :param logk: if True (default: False), assumes that k's are log-spaced.
    :param logP: if True (default: False), log of power spectrum are given and used
        for the underlying interpolator.
    :param extrap_kmax: if set, use power law extrapolation beyond kmax up to
        extrap_kmax; useful for tails of integrals.
    """

    def __init__(self, z, k, P_or_logP, extrap_kmax=None, logk=False, logP=False):
        self.logk, self.logP = logk, logP
        #  Check order
        z, k = [np.atleast_1d(x) for x in [z,k]]
        i_z = np.argsort(z)
        i_k = np.argsort(k)
        self.z, self.k, P_or_logP = z[i_z], k[i_k], P_or_logP[i_z, :][:, i_k]
        self.zmin, self.zmax = np.min(self.z), np.max(self.z)
        self._fk = (lambda k: k if logk else np.log(k))
        self._finvk = (lambda k: np.exp(k) if logk else k)
        self.kmin, self.kmax = np.min(self.k), np.max(self.k)
        # Continue until extrap_kmax using a (log,log)-linear extrapolation
        if extrap_kmax and extrap_kmax > self.kmax:
            logknew = np.log(np.hstack([self.k, extrap_kmax]))
            logPnew = np.empty((P_or_logP.shape[0], P_or_logP.shape[1] + 1))
            logPnew[:, :-1] = P_or_logP if self.logP else np.log(P_or_logP)
            logPnew[:, -1] = (
                logPnew[:, -2] +
                (logPnew[:, -2] - logPnew[:, -3]) / (logknew[-2] - logknew[-3]) *
                (logknew[-1] - logknew[-2]))
            self.k, P_or_logP = np.exp(logknew), logPnew if self.logP else np.exp(logPnew)
        super(self.__class__, self).__init__(self.z, self._fk(self.k), P_or_logP)

    def P(self, z, k, grid=None):
        """
        Get the power spectrum at (z,k).
        """
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.logP:
            return np.exp(self.logP(z, k, grid=grid))
        else:
            return self(z, self._fk(k), grid=grid)

    def logP(self, z, k, grid=None):
        """
        Get the log power spectrum at (z,k).
        """
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.logP:
            return self(z, self._fk(k), grid=grid)
        else:
            return self.P(z, k, grid=grid)
