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
                self._needs[k] = deepcopy(v)
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

    def get_comoving_radial_distance(self, z):
        """
        Returns the comoving radial distance from us to the given redshifts.

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
