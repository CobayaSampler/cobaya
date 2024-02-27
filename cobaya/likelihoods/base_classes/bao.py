r"""
.. module:: bao

:Synopsis: BAO, f_sigma8 and other measurements at single redshifts, with correlations
:Author: Antony Lewis, Pablo Lemos (adapted to Cobaya by Jesus Torrado, with little
        modification)

This code provides a template for BAO, :math:`f\sigma_8`, :math:`H`
and other redshift dependent functions.

The datasets implemented at this moment are:

- ``bao.sixdf_2011_bao``
- ``bao.sdss_dr7_mgs``
- ``bao.sdss_dr12_consensus_bao``
- ``bao.sdss_dr12_consensus_full_shape``
- ``bao.sdss_dr12_consensus_final`` (combined data of the previous two)
- ``bao.sdss_dr16_baoplus_lrg`` (combining data from BOSS DR12 and eBOSS DR16, BAO+RSD)
- ``bao.sdss_dr16_baoplus_qso`` (DR16 BAO+RSD)
- ``bao.sdss_dr16_baoplus_elg`` (DR16 ELG BAO+RSD)
- ``bao.sdss_dr12_lrg_bao_dmdh`` (DR12 LRG BAO-only, independent of DR16 below)
- ``bao.sdss_dr16_lrg_bao_dmdh`` (DR16 LRG BAO-only, independent of DR12 above)
- ``bao.sdss_dr16_bao_elg`` (DR16 ELG BAO-only)
- ``bao.sdss_dr16_qso_bao_dmdh`` (DR16 QSO BAO-only)
- ``bao.sdss_dr16_baoplus_lyauto`` (DR16 LyA BAO-only)
- ``bao.sdss_dr16_baoplus_lyxqso`` (DR16 LyA x QSO BAO-only)


.. |br| raw:: html

   <br />

.. note::

   If you use the likelihood ``sixdf_2011_bao``, please cite:
   |br|
   F. Beutler et al,
   `The 6dF Galaxy Survey: baryon acoustic oscillations and the local Hubble constant`
   `(arXiv:1106.3366) <https://arxiv.org/abs/1106.3366>`_
   |br| |br|
   If you use the likelihood ``sdss_dr7_mgs``, please cite:
   |br|
   A.J. Ross et al,
   `The clustering of the SDSS DR7 main Galaxy sample - I.
   A 4 per cent distance measure at z = 0.15`
   `(arXiv:1409.3242) <https://arxiv.org/abs/1409.3242>`_
   |br| |br|
   If you use any of the likelihoods ``sdss_dr12_*``, please cite:
   |br|
   S. Alam et al,
   `The clustering of galaxies in the completed SDSS-III Baryon Oscillation Spectroscopic
   Survey: cosmological analysis of the DR12 galaxy sample`
   `(arXiv:1607.03155) <https://arxiv.org/abs/1607.03155>`_
   |br| |br|
   If you use any of the likelihoods ``sdss_dr16_*``, please cite:
   |br|
   S. Alam et al,
   `The Completed SDSS-IV extended Baryon Oscillation Spectroscopic Survey: Cosmological
   Implications from two Decades of Spectroscopic Surveys at the Apache Point observatory`
   `(arXiv:2007.08991) <https://arxiv.org/abs/2007.08991>`_


Usage
-----

To use any of these likelihoods, simply mention them in the likelihoods block, or add them
using the :doc:`input generator <cosmo_basic_runs>`.

These likelihoods have no nuisance parameters or particular settings that you may want
to change (except for the installation path; see below).

Note that although called "bao", many of these data combinations also include redshift
distortion data (RSD), encapsulated via a single "f sigma8" parameter (which is not
accurate for some non-LCDM models).


Defining your own BAO likelihood
--------------------------------

You can use the likelihood ``bao.generic`` as a template for any BAO data.

To do that, create a file containing the data points, e.g. ``myBAO.dat``, as

.. code::

   [z] [value at z] [quantity]
   ...

where you can use as many different quantities and redshifts as you like.

The available quantities are

- ``DV_over_rs``: Spherically-averaged distance, over sound horizon radius
- ``rs_over_DV``: Idem, inverse
- ``DM_over_rs``: Comoving angular diameter distance, over sound horizon radius
- ``DA_over_rs``: Physical angular diameter distance, over sound horizon radius
- ``Hz_rs``: Hubble parameter, times sound horizon radius
- ``f_sigma8``: Differential matter linear growth rate,
  times amplitude of present-day fuctuations
- ``F_AP``: Anisotropy (Alcock-Paczynski) parameter

In addition create a file, e.g. ``myBAO.cov``, containing the covariance matrix for those
data, with the same row order as the data file.

Now, add to your likelihood block:

.. literalinclude:: ../cobaya/likelihoods/bao/generic.yaml
   :language: yaml

You can rename your BAO likelihood and use multiple ones with different data (see
:ref:`likelihood_rename`).


Installation
------------

This likelihood can be installed automatically as explained in :doc:`installation_cosmo`.
If are following the instructions there (you should!), you don't need to read the rest
of this section.

Manual installation of the BAO likelihood data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming you are installing all your
likelihoods under ``/path/to/likelihoods``, simply do

.. code:: bash

   $ cd /path/to/likelihoods
   $ git clone https://github.com/CobayaSampler/bao_data.git

After this, mention the path to this likelihood when you include it in an input file, e.g.

.. code-block:: yaml

   likelihood:
     bao.sdss_dr12_consensus_[bao|full_shape|final|...]:
       path: /path/to/likelihoods/bao_data
"""
# Global
import os
import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, \
    RegularGridInterpolator
import pandas as pd
from typing import Optional, Sequence

# Local
from cobaya.log import LoggedError
from cobaya.conventions import Const, packages_path_input
from cobaya.likelihoods.base_classes import InstallableLikelihood


class BAO(InstallableLikelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "BAO"

    install_options = {"github_repository": "CobayaSampler/bao_data",
                       "github_release": "v2.3"}

    prob_dist_bounds: Optional[Sequence[float]]
    measurements_file: Optional[str] = None
    rs_fid: Optional[float] = None
    rs_rescale: Optional[float] = None
    prob_dist: Optional[str] = None
    cov_file: Optional[str] = None
    invcov_file: Optional[str] = None
    redshift: Optional[float] = None
    observable_1: Optional[str] = None
    observable_2: Optional[str] = None
    observable_3: Optional[str] = None
    grid_file: Optional[str] = None
    path: Optional[str]

    def initialize(self):
        if not getattr(self, "path", None) and \
                not getattr(self, packages_path_input, None):
            raise LoggedError(
                self.log, "No path given to BAO data. Set the likelihood property "
                          "'path' or the common property '%s'.", packages_path_input)
        # If no path specified, use the external packages path
        data_file_path = os.path.normpath(getattr(self, "path", None) or
                                          os.path.join(self.packages_path, "data"))
        # Rescaling by a fiducial value of the sound horizon
        if self.rs_rescale is None:
            if self.rs_fid is not None:
                self.rs_rescale = 1 / self.rs_fid
            else:
                self.rs_rescale = 1
        # Load "measurements file" and covmat of requested
        if self.measurements_file:
            try:
                self.data = pd.read_csv(
                    os.path.join(data_file_path, self.measurements_file),
                    header=None, index_col=None, sep=r"\s+", comment="#")
            except IOError:
                raise LoggedError(
                    self.log, "Couldn't find measurements file '%s' in folder '%s'. " % (
                        self.measurements_file, data_file_path) + "Check your paths.")
        elif self.grid_file:
            pass
        else:
            self.data = pd.DataFrame([self.data] if not hasattr(self.data[0], "__len__")
                                     else self.data)

        if not self.grid_file:
            self.use_grid_1d = False
            self.use_grid_2d = False
            self.use_grid_3d = False
            # Columns: z value [err] [type]
            self.has_type = self.data.iloc[:, -1].dtype == np.dtype("O")
            assert self.has_type  # mandatory for now!
            self.has_err = len(self.data.columns) > 2 and self.data[2].dtype == float
            if self.has_err:
                self.data.columns = ["z", "value", "error", "observable"]
            else:
                self.data.columns = ["z", "value", "observable"]
            prefix = "bao_"
            self.data["observable"] = [(c[len(prefix):] if c.startswith(prefix) else c)
                                       for c in self.data["observable"]]

        # Probability distribution
        if self.prob_dist:
            try:
                chi2 = np.loadtxt(os.path.join(data_file_path, self.prob_dist))
            except IOError:
                raise LoggedError(
                    self.log, "Couldn't find probability distribution file '%s' "
                              "in folder '%s'. " % (self.prob_dist, data_file_path) +
                              "Check your paths.")
            try:
                assert self.prob_dist_bounds
                alpha = np.linspace(
                    self.prob_dist_bounds[0], self.prob_dist_bounds[1], len(chi2))
            except (TypeError, AttributeError, IndexError, ValueError):
                raise LoggedError(
                    self.log, "If 'prob_dist' given, 'prob_dist_bounds' needs to be "
                              "specified as [min, max].")
            spline = UnivariateSpline(alpha, -chi2 / 2, s=0, ext=2)
            self.logpdf = lambda _x: (spline(_x)[0] if self.prob_dist_bounds[0]
                                                       <= _x <= self.prob_dist_bounds[1]
                                      else -np.inf)
        elif self.grid_file:
            try:
                self.grid_data = np.loadtxt(
                    os.path.join(data_file_path, self.grid_file))
            except IOError:
                raise LoggedError(
                    self.log, "Couldn't find grid file '%s' in folder '%s'. " % (
                        self.grid_file, data_file_path) + "Check your paths.")
            if self.redshift is None:
                raise LoggedError(
                    self.log, "If using grid data, 'redshift'"
                              "needs to be specified.")

            self.has_type = True  # Not sure what this is
            self.data = pd.DataFrame()

            if self.grid_data.shape[1] == 2:
                self.use_grid_1d = True
                self.use_grid_2d = False
                self.use_grid_3d = False
                if not self.observable_1:
                    raise LoggedError(
                        self.log, "If using grid data, 'observable_1'"
                                  "needs to be specified.")
                self.data["observable"] = [self.observable_1]
                x = self.grid_data[:, 0]
                chi2 = np.log(self.grid_data[:, 1])
                self.interpolator = UnivariateSpline(x, chi2, s=0, ext=2)
            elif self.grid_data.shape[1] == 3:
                self.use_grid_1d = False
                self.use_grid_2d = True
                self.use_grid_3d = False
                if not (self.observable_1 and self.observable_2):
                    raise LoggedError(
                        self.log, "If using grid data, 'observable_1' and 'observable_2'"
                                  "need to be specified.")
                self.data["observable"] = [self.observable_1, self.observable_2]

                x = np.unique(self.grid_data[:, 0])
                y = np.unique(self.grid_data[:, 1])

                Nx = x.shape[0]
                Ny = y.shape[0]

                chi2 = np.reshape(np.log(self.grid_data[:, 2]), [Nx, Ny])

                # Make the interpolator (x refers to at, y refers to ap).
                self.interpolator = RectBivariateSpline(x, y, chi2, kx=3, ky=3)
            elif self.grid_data.shape[1] == 4:
                self.use_grid_1d = False
                self.use_grid_2d = False
                self.use_grid_3d = True
                if not (self.observable_1 and self.observable_2 and self.observable_3):
                    raise LoggedError(
                        self.log,
                        "If using grid data, 'observable_1', 'observable_2' "
                        "and 'observable_3' need to be specified.")
                self.data["observable"] = [self.observable_1, self.observable_2,
                                           self.observable_3]

                x = np.unique(self.grid_data[:, 0])
                y = np.unique(self.grid_data[:, 1])
                z = np.unique(self.grid_data[:, 2])

                Nx = x.shape[0]
                Ny = y.shape[0]
                Nz = z.shape[0]

                chi2 = np.reshape(np.log(self.grid_data[:, 3] + 1e-300), [Nx, Ny, Nz])

                self.interpolator3D = RegularGridInterpolator((x, y, z), chi2,
                                                              bounds_error=False,
                                                              fill_value=np.log(1e-300))

            else:
                raise LoggedError(
                    self.log, "Grid data has the wrong dimensions")
                # Covariance --> read and re-sort as self.data
        else:
            self.use_grid_2d = False
            self.use_grid_3d = False
            try:
                if self.cov_file:
                    self.cov = np.loadtxt(os.path.join(data_file_path, self.cov_file))
                elif self.invcov_file:
                    invcov = np.loadtxt(os.path.join(data_file_path, self.invcov_file))
                    self.cov = np.linalg.inv(np.atleast_2d(invcov))
                elif "error" in self.data.columns:
                    self.cov = np.diag(self.data["error"] ** 2)
                else:
                    raise LoggedError(
                        self.log, "No errors provided, either as cov, invcov "
                                  "or as the 3rd column in the data file.")
                self.invcov = np.linalg.inv(np.atleast_2d(self.cov))
            except IOError:
                raise LoggedError(
                    self.log, "Couldn't find (inv)cov file '%s' in folder '%s'. " % (
                        self.cov_file or self.invcov_file,
                        data_file_path) + "Check your paths.")
            self.logpdf = lambda _x: (lambda x_: -0.5 * x_.dot(self.invcov).dot(x_))(
                _x - self.data["value"].values)
            self.log.info("Initialized.")

    def get_requirements(self):
        # Requisites
        if self.use_grid_1d:
            zs = {self.observable_1: np.array([self.redshift])
                  }
        elif self.use_grid_2d:
            zs = {self.observable_1: np.array([self.redshift]),
                  self.observable_2: np.array([self.redshift])}
        elif self.use_grid_3d:
            zs = {self.observable_1: np.array([self.redshift]),
                  self.observable_2: np.array([self.redshift]),
                  self.observable_3: np.array([self.redshift])
                  }
        else:
            zs = {obs: self.data.loc[self.data["observable"] == obs, "z"].values
                  for obs in self.data["observable"].unique()}
        theory_reqs = {
            "DV_over_rs": {
                "angular_diameter_distance": {"z": zs.get("DV_over_rs", None)},
                "Hubble": {"z": zs.get("DV_over_rs", None)},
                "rdrag": None},
            "rs_over_DV": {
                "angular_diameter_distance": {"z": zs.get("rs_over_DV", None)},
                "Hubble": {"z": zs.get("rs_over_DV", None)},
                "rdrag": None},
            "DM_over_rs": {
                "angular_diameter_distance": {"z": zs.get("DM_over_rs", None)},
                "rdrag": None},
            "DA_over_rs": {
                "angular_diameter_distance": {"z": zs.get("DA_over_rs", None)},
                "rdrag": None},
            "DH_over_rs": {
                "Hubble": {"z": zs.get("DH_over_rs", None)},
                "rdrag": None},
            "Hz_rs": {
                "Hubble": {"z": zs.get("Hz_rs", None)},
                "rdrag": None},
            "f_sigma8": {
                "fsigma8": {"z": zs.get("f_sigma8", None)},
            },
            "F_AP": {
                "angular_diameter_distance": {"z": zs.get("F_AP", None)},
                "Hubble": {"z": zs.get("F_AP", None)}}}
        obs_used_not_implemented = np.unique([obs for obs in self.data["observable"]
                                              if obs not in theory_reqs])
        if len(obs_used_not_implemented):
            raise LoggedError(
                self.log, "This likelihood refers to observables '%s' that "
                          "have not been  implemented yet. Did you mean any of %s? "
                          "If you didn't, please, open an issue in github.",
                obs_used_not_implemented, list(theory_reqs))
        requisites = {}
        if self.has_type:
            for observable in self.data["observable"].unique():
                for req, req_values in theory_reqs[observable].items():
                    if req not in requisites:
                        requisites[req] = req_values
                    else:
                        if isinstance(req_values, dict):
                            for k, v in req_values.items():
                                if v is not None:
                                    requisites[req][k] = np.unique(
                                        np.concatenate((requisites[req][k], v)))
        return requisites

    def theory_fun(self, z, observable):
        # Functions to get the corresponding theoretical prediction:
        # Spherically-averaged distance, over sound horizon radius
        if observable == "DV_over_rs":
            return np.cbrt(
                ((1 + z) * self.provider.get_angular_diameter_distance(z)) ** 2 *
                Const.c_km_s * z / self.provider.get_Hubble(z,
                                                            units="km/s/Mpc")) / self.rs()
        # Idem, inverse
        elif observable == "rs_over_DV":
            return np.cbrt(
                ((1 + z) * self.provider.get_angular_diameter_distance(z)) ** 2 *
                Const.c_km_s * z / self.provider.get_Hubble(z, units="km/s/Mpc")
            ) ** (-1) * self.rs()
        # Comoving angular diameter distance, over sound horizon radius
        elif observable == "DM_over_rs":
            return (1 + z) * self.provider.get_angular_diameter_distance(z) / self.rs()
        # Physical angular diameter distance, over sound horizon radius
        elif observable == "DA_over_rs":
            return self.provider.get_angular_diameter_distance(z) / self.rs()
        # Hubble distance [c/H(z)] over sound horizon radius.
        elif observable == "DH_over_rs":
            return 1 / self.provider.get_Hubble(z, units="1/Mpc") / self.rs()
        # Hubble parameter, times sound horizon radius
        elif observable == "Hz_rs":
            return self.provider.get_Hubble(z, units="km/s/Mpc") * self.rs()
        # Diff Linear Growth Rate times present amplitude
        elif observable == "f_sigma8":
            return self.provider.get_fsigma8(z)
        # Anisotropy (Alcock-Paczynski) parameter
        elif observable == "F_AP":
            return ((1 + z) * self.provider.get_angular_diameter_distance(z) *
                    self.provider.get_Hubble(z, units="km/s/Mpc")) / Const.c_km_s

    def rs(self):
        return self.provider.get_param("rdrag") * self.rs_rescale

    def logp(self, **params_values):
        if self.use_grid_1d:
            x = self.theory_fun(self.redshift, self.observable_1)
            chi2 = float(self.interpolator(x)[0])
            return chi2
        elif self.use_grid_2d:
            x = self.theory_fun(self.redshift, self.observable_1)
            y = self.theory_fun(self.redshift, self.observable_2)
            chi2 = self.interpolator(x, y)[0][0]
            return chi2
        elif self.use_grid_3d:
            x = self.theory_fun(self.redshift, self.observable_1)
            y = self.theory_fun(self.redshift, self.observable_2)
            z = self.theory_fun(self.redshift, self.observable_3)
            chi2 = self.interpolator3D(np.array([x, y, z])[:, 0])
            return chi2
        else:
            theory = np.array([self.theory_fun(z, obs) for z, obs
                               in zip(self.data["z"], self.data["observable"])]).T[0]
            if self.is_debug():
                for i, (z, obs, theo) in enumerate(
                        zip(self.data["z"], self.data["observable"], theory)):
                    self.log.debug("%s at z=%g : %g (theo) ; %g (data)",
                                   obs, z, theo, self.data.iloc[i, 1])
            return self.logpdf(theory)
