r"""
.. module:: _bao_prototype

:Synopsis: BAO, f_sigma8 and other measurements at single redshifts, with correlations
:Author: Antony Lewis (adapted to Cobaya by Jesus Torrado, with little modification)

This code provides a template for BAO, :math:`f\sigma_8`, :math:`H`
and other redshift dependent functions.

The datasets implemented at this moment are:

- ``bao.sixdf_2011_bao``
- ``bao.sdss_dr7_mgs``
- ``bao.sdss_dr12_consensus_bao``
- ``bao.sdss_dr12_consensus_full_shape``
- ``bao.sdss_dr12_consensus_final`` (combined data of the previous two)

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


Usage
-----

To use any of these likelihoods, simply mention them in the likelihoods block, or add them
using the :doc:`input generator <cosmo_basic_runs>`.

These likelihoods have no nuisance parameters or particular settings that you may want
to change (except for the installation path; see below)


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
from scipy.interpolate import UnivariateSpline
import pandas as pd
import logging
from typing import Optional, Sequence

# Local
from cobaya.log import LoggedError
from cobaya.conventions import _packages_path, _c_km_s
from cobaya.likelihoods._base_classes import _InstallableLikelihood


class _bao_prototype(_InstallableLikelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "BAO"

    install_options = {"github_repository": "CobayaSampler/bao_data",
                       "github_release": "v1.1"}

    prob_dist_bounds: Optional[Sequence[float]]
    measurements_file: str
    rs_fid: Optional[float]
    prob_dist: Optional[str]
    cov_file: Optional[str]
    invcov_file: Optional[str]

    def initialize(self):
        self.log.info("Initialising.")
        if not getattr(self, "path", None) and not getattr(self, "packages_path", None):
            raise LoggedError(
                self.log, "No path given to BAO data. Set the likelihood property "
                          "'path' or the common property '%s'.", _packages_path)
        # If no path specified, use the external packages path
        data_file_path = os.path.normpath(getattr(self, "path", None) or
                                          os.path.join(self.packages_path, "data"))
        # Rescaling by a fiducial value of the sound horizon
        if getattr(self, "rs_rescale", None) is None:
            if getattr(self, "rs_fid", None) is not None:
                self.rs_rescale = 1 / self.rs_fid
            else:
                self.rs_rescale = 1
        # Load "measurements file" and covmat of requested
        if getattr(self, "measurements_file", None):
            try:
                self.data = pd.read_csv(
                    os.path.join(data_file_path, self.measurements_file),
                    header=None, index_col=None, sep=r"\s+", comment="#")
            except IOError:
                raise LoggedError(
                    self.log, "Couldn't find measurements file '%s' in folder '%s'. " % (
                        self.measurements_file, data_file_path) + "Check your paths.")
        else:
            self.data = pd.DataFrame([self.data] if not hasattr(self.data[0], "__len__")
                                     else self.data)
        # Columns: z value [err] [type]
        self.has_type = self.data.iloc[:, -1].dtype == np.dtype("O")
        assert self.has_type  # mandatory for now!
        self.has_err = len(self.data.columns) > 2 and self.data[2].dtype == np.float
        if self.has_err:
            self.data.columns = ["z", "value", "error", "observable"]
        else:
            self.data.columns = ["z", "value", "observable"]
        prefix = "bao_"
        self.data["observable"] = [(c[len(prefix):] if c.startswith(prefix) else c)
                                   for c in self.data["observable"]]
        # Probability distribution
        if getattr(self, "prob_dist", None):
            try:
                chi2 = np.loadtxt(os.path.join(data_file_path, self.prob_dist))
            except IOError:
                raise LoggedError(
                    self.log, "Couldn't find probability distribution file '%s' "
                              "in folder '%s'. " % (self.prob_dist, data_file_path) +
                              "Check your paths.")
            try:
                alpha = np.linspace(
                    self.prob_dist_bounds[0], self.prob_dist_bounds[1], len(chi2))
            except (TypeError, AttributeError, IndexError, ValueError):
                raise LoggedError(
                    self.log, "If 'prob_dist' given, 'prob_dist_bounds' needs to be "
                              "specified as [min, max].")
            spline = UnivariateSpline(alpha, -chi2 / 2, s=0)
            self.logpdf = lambda x: (
                spline(x)[0] if self.prob_dist_bounds[0] <= x <= self.prob_dist_bounds[1]
                else -np.inf)
        # Covariance --> read and re-sort as self.data
        else:
            try:
                if getattr(self, "cov_file", None):
                    self.cov = np.loadtxt(os.path.join(data_file_path, self.cov_file))
                elif getattr(self, "invcov_file", None):
                    invcov = np.loadtxt(os.path.join(data_file_path, self.invcov_file))
                    self.cov = np.linalg.inv(invcov)
                elif "error" in self.data.columns:
                    self.cov = np.diag(self.data["error"] ** 2)
                else:
                    raise LoggedError(
                        self.log, "No errors provided, either as cov, invcov "
                                  "or as the 3rd column in the data file.")
                self.invcov = np.linalg.inv(self.cov)
            except IOError:
                raise LoggedError(
                    self.log, "Couldn't find (inv)cov file '%s' in folder '%s'. " % (
                        getattr(self, "cov_file", getattr(self, "invcov_file", None)),
                        data_file_path) + "Check your paths.")
            self.logpdf = lambda x: (lambda x_: -0.5 * x_.dot(self.invcov).dot(x_))(
                x - self.data["value"].values)

    def get_requirements(self):
        # Requisites
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
            "Hz_rs": {
                "Hubble": {"z": zs.get("Hz_rs", None)},
                "rdrag": None},
            "f_sigma8": {
                "fsigma8": {"z": zs.get("f_sigma8", None)},
                "Hubble": {"z": zs.get("Hz_rs", None)}},
            "F_AP": {
                "angular_diameter_distance": {"z": zs.get("F_AP", None)},
                "Hubble": {"z": zs.get("F_AP", None)}}}
        obs_used_not_implemented = np.unique([obs for obs in self.data["observable"]
                                              if obs not in theory_reqs])
        if len(obs_used_not_implemented):
            raise LoggedError(
                self.log, "This likelihood refers to observables '%s' that have not been"
                          " implemented yet. Did you mean any of %s? "
                          "If you didn't, please, open an issue in github.",
                obs_used_not_implemented, list(theory_reqs))
        requisites = {}
        if self.has_type:
            for obs in self.data["observable"].unique():
                requisites.update(theory_reqs[obs])
        return requisites

    def theory_fun(self, z, observable):
        # Functions to get the corresponding theoretical prediction:
        # Spherically-averaged distance, over sound horizon radius
        if observable == "DV_over_rs":
            return np.cbrt(
                ((1 + z) * self.provider.get_angular_diameter_distance(z)) ** 2 *
                _c_km_s * z / self.provider.get_Hubble(z, units="km/s/Mpc")) / self.rs()
        # Idem, inverse
        elif observable == "rs_over_DV":
            return np.cbrt(
                ((1 + z) * self.provider.get_angular_diameter_distance(z)) ** 2 *
                _c_km_s * z / self.provider.get_Hubble(z, units="km/s/Mpc")) ** (
                       -1) * self.rs()
        # Comoving angular diameter distance, over sound horizon radius
        elif observable == "DM_over_rs":
            return (1 + z) * self.provider.get_angular_diameter_distance(z) / self.rs()
        # Physical angular diameter distance, over sound horizon radius
        elif observable == "DA_over_rs":
            return self.provider.get_angular_diameter_distance(z) / self.rs()
        # Hubble parameter, times sound horizon radius
        elif observable == "Hz_rs":
            return self.provider.get_Hubble(z, units="km/s/Mpc") * self.rs()
        # Diff Linear Growth Rate times present amplitude
        elif observable == "f_sigma8":
            return self.provider.get_fsigma8(z)
        # Anisotropy (Alcock-Paczynski) parameter
        elif observable == "F_AP":
            return ((1 + z) * self.provider.get_angular_diameter_distance(z) *
                    self.provider.get_Hubble(z, units="km/s/Mpc")) / _c_km_s

    def rs(self):
        return self.provider.get_param("rdrag") * self.rs_rescale

    def logp(self, **params_values):
        theory = np.array([self.theory_fun(z, obs) for z, obs
                           in zip(self.data["z"], self.data["observable"])]).T[0]
        if self.log.getEffectiveLevel() == logging.DEBUG:
            for i, (z, obs, theo) in enumerate(
                    zip(self.data["z"], self.data["observable"], theory)):
                self.log.debug("%s at z=%g : %g (theo) ; %g (data)",
                               obs, z, theo, self.data.iloc[i, 1])
        return self.logpdf(theory)
