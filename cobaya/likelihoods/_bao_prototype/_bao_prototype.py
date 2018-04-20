"""
.. module:: _bao_prototype

:Synopsis: BAO, f_sigma8 and other measurements at single redshifts, with correlations
:Author: Antony Lewis (adapted to Cobaya by Jesus Torrado, with little modification)

This code provides a template for BAO, :math:`f\sigma_8`, :math:`H`
and other redshift dependent functions.

The datasets implemented at this moment are:

- ``sdss_dr12_consensus_bao``
- ``sdss_dr12_consensus_full_shape``
- ``sdss_dr12_consensus_final`` (combined data of the previous two)

.. |br| raw:: html

   <br />

.. note::

   If you use any of the likelihoods above, please cite:
   |br|
   S. Alam el at,
   `The clustering of galaxies in the completed SDSS-III Baryon Oscillation Spectroscopic
   Survey: cosmological analysis of the DR12 galaxy sample`
   `(arXiv:1607.03155) <https://arxiv.org/abs/1607.03155>`_


Usage
-----

To use any of these likelihoods, simply mention them in the theory block.

An example of usage can be found in :doc:`examples_bao`.

These likelihoods have no nuisance parameters or particular settings that you may want
to change (except for the installation path; see below)


Defining your own BAO likelihood
--------------------------------

You can use the likelihood ``bao_generic`` as a template for any BAO data.

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

.. literalinclude:: ../cobaya/likelihoods/bao_generic/bao_generic.yaml
   :language: yaml


Installation
------------

This likelihood can be installed automatically as explained in :doc:`installation_cosmo`.
If are following the instructions there (you should!), you don't need to read the rest
of this section.

Manual installation of the SDSS likelihood data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming you are installing all your
likelihoods under ``/path/to/likelihoods``, simply do

.. code:: bash

   $ cd /path/to/likelihoods
   $ git clone https://github.com/JesusTorrado/sdss_dr12.git

After this, mention the path to this likelihood when you include it in an input file as

.. code-block:: yaml

   likelihood:
     sdss_dr12_consensus_[bao|full_shape|final|...]:
       path: /path/to/likelihoods/sdss_dr12


"""


# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Global
import os
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import HandledException
from cobaya.conventions import _path_install, _c
from cobaya.tools import get_path_to_installation


class _bao_prototype(Likelihood):

    def initialise(self):
        # If no path specified, use the modules path
        data_file_path = (getattr(self, "path", None) or
                          os.path.join(get_path_to_installation(), "data/sdss_dr12"))
        if not data_file_path:
            self.log.error("No path given to BAO data. Set the likelihood property "
                           "'path' or the common property '%s'.", _path_install)
            raise HandledException
        # Rescaling by a fiducial value of the sound horizon
        if self.rs_fid is None:
            self.rs_fid = 1
        # Load "measurements file" and covmat of requested
        try:
            self.data = pd.read_csv(os.path.join(data_file_path, self.measurements_file),
                                    header=None, index_col=None, sep="\s+", comment="#")
        except IOError:
            self.log.error("Couldn't find measurements file '%s' in folder '%s'. "%(
                self.measurements_file, data_file_path) + "Check your paths.")
            raise HandledException
        # Colums: z value [err] [type]
        self.has_type = self.data.iloc[:, -1].dtype == np.dtype("O")
        assert self.has_type  # mandatory for now!
        self.has_err = len(self.data.columns) > 2 and self.data.iloc[2].dtype == np.float
        assert not self.has_err  # not supported for now!
        self.data.columns = ["z", "value", "observable"]
        prefix = "bao_"
        self.data["observable"] = [(c[len(prefix):] if c.startswith(prefix) else c)
                                   for c in self.data["observable"]]
        # Covariance --> read and re-sort as self.data
        try:
            if hasattr(self, "cov_file"):
                self.cov = np.loadtxt(os.path.join(data_file_path, self.cov_file))
            elif hasattr(self, "invcov_file"):
                invcov = np.loadtxt(os.path.join(data_file_path, self.invcov_file))
                self.cov = np.linalg.inv(invcov)
            else:
                raise NotImplementedError("Manual errors not implemented yet.")
                # self.cov = np.diag(ERROR_HERE**2)
        except IOError:
            self.log.error("Couldn't find (inv)cov file '%s' in folder '%s'. "%(
                getattr(self, "cov_file", getattr(self, "invcov_file", None)),
                data_file_path) + "Check your paths.")
            raise HandledException
        self.norm = multivariate_normal(mean=self.data["value"].values, cov=self.cov)

    def add_theory(self):
        if self.theory.__class__ == "classy":
            self.log.error(
                "BAO likelihood not yet compatible with CLASS (help appreciated!)")
            raise HandledException
        # Requisites
        zs = {obs:self.data.loc[self.data["observable"] == obs, "z"].values
              for obs in self.data["observable"].unique()}
        theory_reqs = {
            "DV_over_rs": {
                "angular_diameter_distance": {"redshifts": zs.get("DV_over_rs", None)},
                "h_of_z": {"redshifts": zs.get("DV_over_rs", None), "units": "km/s/Mpc"},
                "rdrag": None},
            "rs_over_DV": {
                "angular_diameter_distance": {"redshifts": zs.get("rs_over_DV", None)},
                "h_of_z": {"redshifts": zs.get("rs_over_DV", None), "units": "km/s/Mpc"},
                "rdrag": None},
            "DM_over_rs": {
                "angular_diameter_distance": {"redshifts": zs.get("DM_over_rs", None)},
                "rdrag": None},
            "DA_over_rs": {
                "angular_diameter_distance": {"redshifts": zs.get("DA_over_rs", None)},
                "rdrag": None},
            "Hz_rs": {
                "h_of_z": {"redshifts": zs.get("Hz_rs", None), "units": "km/s/Mpc"},
                "rdrag": None},
            "f_sigma8": {
                "fsigma8": {"redshifts": zs.get("f_sigma8", None)},
                "h_of_z": {"redshifts": zs.get("Hz_rs", None), "units": "km/s/Mpc"}},
            "F_AP": {
                "angular_diameter_distance": {"redshifts": zs.get("F_AP", None)},
                "h_of_z": {"redshifts": zs.get("F_AP", None), "units": "km/s/Mpc"}}}
        obs_used_not_implemented = np.unique([obs for obs in self.data["observable"]
                                              if obs not in theory_reqs])
        if len(obs_used_not_implemented):
            self.log.error("This likelihood refers to observables '%s' that have not been"
                           " implemented yet. Did you mean any of %s? "
                           "If you didn't, please, open an issue in github.",
                           obs_used_not_implemented, list(theory_reqs.keys()))
            raise HandledException

        requisites = {}
        if self.has_type:
            for obs in self.data["observable"].unique():
                requisites.update(theory_reqs[obs])
        self.theory.needs(requisites)

    def theory_fun(self, z, observable):
        # Functions to get the corresponding theoretical prediction:
        # Spherically-averaged distance, over sound horizon radius
        if observable == "DV_over_rs":
            return np.cbrt(((1+z)*self.theory.get_angular_diameter_distance(z))**2 *
                           _c*z/self.theory.get_h_of_z(z))/self.rs()
        # Idem, inverse
        elif observable == "rs_over_DV":
            return np.cbrt(((1+z)*self.theory.get_angular_diameter_distance(z))**2 *
                           _c*z/self.theory.get_h_of_z(z))**(-1)*self.rs()
        # Comoving angular diameter distance, over sound horizon radius
        elif observable == "DM_over_rs":
            return (1+z)*self.theory.get_angular_diameter_distance(z)/self.rs()
        # Physical angular diameter distance, over sound horizon radius
        elif observable == "DA_over_rs":
            return self.theory.get_angular_diameter_distance(z)/self.rs()
        # Hubble parameter, times sound horizon radius
        elif observable == "Hz_rs":
            return self.theory.get_h_of_z(z)*self.rs()
        # Diff Linear Growth Rate times present amplitude
        elif observable == "f_sigma8":
            return self.theory.get_fsigma8(z)
        # Anisotropy (Alcock-Paczynski) parameter
        elif observable == "F_AP":
            return ((1+z) * self.theory.get_angular_diameter_distance(z) *
                    self.theory.get_h_of_z(z))/_c

    def rs(self):
        return self.theory.get_param("rdrag") / self.rs_fid

    def logp(self, **params_values):
        theory = np.array([self.theory_fun(z,obs) for z, obs
                           in zip(self.data["z"], self.data["observable"])]).T[0]
        return self.norm.logpdf(theory)
