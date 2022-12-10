"""
.. module:: fsigma8

:Synopsis: Growth of LSS, fsigma8, measurements at individual redshifts
:Author: Minh Nguyen

This code provides a simple Gaussian likelihood for measurements of LSS growth, :math:`f\sigma_8`, as a function of redshifts

Available datasets are:

[list of datasets]

You can find them and/or add your own at:

[data location]

Input data files:

    - measurement_file: measurements of the form, [column] is optional
    .. code::

        z1 f(z1)sigma8(z1) [f(z1)sigma8(z1)_error]
        ...
        zN f(zN)sigma8(zN) [f(zN)sigma8(zN)_error]

    - cov_file: (NxN) covariance matrix of the corresponding measurement
    - invcov_file: inverse of the above covariance matrix (only supply either of the two)

.. |br| raw:: html

    <br />

.. note::

    - Currently assuming a diagonal covariance
"""
# Global
import os, sys
import pkg_resources
from typing import Optional, Sequence

import pandas as pd
import numpy as np

# Local
from cobaya.log import LoggedError, is_debug
from cobaya.conventions import Const, packages_path_input
from cobaya.likelihood import Likelihood

class fsigma8(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "fsigma8"

    # Default path
    path: Optional[str]
    data_dir: Optional[str] = pkg_resources.resource_filename("fsigma8", "data/")
    measurement_file: Optional[str] = None
    cov_file: Optional[str] = None
    invcov_file: Optional[str] = None
    # Default param (if any)

    def initialize(self):
        """
        Read in data files, and initialize the likelihood
        """
        if not getattr(self, "path", None) and \
                not getattr(self, packages_path_input, None):
            raise LoggedError(
                    self.log, "No path given to growth (fsigma8) data. Check and set "
                              "the likelihood's property 'path' or the common property '%s'.", packages_path_input)
        # If no path specified, use the modules path for file path
        data_path = os.path.normpath(getattr(self, "path", None) or
                                          os.path.join(self.packages_path, "data"))
        # Assemble the full path to the directory of data file(s)
        self.data_dir = os.path.join(data_path, self.data_dir)
        if not os.path.exists(self.data_dir):
            raise LoggedError(
                    self.log,
                    "The data directory 'data_dir' does not exist at [{self.data_dir}].")
        # Load measurement file
        if self.measurement_file:
            try:
                self.data = pd.read_csv(
                        os.path.join(self.data_dir, self.measurement_file),
                        header=None, index_col=False, sep=r"\s+", comment="#")
            except IOError:
                raise LoggedError(
                        self.log, "Cannot locate measurement file '%s' in folder '%s'. " % (
                            self.measurement_file, self.data_dir))

        else:
            self.data = pd.DataFrame([self.data] if not hasattr(self.data[0], "__len__")
                                     else self.data)

        # Label the columns
        self.has_err = len(self.data.columns) > 2 and self.data[2].dtype == float
        if self.has_err:
            self.data.columns = ["z", "fsigma8", "fsigma8_err"]
        else:
            self.data.columns = ["z", "fsigma8"]

        # Load covariance matrix
        try:
            if self.cov_file:
                self.cov = np.loadtxt(os.path.join(self.data_dir, self.cov_file))
            elif self.invcov_file:
                invcov = np.loadtxt(os.path.join(self.data_dir, self.invcov_file))
                self.cov = np.linalg.inv(invcov)
            elif "fsigma8_err" in self.data.columns:
                self.cov = np.diag(self.data["fsigma8_err"] ** 2)
            else:
                raise LoggedError(
                    self.log, "No measurment error provided. Supply it either as a cov_file/invcov_file "
                                "or as the 3rd column in the measurment file.")
            self.invcov = np.linalg.inv(self.cov)
        except IOError:
            raise LoggedError(
                self.log, "Cannot locate the cov_file or invcov_file '%s' in folder '%s'. " % (
                    self.cov_file or self.invcov_file,
                    data_file_path))
        # Gaussian likelihood on _x
        # Note the nested lambda functions, i.e. x_ = _x - self.data["value"].values
        self.logpdf = lambda _x: (lambda x_: -0.5 * x_.dot(self.invcov).dot(x_))(
            _x - self.data["fsigma8"].values)
        self.log.info("Initialized.")

    def get_requirements(self):
        """
        Return a dictionary of requisite(s) to be computed by the (Bolztmann) theory code
        In our case, it is the quantity fsigma8 at the measured redshifts
        """
        z=self.data["z"].to_numpy()
        requisites = {'gamma0': None,
                      'Omega_b': {'z': z},
                      'Omega_cdm': {'z': z},
                      'Omega_nu_massive': {'z': z},
                      'sigma8_z': {'z': z},
                      'fsigma8': {'z': z},
                      'Pk_interpolator': {
                          'z': z,
                          'k_max': 2.0,
                          'vars_pairs': [["delta_tot", "delta_tot"]],
                          'nonlinear': False},
                      'Pk_grid': {
                          'z': z,
                          'k_max': 2.0,
                          'vars_pairs': [["delta_tot", "delta_tot"]],
                          'nonlinear': True}}
        return requisites

    def theory_function(self, z):
        """
        Interface with the (Bolztmann) theory code to actually retrieve the requisite(s)
        """
        k_ref=2E-4
        Omegam_z = self.provider.get_Omega_b(z)+self.provider.get_Omega_cdm(z)+self.provider.get_Omega_nu_massive(z)
        f_z = Omegam_z**(self.provider.get_param('gamma0'))
        sigma8_z = self.provider.get_sigma8_z(z)
        Plin = self.provider.get_Pk_interpolator(var_pair=("delta_tot","delta_tot"),
                                                 nonlinear=False,
                                                 extrap_kmax=10.)
        Pnonlin = self.provider.get_Pk_interpolator(var_pair=("delta_tot","delta_tot"),
                                                    nonlinear=True,
                                                    extrap_kmax=10.)
        # Rescale sigma8_z to account for the possibility of gamma!=0.55
        sigma8_z *= np.sqrt(Pnonlin.P(z,k_ref)/Plin.P(z,k_ref))
        fsigma8_z = f_z*sigma8_z
        if self.is_debug():
            fsigma8_z_ref = self.provider.get_fsigma8(z)
            self.log.debug("fsigma8_z vs. fsigma8_z_ref at z=%g:
                           %g (fsigma8_gamma) ; %g (fsigma8_ref)",
                           z, fsigma8_z, fsigma8_z_ref)
        return fsigma8_z

    def logp(self, **params_values):
        """
        Compute the log-likelihood P(data|params_values), where params_value are sampled values of parameters
        """
        theoretical_prediction = \
            np.array([self.theory_function(z)
                        for z in self.data["z"]])
        # If theoretical_prediction does not have shape [nr_z,] then uncomment this
        theoretical_prediction = theoretical_prediction.reshape((len(self.data["z"]),))
        if self.is_debug():
            for i, (z, theory) in enumerate(
                    zip(self.data["z"], theoretical_prediction)):
                self.log.debug("fsigma8 at z=%g: %g (theo) ; %g (data)",
                                z, theory, self.data.iloc[i, 1])
        return self.logpdf(theoretical_prediction)
