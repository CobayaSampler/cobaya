r"""
.. module:: cc

:Synopsis: Cosmic chronometers likelihood class
:Author: Salih Yesil

Cosmic chronometers (CC) measure the Hubble rate :math:`H(z)` from the differential age
evolution of passively evolving galaxies. This is a Gaussian likelihood for a set of such
:math:`H(z)` measurements, including the systematic uncertainties of the method with the
full covariance matrix of `Moresco et al. (2020)
<https://ui.adsabs.harvard.edu/abs/2020ApJ...898...82M/abstract>`_:

.. math::

   C_{ij} = \sigma_i^2\,\delta_{ij} + \sum_k H_i f_k(z_i)\, H_j f_k(z_j)\,,

where :math:`\sigma_i` are the tabulated (uncorrelated) errors of the measurements and
:math:`f_k(z)` are the fractional systematic errors of the components :math:`k` listed
in ``systematics`` (by default the initial mass function and the stellar population
synthesis model), which are fully correlated across redshift. The fractional errors are
linearly interpolated in redshift from the table of Moresco et al. (2020), and held at
the values at the ends of the table beyond it.

This is the construction of the public `CCcovariance
<https://gitlab.com/mmoresco/CCcovariance>`_ repository by M. Moresco. The data files of
``cc.moresco2020`` are downloaded from that repository (at a fixed commit) when
installing the likelihood with ``cobaya-install cc.moresco2020``.

Using your own measurements
---------------------------

The measurements can be replaced with the ``data_file`` option: a text file (comma- or
space-separated, ``#`` for comments) whose columns are :math:`z`, :math:`H(z)` and its
uncorrelated error, in km/s/Mpc. If the measurements come with their own covariance
matrix, pass it (in (km/s/Mpc)\ :sup:`2`, one row per line) with ``covmat_file``; it is
then used instead of the errors in the third column, which can be omitted.

.. code:: yaml

   likelihood:
     cc.moresco2020:
       data_file: my_cc_data.txt
       covmat_file: my_cc_covmat.txt  # optional
       systematics: []

Relative paths that you pass are understood with respect to the current working
directory (the defaults of a likelihood class, with respect to the folder of its
installed data, or of the class if it has no data to install).

.. warning::

   The systematic terms listed in ``systematics`` are *added* to the errors (or to the
   covariance matrix) of the data. If the errors of your measurements already include
   the systematics of the method, as is the case for many published covariance
   matrices, set ``systematics: []`` so that they are not counted twice.

To make a set of measurements available by name, create a class inheriting from
:class:`cc.CC` with a ``.yaml`` file setting these options, as ``cc.moresco2020`` does.
"""

import os

import numpy as np

from cobaya.likelihoods.base_classes.InstallableLikelihood import InstallableLikelihood
from cobaya.log import LoggedError


class CC(InstallableLikelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "CC"

    # variables from yaml
    path: str | None = None
    data_file: str
    covmat_file: str | None = None
    systematics_file: str | None = None
    systematics: list[str] = []

    def initialize(self):
        # Folder of the default data files: `path` if given, else the installation
        # folder (for classes with data to install), else the folder of the class
        if self.path:
            self._data_folder = self.path
        elif self.get_install_options() and self.packages_path:
            self._data_folder = self.get_path(self.packages_path)
        else:
            self._data_folder = self.get_class_path()
        data = self._load_table("data_file")
        self.z, self.H_data = data[:, 0], data[:, 1]
        if self.covmat_file:
            self.cov = self._load_table("covmat_file")
            if self.cov.shape != (len(self.z), len(self.z)):
                raise LoggedError(
                    self.log,
                    "The covariance matrix in '%s' has shape %r, but there are %d "
                    "measurements.",
                    self.covmat_file,
                    self.cov.shape,
                    len(self.z),
                )
            if not np.allclose(self.cov, self.cov.T):
                raise LoggedError(
                    self.log,
                    "The covariance matrix in '%s' is not symmetric.",
                    self.covmat_file,
                )
        else:
            if data.shape[1] < 3:
                raise LoggedError(
                    self.log,
                    "No errors (3rd column) in '%s', and no covmat_file given.",
                    self.data_file,
                )
            self.cov = np.diag(data[:, 2] ** 2)
        if self.systematics:
            if not self._is_default("data_file") or self.covmat_file:
                self.log.info(
                    "Adding the systematic terms %r to the errors of the data. If these "
                    "already include them, set 'systematics: []'.",
                    self.systematics,
                )
            with open(self._path("systematics_file")) as f:
                columns = f.readline().lstrip("#").split()
            table = self._load_table("systematics_file")
            for name in self.systematics:
                if name not in columns[1:]:
                    raise LoggedError(
                        self.log,
                        "Unknown systematic '%s'. Available ones are %r.",
                        name,
                        columns[1:],
                    )
                fractional = np.interp(
                    self.z, table[:, 0], table[:, columns.index(name)] / 100
                )
                self.cov = self.cov + np.outer(
                    self.H_data * fractional, self.H_data * fractional
                )
        self.invcov = np.linalg.inv(self.cov)

    def _is_default(self, option):
        return getattr(self, option) == self.get_defaults().get(option)

    def _path(self, option):
        """
        Full path of a file given by ``option``: relative to the data folder (see
        ``initialize``) if it is the class default, and to the current working directory
        otherwise.
        """
        filename = getattr(self, option)
        if os.path.isabs(filename):
            return filename
        if self._is_default(option):
            return os.path.join(self._data_folder, filename)
        return os.path.abspath(filename)

    def _load_table(self, option):
        path = self._path(option)
        try:
            with open(path) as f:
                first_data_line = next(
                    line for line in f if line.strip() and not line.startswith("#")
                )
            delimiter = "," if "," in first_data_line else None
            # the first numerical columns (e.g. ignoring a reference string at the end)
            n_numeric = 0
            for value in first_data_line.split(delimiter):
                try:
                    float(value)
                except ValueError:
                    break
                n_numeric += 1
            return np.atleast_2d(
                np.loadtxt(
                    path, comments="#", delimiter=delimiter, usecols=range(n_numeric)
                )
            )
        except (OSError, StopIteration, ValueError) as excpt:
            raise LoggedError(
                self.log, "Could not read file '%s' (%s): %s", path, option, excpt
            ) from excpt

    def get_requirements(self):
        return {"Hubble": {"z": self.z}}

    def logp(self, **params_values):
        H_theory = self.provider.get_Hubble(self.z, units="km/s/Mpc")
        residual = self.H_data - H_theory
        return -0.5 * residual @ self.invcov @ residual
