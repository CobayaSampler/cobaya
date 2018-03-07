
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Global
import os
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import logging

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import HandledException
from cobaya.conventions import _path_install
from cobaya.tools import get_path_to_installation


class _bao_prototype(Likelihood):

    def initialise(self):
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        # If no path specified, use the modules path
        data_file_path = (self.path or
                          os.path.join(get_path_to_installation(), "data/sdss_dr12"))
        if not data_file_path:
            self.log.error("No path given to BAO data. Set the likelihood property "
                           "'path' or the common property '%s'.", _path_install)
            raise HandledException
        # Load "measurements file" and covmat of requested
        try:
            self.data = pd.read_csv(os.path.join(data_file_path, self.measurements_file),
                                    header=None, index_col=None, sep="\s+")
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
        # Functions to get the corresponding theoretical prediction
        self.theory_fun = lambda z, observable: ({
            "DM_over_rs":
                (1+z)*self.theory.get_angular_diameter_distance(z)/self.rs(),
            "Hz_rs":
                self.theory.get_h_of_z(z)*self.rs(),
#            "DV_over_rs":
#                "this%Calculator%BAO_D_v"(z)/self.rs(),
#            "Hz_rs_103":
#                self.theory.get_h_of_z(z)*self.rs()*1e-3,
#            "rs_over_DV":
#                self.rs()/"this%Calculator%BAO_D_v"(z),
#            "Az":
#                "this%Acoustic(CMB,z)",
#            "DA_over_rs":
#                self.theory.get_angular_diameter_distance(z)/self.rs(),
#            "F_AP":
#                ((1+z)*self.theory.get_angular_diameter_distance(z)*
#                 self.theory.get_h_of_z(z)),
#            "f_sigma8":
#                "Theory%growth_z%Value"(z)
        }[observable])
        # Requisites
        zs = {obs:self.data.loc[self.data["observable"] == obs, "z"].values
              for obs in self.data["observable"].unique()}
        theory_reqs = {
            "DM_over_rs": {
                "angular_diameter_distance": {"redshifts": zs.get("DM_over_rs", None)},
                "rdrag": None},
            "Hz_rs": {
                "h_of_z": {"redshifts": zs.get("Hz_rs", None), "units": "km/s/Mpc"},
                "rdrag": None},
            "DV_over_rs": {
                "BAO_D_v(z)": None, "rdrag": None},
            "Hz_rs_103": {
                "h_of_z": {"redshifts": zs.get("Hz_rs_103", None), "units": "km/s/Mpc"},
                "rdrag": None},
            "rs_over_DV": {
                "BAO_D_v(z)": None},
            "Az": {
                "Acoustic(CMB,z)": None},
            "DA_over_rs": {
                "angular_diameter_distance": {"redshifts": zs.get("DA_over_rs", None)},
                "rdrag": None},
            "F_AP": {
                "angular_diameter_distance": {"redshifts": zs.get("F_AP", None)},
                "h_of_z": {"redshifts": zs.get("F_AP", None), "units": "km/s/Mpc"}},
            "f_sigma8": {
                "growth_z%Value(z)"}}
        #    this%needs_powerspectra =  any(this%type_bao == f_sigma8)
        #    if (this%needs_powerspectra) then
        #        this%needs_exact_z =  .true.
        #        allocate(this%exact_z(this%num_bao))
        #        this%exact_z = this%bao_z
        #    end if
        requisites = {}
        if self.has_type:
            for obs in self.data["observable"].unique():
                requisites.update(theory_reqs[obs])
        self.theory.needs(requisites)

    def rs(self):
        return self.theory.get_param("rdrag") * self.rs_rescale

    def logp(self, **params_values):
        theory = np.array([self.theory_fun(z,obs) for z, obs
                           in zip(self.data["z"], self.data["observable"])]).T[0]
        return self.norm.logpdf(theory)
