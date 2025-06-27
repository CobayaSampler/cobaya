"""
.. module:: planck_pliklite

:Synopsis: Definition of python-native nuisance-free CMB likelihoods: e.g. plik_lite
:Author: Erminia Calabrese, Antony Lewis

Nuisance-marginalized likelihood, based on covarianced and binned CL, with settings read
from .dataset file.

"""

import os

import numpy as np

from cobaya.likelihoods.base_classes.DataSetLikelihood import DataSetLikelihood

cl_names = ["tt", "te", "ee"]


class PlanckPlikLite(DataSetLikelihood):
    install_options = {
        "github_repository": "CobayaSampler/planck_native_data",
        "github_release": "v1",
        "asset": "plik_lite_2018_AL.zip",
        "directory": "planck_2018_pliklite_native",
    }

    bibtex_file = "planck2018.bibtex"
    type = "CMB"

    def init_params(self, ini):
        self.use_cl = [c.lower() for c in ini.list("use_cl")]
        assert len(self.use_cl)
        nbintt = ini.int("nbintt")
        nbinte = ini.int("nbinte")
        nbinee = ini.int("nbinee")
        lmax = ini.int("lmax")
        use_bins = ini.int_list("use_bins", [])
        bins_for_L_range = ini.int_list("bins_for_L_range", [])
        self.calibration_param = ini.string("calibration_param", "A_planck")
        assert len(bins_for_L_range) in [0, 2]

        data = np.loadtxt(ini.relativeFileName("data"))

        bin_lmin_offset = ini.int("bin_lmin_offset")
        self.blmin = (
            np.loadtxt(ini.relativeFileName("blmin")).astype(int) + bin_lmin_offset
        )
        self.blmax = (
            np.loadtxt(ini.relativeFileName("blmax")).astype(int) + bin_lmin_offset
        )
        self.lav = (self.blmin + self.blmax) // 2
        weights = np.loadtxt(ini.relativeFileName("weights"))
        ls = np.arange(len(weights)) + bin_lmin_offset
        self.bin_lmin_offset = bin_lmin_offset
        weights *= 2 * np.pi / ls / (ls + 1)  # we work directly with  DL not CL
        self.weights = np.hstack((np.zeros(bin_lmin_offset), weights))

        self.nbins = nbintt + nbinee + nbinte

        bin_cov_file = ini.relativeFileName("cov_file_binary")

        if os.path.exists(bin_cov_file):
            from scipy.io import FortranFile

            f = FortranFile(bin_cov_file, "r")
            cov = f.read_reals(dtype=float).reshape((self.nbins, self.nbins))
            cov = np.tril(cov) + np.tril(cov, -1).T  # make symmetric
        else:
            cov = np.loadtxt(ini.relativeFileName("cov_file"))
            # full n row x n col matrix converted from fortran binary

        self.lmax = lmax

        maxbin = max(nbintt, nbinte, nbinee)
        assert cov.shape[0] == self.nbins
        self.lav = self.lav[:maxbin]

        if len(use_bins) and np.max(use_bins) >= maxbin:
            raise ValueError("use_bins has bin index out of range")
        if len(bins_for_L_range):
            if len(use_bins):
                raise ValueError("can only use one bin filter")
            use_bins = [
                use_bin
                for use_bin in range(maxbin)
                if bins_for_L_range[0]
                <= (self.blmin[use_bin] + self.blmax[use_bin]) / 2
                <= bins_for_L_range[1]
            ]
            print(
                "Actual L range: {} - {}".format(
                    self.blmin[use_bins[0]], self.blmax[use_bins[-1]]
                )
            )

        self.used = np.zeros(3, dtype=bool)
        self.used_bins = []
        used_indices = []
        offset = 0
        self.bandpowers = {}
        self.errors = {}

        for i, (cl, nbin) in enumerate(zip(cl_names, [nbintt, nbinte, nbinee])):
            self.used[i] = cl_names[i] in self.use_cl
            sc = self.lav[:nbin] * (self.lav[:nbin] + 1) / 2.0 / np.pi
            self.bandpowers[cl] = data[offset : offset + nbin, 1] * sc
            self.errors[cl] = data[offset : offset + nbin, 2] * sc
            if self.used[i]:
                if len(use_bins):
                    self.used_bins.append(
                        np.array(
                            [use_bin for use_bin in use_bins if use_bin < nbin], dtype=int
                        )
                    )
                else:
                    self.used_bins.append(np.arange(nbin, dtype=int))
                used_indices.append(self.used_bins[-1] + offset)
            else:
                self.used_bins.append(np.arange(0, dtype=int))
            offset += nbin
        self.used_indices = np.hstack(used_indices)
        assert self.nbins == cov.shape[0] == data.shape[0]
        self.X_data = data[self.used_indices, 1]
        self.cov = cov[np.ix_(self.used_indices, self.used_indices)]
        self.invcov = np.linalg.inv(self.cov)

    def get_requirements(self):
        # State requisites to the theory code
        self.l_max = self.lmax
        return {"Cl": {cl: self.l_max for cl in self.use_cl}}

    def binning_matrix(self, ix=0):
        # not used by main likelihood code
        lmax = self.blmax[self.used_bins[ix][-1]]
        lmin = self.blmin[self.used_bins[ix][0]]
        m = np.zeros((len(self.used_bins[ix]), lmax - lmin + 1))
        for i in self.used_bins[ix]:
            m[i, self.blmin[i] - lmin : self.blmax[i] + 1 - lmin] = self.weights[
                self.blmin[i] : self.blmax[i] + 1
            ]
        return lmin, lmax, m

    def get_chi_squared(self, L0, ctt, cte, cee, A_planck=1):
        cl = np.empty(self.used_indices.shape)
        ix = 0
        for tp, cell in enumerate([ctt, cte, cee]):
            for i in self.used_bins[tp]:
                cl[ix] = np.dot(
                    cell[self.blmin[i] - L0 : self.blmax[i] - L0 + 1],
                    self.weights[self.blmin[i] : self.blmax[i] + 1],
                )
                ix += 1
        cl /= A_planck**2
        diff = self.X_data - cl
        return self._fast_chi_squared(self.invcov, diff)

    def chi_squared(self, c_l_arr, A_planck=1):
        r"""
        Get chi squared from CL array from file

        :param c_l_arr: file of L and L(L+1)CL/2\pi values for C_TT, C_TE, C_EE
        :param A_planck: calibration parameter
        :return: chi-squared
        """
        L0 = int(c_l_arr[0, 0])
        return self.get_chi_squared(
            L0, c_l_arr[:, 1], c_l_arr[:, 2], c_l_arr[:, 3], A_planck
        )

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True)
        return -0.5 * self.get_chi_squared(
            0,
            Cls.get("tt"),
            Cls.get("te"),
            Cls.get("ee"),
            data_params[self.calibration_param],
        )
