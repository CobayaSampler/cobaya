"""
.. module:: planck_2018_CamSpec

:Synopsis: Definition of python-native CamSpec 2018 likelihood (not official Planck product)
:Author: Antony Lewis (from CamSpec f90 source by GPE, StG and AL)

This python version loads the covariance, and cuts it as requested (then inverting). It
can take a few seconds the first time it is loaded, but the inverse can be cached.
The Planck likelihood code (clik) is not required.

Use dataset_params : { 'use_cl': '100x100 143x143 217x217 143x217'} to use e.g. just TT ,
or other combination with TE and EE.

Set use_range to string representation of L range to use, e.g. 50-100, 200-500, 1470-2500,
or pass a dictionary of ranges for each spectrum.

It is used by the 2018 and more recent CamSpec Planck likelihoods.

"""

import hashlib
import os

import numpy as np
import scipy

from cobaya.likelihoods.base_classes.DataSetLikelihood import DataSetLikelihood
from cobaya.output import FileLock

use_cache = True


def range_to_ells(use_range):
    """splits range string like '2-5 7 15-3000' into list of specific numbers"""

    if isinstance(use_range, str):
        ranges = []
        for ell_range in use_range.split():
            if "-" in ell_range:
                mn, mx = (int(x) for x in ell_range.split("-"))
                ranges.append(range(mn, mx + 1))
            else:
                ranges.append(int(ell_range))
        return np.concatenate(ranges)
    else:
        return use_range


class Planck2018CamSpecPython(DataSetLikelihood):
    install_options = {
        "github_repository": "CobayaSampler/planck_native_data",
        "github_release": "v1",
        "asset": "CamSpec2018.zip",
        "directory": "planck_2018_CamSpec_native",
    }

    type = "CMB"

    @classmethod
    def get_bibtex(cls):
        if not (res := super().get_bibtex()):
            from cobaya.likelihoods.base_classes.planck_clik import Planck2018Clik

            return Planck2018Clik.get_bibtex()
        return res

    def read_normalized(self, filename, pivot=None):
        # arrays all based at L=0, in L(L+1)/2pi units
        self.mpi_debug("Loading: %s", filename)
        dat = np.loadtxt(filename)
        assert int(dat[0, 0]) == 2
        dat = np.hstack(([0, 0], dat[:, 1]))
        if pivot is not None:
            assert pivot < dat.shape[0] + 2
            dat /= dat[pivot]
        return dat

    def init_params(self, ini, silent=False):
        spectra = np.loadtxt(ini.relativeFileName("cl_hat_file"))
        covmat_cl = ini.split("covmat_cl")
        self.use_cl = ini.split("use_cl", covmat_cl)
        if ini.hasKey("use_range"):
            used_ell = ini.params["use_range"]
            if isinstance(used_ell, dict):
                self.mpi_info("Using range %s", used_ell)
                used_ell = {key: range_to_ells(value) for key, value in used_ell.items()}
            else:
                if silent:
                    self.mpi_info("CamSpec using range: %s", used_ell)
                used_ell = range_to_ells(used_ell)
        else:
            used_ell = None
        data_vector = []
        nX = 0
        used_indices = []
        with open(ini.relativeFileName("data_ranges"), encoding="utf-8-sig") as f:
            lines = f.readlines()
            while not lines[-1].strip():
                lines = lines[:-1]
            self.Nspec = len(lines)
            lmin = np.zeros(self.Nspec, dtype=int)
            lmax = np.zeros(self.Nspec, dtype=int)
            self.cl_names = []
            self.ell_ranges = np.empty(self.Nspec, dtype=object)
            self.used_sizes = np.zeros(self.Nspec, dtype=int)
            for i, line in enumerate(lines):
                items = line.split()
                tp = items[0]
                self.cl_names.append(tp)
                lmin[i], lmax[i] = (int(x) for x in items[1:])
                if lmax[i] and lmax[i] >= lmin[i]:
                    n = lmax[i] - lmin[i] + 1
                    data_vector.append(spectra[lmin[i] : lmax[i] + 1, i])
                    if tp in self.use_cl:
                        if used_ell is not None and (
                            not isinstance(used_ell, dict) or tp in used_ell
                        ):
                            if isinstance(used_ell, dict):
                                ells = used_ell[tp]
                            else:
                                ells = used_ell
                            self.ell_ranges[i] = np.array(
                                [L for L in range(lmin[i], lmax[i] + 1) if L in ells],
                                dtype=int,
                            )
                            used_indices.append(self.ell_ranges[i] + (nX - lmin[i]))
                        else:
                            used_indices.append(range(nX, nX + n))
                            self.ell_ranges[i] = range(lmin[i], lmax[i] + 1)
                        self.used_sizes[i] = len(self.ell_ranges[i])
                    else:
                        lmax[i] = -1
                    nX += n

        self.cl_used = np.array(
            [name in self.use_cl for name in self.cl_names], dtype=bool
        )
        covfile = ini.relativeFileName("covmat_fiducial")
        with open(covfile, "rb") as cov_f:
            cov = np.fromfile(cov_f, dtype=[np.float32, np.float64]["64.bin" in covfile])
        assert nX**2 == cov.shape[0]
        used_indices = np.concatenate(used_indices)
        self.data_vector = np.concatenate(data_vector)[used_indices]
        self.cov = cov.reshape(nX, nX)[np.ix_(used_indices, used_indices)].astype(
            np.float64
        )
        if not silent:
            for name, mn, mx in zip(self.cl_names, lmin, lmax):
                if name in self.use_cl:
                    self.mpi_info("L-range for %s: %s %s", name, mn, mx)
            self.mpi_info("Number of data points: %s", self.cov.shape[0])
        self.lmax = lmax
        self.lmin = lmin
        max_l = np.max(self.lmax)
        self.ls = np.arange(max_l + 1)
        self.llp1 = self.ls * (self.ls + 1)

        if np.any(self.cl_used[:4]):
            pivot = 3000
            self.sz_143 = self.read_normalized(ini.relativeFileName("sz143file"), pivot)[
                : max_l + 1
            ]
            self.ksz = self.read_normalized(ini.relativeFileName("kszfile"), pivot)[
                : max_l + 1
            ]
            self.tszxcib = self.read_normalized(
                ini.relativeFileName("tszxcibfile"), pivot
            )[: max_l + 1]

            self.cib_217 = self.read_normalized(
                ini.relativeFileName("cib217file"), pivot
            )[: max_l + 1]

            self.dust = np.vstack(
                (
                    self.read_normalized(ini.relativeFileName("dust100file"))[
                        : max_l + 1
                    ],
                    self.read_normalized(ini.relativeFileName("dust143file"))[
                        : max_l + 1
                    ],
                    self.read_normalized(ini.relativeFileName("dust217file"))[
                        : max_l + 1
                    ],
                    self.read_normalized(ini.relativeFileName("dust143x217file"))[
                        : max_l + 1
                    ],
                )
            )
            self.lnrat = self.ls * 0
            l_min = np.min(lmin[self.cl_used])
            self.lnrat[l_min:] = np.log(self.ls[l_min:] / np.float64(pivot))

        cache_file = self.dataset_filename.replace(
            ".dataset",
            "_covinv_%s.npy" % hashlib.md5(str(ini.params).encode("utf8")).hexdigest(),
        )
        if not use_cache:
            self.covinv = np.linalg.inv(self.cov)
        else:
            with FileLock(cache_file, log=self.log, wait=True):
                if os.path.exists(cache_file):
                    self.covinv = np.load(cache_file).astype(np.float64)
                else:
                    self.covinv = np.linalg.inv(self.cov)
                    np.save(cache_file, self.covinv.astype(np.float32))

    def get_foregrounds(self, data_params):
        sz_bandpass100_nom143 = 2.022
        cib_bandpass143_nom143 = 1.134
        sz_bandpass143_nom143 = 0.95
        cib_bandpass217_nom217 = 1.33

        Aps = np.empty(4)
        Aps[0] = data_params["aps100"]
        Aps[1] = data_params["aps143"]
        Aps[2] = data_params["aps217"]
        Aps[3] = data_params["psr"] * np.sqrt(Aps[1] * Aps[2])
        Aps *= 1e-6 / 9  # scaling convention

        Adust = np.atleast_2d(
            [
                data_params["dust100"],
                data_params["dust143"],
                data_params["dust217"],
                data_params["dust143x217"],
            ]
        ).T

        acib143 = data_params.get("acib143", -1)
        acib217 = data_params["acib217"]
        cibr = data_params["cibr"]
        ncib = data_params["ncib"]
        cibrun = data_params["cibrun"]

        asz143 = data_params["asz143"]
        xi = data_params["xi"]
        aksz = data_params["aksz"]

        lmax = np.max(self.lmax)

        cl_cib = np.exp(ncib * self.lnrat + cibrun * self.lnrat**2 / 2) * self.cib_217
        if acib143 < 0:
            # fix 143 from 217
            acib143 = 0.094 * acib217 / cib_bandpass143_nom143 * cib_bandpass217_nom217
            # The above came from ratioing Paolo's templates, which were already
            # colour-corrected, and assumed perfect correlation

        ksz = aksz * self.ksz
        C_foregrounds = np.empty((4, lmax + 1))
        # 100
        C_foregrounds[0, :] = ksz + asz143 * sz_bandpass100_nom143 * self.sz_143

        # 143
        A_sz_143_bandpass = asz143 * sz_bandpass143_nom143
        A_cib_143_bandpass = acib143 * cib_bandpass143_nom143
        zCIB = A_cib_143_bandpass * cl_cib
        C_foregrounds[1, :] = (
            zCIB
            + ksz
            + A_sz_143_bandpass * self.sz_143
            - 2.0 * np.sqrt(A_cib_143_bandpass * A_sz_143_bandpass) * xi * self.tszxcib
        )

        # 217
        A_cib_217_bandpass = acib217 * cib_bandpass217_nom217
        zCIB = A_cib_217_bandpass * cl_cib
        C_foregrounds[2, :] = zCIB + ksz

        # 143x217
        zCIB = np.sqrt(A_cib_143_bandpass * A_cib_217_bandpass) * cl_cib
        C_foregrounds[3, :] = (
            cibr * zCIB
            + ksz
            - np.sqrt(A_cib_217_bandpass * A_sz_143_bandpass) * xi * self.tszxcib
        )

        # Add dust and point sources
        C_foregrounds += Adust * self.dust + np.outer(Aps, self.llp1)

        return C_foregrounds

    def get_cals(self, data_params):
        calPlanck = data_params.get("A_planck", 1) ** 2
        cal0 = data_params.get("cal0", 1)
        cal2 = data_params.get("cal2", 1)
        calTE = data_params.get("calTE", 1)
        calEE = data_params.get("calEE", 1)
        return np.array([cal0, 1, cal2, np.sqrt(cal2), calTE, calEE]) * calPlanck

    def chi_squared(self, CT, CTE, CEE, data_params):
        cals = self.get_cals(data_params)
        if np.any(self.cl_used[:4]):
            foregrounds = self.get_foregrounds(data_params)
        delta_vector = self.data_vector.copy()
        ix = 0
        for i, (cal, n) in enumerate(zip(cals, self.used_sizes)):
            if n > 0:
                if i <= 3:
                    delta_vector[ix : ix + n] -= (
                        CT[self.ell_ranges[i]] + foregrounds[i][self.ell_ranges[i]]
                    ) / cal
                elif i == 4:
                    delta_vector[ix : ix + n] -= CTE[self.ell_ranges[i]] / cal
                elif i == 5:
                    delta_vector[ix : ix + n] -= CEE[self.ell_ranges[i]] / cal
                ix += n
        return self._fast_chi_squared(self.covinv, delta_vector)

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True)
        return -0.5 * self.chi_squared(
            Cls.get("tt"), Cls.get("te"), Cls.get("ee"), data_params
        )

    def get_requirements(self):
        # State requisites to the theory code
        l_max = np.max(self.lmax)
        used = []
        if np.any(self.cl_used[:4]):
            used += ["tt"]
        if "TE" in self.use_cl:
            used += ["te"]
        if "EE" in self.use_cl:
            used += ["ee"]
        return {"Cl": {cl: l_max for cl in used}}

    def coadded_TT(
        self,
        data_params=None,
        foregrounds=None,
        cals=None,
        want_cov=True,
        data_vector=None,
    ):
        nTT = np.sum(self.used_sizes[:4])
        assert nTT
        if foregrounds is not None and cals is not None and data_params is not None:
            raise ValueError("data_params not used")
        if foregrounds is None:
            assert data_params is not None
            foregrounds = self.get_foregrounds(data_params)
        if cals is None:
            assert data_params is not None
            cals = self.get_cals(data_params)
        if data_vector is None:
            data_vector = self.data_vector
        delta_vector = data_vector[:nTT].copy()
        cal_vector = np.zeros(delta_vector.shape)
        lmin = np.min([min(r) for r in self.ell_ranges[:4]])
        lmax = np.max([max(r) for r in self.ell_ranges[:4]])
        n_p = lmax - lmin + 1
        LS = np.zeros(delta_vector.shape, dtype=int)
        ix = 0
        for i, (cal, n) in enumerate(zip(cals[:4], self.used_sizes[:4])):
            if n > 0:
                delta_vector[ix : ix + n] -= foregrounds[i][self.ell_ranges[i]] / cal
                LS[ix : ix + n] = self.ell_ranges[i]
                cal_vector[ix : ix + n] = cal
                ix += n
        pcov = np.zeros((n_p, n_p))
        d = self.covinv[:nTT, :nTT].dot(delta_vector)
        dL = np.zeros(n_p)
        ix1 = 0
        ell_offsets = [LS - lmin for LS in self.ell_ranges[:4]]
        contiguous = not any(
            np.count_nonzero(LS - np.arange(LS[0], LS[-1] + 1, dtype=int))
            for LS in self.ell_ranges[:4]
        )
        for i, (cal, LS, n) in enumerate(zip(cals[:4], ell_offsets, self.used_sizes[:4])):
            dL[LS] += d[ix1 : ix1 + n] / cal
            ix = 0
            for cal2, r in zip(cals[:4], ell_offsets):
                if contiguous:
                    pcov[LS[0] : LS[0] + n, r[0] : r[0] + len(r)] += self.covinv[
                        ix1 : ix1 + n, ix : ix + len(r)
                    ] / (cal2 * cal)
                else:
                    pcov[np.ix_(LS, r)] += self.covinv[
                        ix1 : ix1 + n, ix : ix + len(r)
                    ] / (cal2 * cal)
                ix += len(r)
            ix1 += n

        CTot = np.zeros(self.ls[-1] + 1)
        if want_cov:
            pcovinv = np.linalg.inv(pcov)
            CTot[lmin : lmax + 1] = pcovinv.dot(dL)
            return CTot, pcovinv
        else:
            try:
                CTot[lmin : lmax + 1] = scipy.linalg.solve(pcov, dL, assume_a="pos")
            except Exception:
                CTot[lmin : lmax + 1] = np.linalg.solve(pcov, dL)
            return CTot

    def get_weights(self, data_params):
        # get weights for each temperature spectrum as function of L
        ix = 0
        f = self.get_foregrounds(data_params) * 0
        weights = []
        for i in range(4):
            ells = self.ell_ranges[i]
            vec = np.zeros(self.data_vector.shape)
            vec[ix : ix + len(ells)] = 1
            Ti = self.coadded_TT(
                data_params, data_vector=vec, want_cov=False, foregrounds=f
            )
            weights.append((ells, Ti[ells]))
            ix += len(ells)
        return weights

    def diff(self, spec1, spec2, data_params):
        """
        Get difference (residual) between frequency spectra and the covariance
        :param spec1: name of spectrum 1
        :param spec2:  name of spectrum 2
        :param data_params: dictionary of parameters
        :return: ell range array, difference array, covariance matrix
        """
        foregrounds = self.get_foregrounds(data_params)
        cals = self.get_cals(data_params)
        i = self.cl_names.index(spec1)
        j = self.cl_names.index(spec2)
        off1 = np.sum(self.used_sizes[:i])
        off2 = np.sum(self.used_sizes[:j])
        lmax = np.min([max(r) for r in self.ell_ranges[[i, j]]])
        lmin = np.max([min(r) for r in self.ell_ranges[[i, j]]])

        diff = np.zeros(self.ls[-1] + 1)
        diff[self.ell_ranges[i]] = (
            self.data_vector[off1 : off1 + self.used_sizes[i]] * cals[i]
            - foregrounds[i][self.ell_ranges[i]]
        )
        diff[self.ell_ranges[j]] -= (
            self.data_vector[off2 : off2 + self.used_sizes[j]] * cals[j]
            - foregrounds[j][self.ell_ranges[j]]
        )
        cov = self.cov
        n_p = lmax - lmin + 1
        off1 += lmin - np.min(self.ell_ranges[i])
        off2 += lmin - np.min(self.ell_ranges[j])
        pcov = (
            cals[i] ** 2 * cov[off1 : off1 + n_p, off1 : off1 + n_p]
            + cals[j] ** 2 * cov[off2 : off2 + n_p, off2 : off2 + n_p]
            - cals[i]
            * cals[j]
            * (
                cov[off2 : off2 + n_p, off1 : off1 + n_p]
                + cov[off1 : off1 + n_p, off2 : off2 + n_p]
            )
        )
        return range(lmin, lmax + 1), diff[lmin : lmax + 1], pcov
