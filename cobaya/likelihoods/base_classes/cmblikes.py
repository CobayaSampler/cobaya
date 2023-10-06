"""
.. module:: cmblikes

:Synopsis: Definition of the CMBlikes class for CMB real or simulated data.
:Author: Antony Lewis (adapted to Cobaya by Jesus Torrado with little modification)

# Load CosmoMC format .dataset files
# AL July 2014 - Dec 2017

"""
# Global
import os
import numpy as np
from typing import List
from getdist import ParamNames, IniFile
from scipy.linalg import sqrtm

# Local
from cobaya.log import LoggedError
from cobaya.likelihoods.base_classes import DataSetLikelihood

CMB_keys = ['tt', 'te', 'ee', 'bb']


class CMBlikes(DataSetLikelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "CMB"

    # used to form spectra names, e.g. AmapxBmap
    map_separator: str = 'x'

    def get_requirements(self):
        # State requisites to the theory code
        requested_cls = [self.field_names[i] + self.field_names[j]
                         for i, j in zip(*np.where(self.cl_lmax != 0))]
        # l_max has to take into account the window function of the lensing
        # so we check the computed l_max ("l_max" option) is higher than the requested one
        requested_l_max = int(np.max(self.cl_lmax))
        if (getattr(self, "l_max", None) or np.inf) < requested_l_max:
            raise LoggedError(
                self.log, "You are setting a very low l_max. "
                          "The likelihood value will probably not be correct. "
                          "Make sure to make 'l_max'>=%d", requested_l_max)
        self.l_max = max(requested_l_max, getattr(self, "l_max", 0) or 0)
        return {"Cl": {cl: self.l_max for cl in requested_cls}}

    def typeIndex(self, field):
        return self.field_names.index(field)

    def PairStringToMapIndices(self, S):
        if len(S) == 2:
            if self.has_map_names:
                raise LoggedError(self.log, 'CL names must use MAP1xMAP2 names')
            return self.map_names.index(S[0]), self.map_names.index(S[1])
        else:
            try:
                i = S.index(self.map_separator)
            except ValueError:
                raise LoggedError(self.log, 'invalid spectrum name %s' % S)
            return self.map_names.index(S[0:i]), self.map_names.index(S[i + 1:])

    def PairStringToUsedMapIndices(self, used_index, S):
        i1, i2 = self.PairStringToMapIndices(S)
        i1 = used_index[i1]
        i2 = used_index[i2]
        if i2 > i1:
            return i2, i1
        else:
            return i1, i2

    def UseString_to_cols(self, L):
        cl_i_j = self.UseString_to_Cl_i_j(L, self.map_used_index)
        cols = -np.ones(cl_i_j.shape[1], dtype=int)
        for i in range(cl_i_j.shape[1]):
            i1, i2 = cl_i_j[:, i]
            if i1 == -1 or i2 == -1:
                continue
            ix = 0
            for ii in range(self.nmaps):
                for jj in range(ii + 1):
                    if ii == i1 and jj == i2:
                        cols[i] = ix
                    ix += 1
        return cols

    def UseString_to_Cl_i_j(self, S, used_index):
        if not isinstance(S, (list, tuple)):
            S = S.split()
        cl_i_j = np.zeros((2, len(S)), dtype=int)
        for i, p in enumerate(S):
            cl_i_j[:, i] = self.PairStringToUsedMapIndices(used_index, p)
        return cl_i_j

    def MapPair_to_Theory_i_j(self, order, pair):
        i = self.map_fields[order[pair[0]]]
        j = self.map_fields[order[pair[1]]]
        if i <= j:
            return i, j
        else:
            return j, i

    def Cl_used_i_j_name(self, pair):
        return self.Cl_i_j_name(self.used_map_order, pair)

    def Cl_i_j_name(self, names, pair):
        name1 = names[pair[0]]
        name2 = names[pair[1]]
        if self.has_map_names:
            return name1 + self.map_separator + name2
        else:
            return name1 + name2

    def get_cols_from_order(self, order):
        # Converts string Order = TT TE EE XY... or AAAxBBB AAAxCCC BBxCC
        # into indices into array of power spectra (and -1 if not present)
        cols = np.empty(self.ncl, dtype=int)
        cols[:] = -1
        names = order.strip().split()
        ix = 0
        for i in range(self.nmaps):
            for j in range(i + 1):
                name = self.Cl_used_i_j_name([i, j])
                if name not in names and i != j:
                    name = self.Cl_used_i_j_name([j, i])
                if name in names:
                    if cols[ix] != -1:
                        raise LoggedError(self.log,
                                          'get_cols_from_order: duplicate CL type')
                    cols[ix] = names.index(name)
                ix += 1
        return cols

    def elements_to_matrix(self, X, M):
        ix = 0
        for i in range(self.nmaps):
            M[i, 0:i] = X[ix:ix + i]
            M[0:i, i] = X[ix:ix + i]
            ix += i
            M[i, i] = X[ix]
            ix += 1

    def matrix_to_elements(self, M, X):
        ix = 0
        for i in range(self.nmaps):
            X[ix:ix + i + 1] = M[i, 0:i + 1]
            ix += i + 1

    def read_cl_array(self, ini, file_stem, return_full=False):
        # read file of CL or bins (indexed by L)
        filename = ini.relativeFileName(file_stem + '_file')
        cl = np.zeros((self.ncl, self.nbins_used))
        order = ini.string(file_stem + '_order', '')
        if not order:
            incols = last_top_comment(filename)
            if not incols:
                raise LoggedError(self.log, 'No column order given for ' + filename)
        else:
            incols = 'L ' + order
        cols = self.get_cols_from_order(incols)
        data = np.loadtxt(filename)
        Ls = data[:, 0].astype(int)
        if self.binned:
            Ls -= 1
        for i, L in enumerate(Ls):
            if self.bin_min <= L <= self.bin_max:
                for ix in range(self.ncl):
                    if cols[ix] != -1:
                        cl[ix, L - self.bin_min] = data[i, cols[ix]]
        if Ls[-1] < self.bin_max:
            raise LoggedError(
                self.log, 'CMBLikes_ReadClArr: C_l file does not go up to maximum used: '
                          '%s', self.bin_max)
        if return_full:
            return incols.split(), data, cl
        else:
            return cl

    def read_bin_windows(self, ini, file_stem):
        bins = BinWindows(self.pcl_lmin, self.pcl_lmax, self.nbins_used, self.ncl)
        in_cl = ini.split(file_stem + '_in_order')
        out_cl = ini.split(file_stem + '_out_order', in_cl)
        bins.cols_in = self.UseString_to_Cl_i_j(in_cl, self.map_required_index)
        bins.cols_out = self.UseString_to_cols(out_cl)
        norder = bins.cols_in.shape[1]
        if norder != bins.cols_out.shape[0]:
            raise LoggedError(
                self.log, '_in_order and _out_order must have same number of entries')
        bins.binning_matrix = np.zeros(
            (norder, self.nbins_used, self.pcl_lmax - self.pcl_lmin + 1))
        windows = ini.relativeFileName(file_stem + '_files')
        for b in range(self.nbins_used):
            window = np.loadtxt(windows % (b + 1 + self.bin_min))
            err = False
            for i, L in enumerate(window[:, 0].astype(int)):
                if self.pcl_lmin <= L <= self.pcl_lmax:
                    bins.binning_matrix[:, b, L - self.pcl_lmin] = window[i, 1:]
                else:
                    err = err or np.count_nonzero(window[i, 1:])
            if err:
                self.log.warning('%s %u outside pcl_lmin-cl_max range: %s' %
                                 (file_stem, b, windows % (b + 1)))
        if ini.hasKey(file_stem + '_fix_cl_file'):
            raise LoggedError(self.log, 'fix_cl_file not implemented yet')
        return bins

    def init_map_cls(self, nmaps, order):
        if nmaps != len(order):
            raise LoggedError(self.log, 'init_map_cls: size mismatch')

        class CrossPowerSpectrum:
            map_ij: List[int]
            theory_ij: List[int]
            CL: np.ndarray

        cls = np.empty((nmaps, nmaps), dtype=object)
        for i in range(nmaps):
            for j in range(i + 1):
                CL = CrossPowerSpectrum()
                cls[i, j] = CL
                CL.map_ij = [order[i], order[j]]
                CL.theory_ij = self.MapPair_to_Theory_i_j(order, [i, j])
                CL.CL = np.zeros(self.pcl_lmax - self.pcl_lmin + 1)
        return cls

    def init_params(self, ini):
        self.field_names = getattr(self, 'field_names', ['T', 'E', 'B', 'P'])
        self.tot_theory_fields = len(self.field_names)
        self.map_names = ini.split('map_names', default=[])
        self.has_map_names = bool(self.map_names)
        if self.has_map_names:
            # e.g. have multiple frequencies for given field measurement
            map_fields = ini.split('map_fields')
            if len(map_fields) != len(self.map_names):
                raise LoggedError(self.log,
                                  'number of map_fields does not match map_names')
            self.map_fields = [self.typeIndex(f) for f in map_fields]
        else:
            self.map_names = self.field_names
            self.map_fields = np.arange(len(self.map_names), dtype=int)
        fields_use = ini.split('fields_use', [])
        if len(fields_use):
            index_use = [self.typeIndex(f) for f in fields_use]
            use_theory_field = [i in index_use for i in range(self.tot_theory_fields)]
        else:
            if not self.has_map_names:
                raise LoggedError(self.log, 'must have fields_use or map_names')
            use_theory_field = [True] * self.tot_theory_fields
        maps_use = ini.split('maps_use', [])
        if len(maps_use):
            if any(not i for i in use_theory_field):
                self.log.warning('maps_use overrides fields_use')
            self.use_map = [False] * len(self.map_names)
            for j, map_used in enumerate(maps_use):
                if map_used in self.map_names:
                    self.use_map[self.map_names.index(map_used)] = True
                else:
                    raise LoggedError(self.log, 'maps_use item not found - %s' % map_used)
        else:
            self.use_map = [use_theory_field[self.map_fields[i]]
                            for i in range(len(self.map_names))]
        # Bandpowers can depend on more fields than are actually used in likelihood
        # e.g. for correcting leakage or other linear corrections
        self.require_map = self.use_map[:]
        if self.has_map_names:
            if ini.hasKey('fields_required'):
                raise LoggedError(self.log, 'use maps_required not fields_required')
            maps_use = ini.split('maps_required', [])
        else:
            maps_use = ini.split('fields_required', [])
        if len(maps_use):
            for j, map_used in enumerate(maps_use):
                if map_used in self.map_names:
                    self.require_map[self.map_names.index(map_used)] = True
                else:
                    raise LoggedError(self.log, 'required item not found %s' % map_used)
        self.required_theory_field = [False for _ in self.field_names]
        for i in range(len(self.map_names)):
            if self.require_map[i]:
                self.required_theory_field[self.map_fields[i]] = True
        self.ncl_used = 0  # set later reading covmat
        self.like_approx = ini.string('like_approx', 'gaussian')
        self.nmaps = np.count_nonzero(self.use_map)
        self.nmaps_required = np.count_nonzero(self.require_map)
        self.required_order = np.zeros(self.nmaps_required, dtype=int)
        self.map_required_index = -np.ones(len(self.map_names), dtype=int)
        ix = 0
        for i in range(len(self.map_names)):
            if self.require_map[i]:
                self.map_required_index[i] = ix
                self.required_order[ix] = i
                ix += 1
        self.map_used_index = -np.ones(len(self.map_names), dtype=int)
        ix = 0
        self.used_map_order = []
        for i, map_name in enumerate(self.map_names):
            if self.use_map[i]:
                self.map_used_index[i] = ix
                self.used_map_order.append(map_name)
                ix += 1
        self.ncl = (self.nmaps * (self.nmaps + 1)) // 2
        self.pcl_lmax = ini.int('cl_lmax')
        self.pcl_lmin = ini.int('cl_lmin')
        self.binned = ini.bool('binned', True)
        if self.binned:
            self.nbins = ini.int('nbins')
            self.bin_min = ini.int('use_min', 1) - 1
            self.bin_max = ini.int('use_max', self.nbins) - 1
            # needed by read_bin_windows:
            self.nbins_used = self.bin_max - self.bin_min + 1
            self.bins = self.read_bin_windows(ini, 'bin_window')
        else:
            if self.nmaps != self.nmaps_required:
                raise LoggedError(
                    self.log, 'unbinned likelihood must have nmaps==nmaps_required')
            self.nbins = self.pcl_lmax - self.pcl_lmin + 1
            if self.like_approx != 'exact':
                self.log.warning('Unbinned likelihoods untested in this version')
            self.bin_min = ini.int('use_min', self.pcl_lmin)
            self.bin_max = ini.int('use_max', self.pcl_lmax)
            self.nbins_used = self.bin_max - self.bin_min + 1
        self.full_bandpower_headers, self.full_bandpowers, self.bandpowers = \
            self.read_cl_array(ini, 'cl_hat', return_full=True)
        if self.like_approx == 'HL':
            self.cl_fiducial = self.read_cl_array(ini, 'cl_fiducial')
        else:
            self.cl_fiducial = None
        includes_noise = ini.bool('cl_hat_includes_noise', False)
        self.cl_noise = None
        if self.like_approx != 'gaussian' or includes_noise:
            self.cl_noise = self.read_cl_array(ini, 'cl_noise')
            if not includes_noise:
                self.bandpowers += self.cl_noise
            elif self.like_approx == 'gaussian':
                self.bandpowers -= self.cl_noise
        self.cl_lmax = np.zeros((self.tot_theory_fields, self.tot_theory_fields))
        for i in range(self.tot_theory_fields):
            if self.required_theory_field[i]:
                self.cl_lmax[i, i] = self.pcl_lmax
        if self.required_theory_field[0] and self.required_theory_field[1]:
            self.cl_lmax[1, 0] = self.pcl_lmax

        if self.like_approx != 'gaussian':
            cl_fiducial_includes_noise = ini.bool('cl_fiducial_includes_noise', False)
        else:
            cl_fiducial_includes_noise = False
        self.bandpower_matrix = np.zeros((self.nbins_used, self.nmaps, self.nmaps))
        self.noise_matrix = self.bandpower_matrix.copy()
        self.fiducial_sqrt_matrix = self.bandpower_matrix.copy()
        if self.cl_fiducial is not None and not cl_fiducial_includes_noise:
            self.cl_fiducial += self.cl_noise
        for b in range(self.nbins_used):
            self.elements_to_matrix(self.bandpowers[:, b], self.bandpower_matrix[b, :, :])
            if self.cl_noise is not None:
                self.elements_to_matrix(self.cl_noise[:, b], self.noise_matrix[b, :, :])
            if self.cl_fiducial is not None:
                self.elements_to_matrix(self.cl_fiducial[:, b],
                                        self.fiducial_sqrt_matrix[b, :, :])
                self.fiducial_sqrt_matrix[b, :, :] = (
                    sqrtm(self.fiducial_sqrt_matrix[b, :, :]))
        if self.like_approx == 'exact':
            self.fsky = ini.float('fullsky_exact_fksy')
        else:
            self.cov = self.ReadCovmat(ini)
            self.covinv = np.linalg.inv(self.cov)
        if 'linear_correction_fiducial_file' in ini.params:
            self.fid_correction = self.read_cl_array(ini, 'linear_correction_fiducial')
            self.linear_correction = self.read_bin_windows(ini,
                                                           'linear_correction_bin_window')
        else:
            self.linear_correction = None
        if ini.hasKey('nuisance_params'):
            s = ini.relativeFileName('nuisance_params')
            self.nuisance_params = ParamNames(s)
            if ini.hasKey('calibration_param'):
                raise Exception('calibration_param not allowed with nuisance_params')
            if ini.hasKey('calibration_paramname'):
                self.calibration_param = ini.string('calibration_paramname')
            else:
                self.calibration_param = None
        elif ini.string('calibration_param', ''):
            s = ini.relativeFileName('calibration_param')
            if '.paramnames' not in s:
                raise LoggedError(
                    self.log, 'calibration_param must be paramnames file unless '
                              'nuisance_params also specified')
            self.nuisance_params = ParamNames(s)
            self.calibration_param = self.nuisance_params.list()[0]
        else:
            self.calibration_param = None
        if ini.hasKey('log_calibration_prior'):
            self.log.warning('log_calibration_prior in .dataset ignored, '
                             'set separately in .yaml file')
        self.aberration_coeff = ini.float('aberration_coeff', 0.0)

        self.map_cls = self.init_map_cls(self.nmaps_required, self.required_order)

    def ReadCovmat(self, ini):
        """Read the covariance matrix, and the array of which CL are in the covariance,
        which then defines which set of bandpowers are used
        (subject to other restrictions)."""
        covmat_cl = ini.string('covmat_cl', allowEmpty=False)
        self.full_cov = np.loadtxt(ini.relativeFileName('covmat_fiducial'))
        covmat_scale = ini.float('covmat_scale', 1.0)
        cl_in_index = self.UseString_to_cols(covmat_cl)
        self.ncl_used = np.sum(cl_in_index >= 0)
        self.cl_used_index = np.zeros(self.ncl_used, dtype=int)
        cov_cl_used = np.zeros(self.ncl_used, dtype=int)
        ix = 0
        for i, index in enumerate(cl_in_index):
            if index >= 0:
                self.cl_used_index[ix] = index
                cov_cl_used[ix] = i
                ix += 1
        if self.binned:
            num_in = len(cl_in_index)
            pcov = np.empty((self.nbins_used * self.ncl_used,
                             self.nbins_used * self.ncl_used))
            for binx in range(self.nbins_used):
                for biny in range(self.nbins_used):
                    pcov[binx * self.ncl_used: (binx + 1) * self.ncl_used,
                    biny * self.ncl_used: (biny + 1) * self.ncl_used] = (
                            covmat_scale * self.full_cov[
                        np.ix_((binx + self.bin_min) * num_in + cov_cl_used,
                               (biny + self.bin_min) * num_in + cov_cl_used)])
        else:
            raise LoggedError(self.log, 'unbinned covariance not implemented yet')
        return pcov

    # noinspection PyTypeChecker
    def writeData(self, froot):  # pragma: no cover
        np.savetxt(froot + '_cov.dat', self.cov)
        np.savetxt(froot + '_bandpowers.dat', self.full_bandpowers,
                   header=" ".join(self.full_bandpower_headers))
        self.bins.write(froot, 'bin')
        if self.linear_correction is not None:
            self.linear_correction.write(froot, 'linear_correction_bin')

            with open(froot + '_lensing_fiducial_correction', 'w', encoding="utf-8") as f:
                f.write("#%4s %12s \n" % ('bin', 'PP'))
                for b in range(self.nbins):
                    f.write("%5u %12.5e\n" % (b + 1, self.fid_correction[b]))

    def diag_sigma(self):
        return np.sqrt(np.diag(self.full_cov))

    def plot_lensing(self, column='PP', ells=None, units="muK2", ax=None):
        if not np.count_nonzero(self.map_cls):
            raise LoggedError(
                self.log, "No Cl's have been computed yet. "
                          "Make sure you have evaluated the likelihood.")
        try:
            Cl_theo = self.provider.get_Cl(ell_factor=True, units=units)
            Cl = Cl_theo.get(column.lower())
        except KeyError:
            raise LoggedError(self.log, "'%s' spectrum has not been computed." % column)
        import matplotlib.pyplot as plt
        lbin = self.full_bandpowers[:, self.full_bandpower_headers.index('L_av')]
        binned_phicl_err = self.diag_sigma()
        ax = ax or plt.gca()
        bandpowers = self.full_bandpowers[:, self.full_bandpower_headers.index('PP')]
        if 'L_min' in self.full_bandpower_headers:
            lmin = self.full_bandpowers[:, self.full_bandpower_headers.index('L_min')]
            lmax = self.full_bandpowers[:, self.full_bandpower_headers.index('L_max')]
            ax.errorbar(lbin, bandpowers,
                        yerr=binned_phicl_err, xerr=[lbin - lmin, lmax - lbin], fmt='o')
        else:
            ax.errorbar(lbin, bandpowers, yerr=binned_phicl_err, fmt='o')
        if ells is not None:
            Cl = Cl[ells]
        else:
            ells = Cl_theo["ell"]
        ax.plot(ells, Cl, color='k')
        ax.set_xlim([2, ells[-1]])
        return ax

    def get_binned_map_cls(self, Cls, corrections=True):
        band = self.bins.bin(Cls)
        if self.linear_correction is not None and corrections:
            band += self.linear_correction.bin(Cls) - self.fid_correction.T
        return band

    def get_theory_map_cls(self, Cls, data_params=None):
        for i in range(self.nmaps_required):
            for j in range(i + 1):
                CL = self.map_cls[i, j]
                combination = "".join([self.field_names[k] for k in CL.theory_ij]).lower()
                cls = Cls.get(combination)
                if cls is not None:
                    CL.CL[:] = cls[self.pcl_lmin:self.pcl_lmax + 1]
                else:
                    CL.CL[:] = 0
        self.adapt_theory_for_maps(self.map_cls, data_params or {})

    def adapt_theory_for_maps(self, cls, data_params):
        if self.aberration_coeff:
            self.add_aberration(cls)
        self.add_foregrounds(cls, data_params)
        if self.calibration_param is not None and self.calibration_param in data_params:
            for i in range(self.nmaps_required):
                for j in range(i + 1):
                    CL = cls[i, j]
                    if CL is not None:
                        if CL.theory_ij[0] <= 2 and CL.theory_ij[1] <= 2:
                            CL.CL /= data_params[self.calibration_param] ** 2

    def add_foregrounds(self, cls, data_params):
        pass

    def add_aberration(self, cls):
        # adapted from CosmoMC function by Christian Reichardt
        ells = np.arange(self.pcl_lmin, self.pcl_lmax + 1)
        cl_norm = ells * (ells + 1)
        for i in range(self.nmaps_required):
            for j in range(i + 1):
                CL = cls[i, j]
                if CL is not None:
                    if CL.theory_ij[0] <= 2 and CL.theory_ij[1] <= 2:
                        # first get Cl instead of Dl
                        cl_deriv = CL.CL / cl_norm
                        # second take derivative dCl/dl
                        cl_deriv[1:-1] = (cl_deriv[2:] - cl_deriv[:-2]) / 2
                        # handle endpoints approximately
                        cl_deriv[0] = cl_deriv[1]
                        cl_deriv[-1] = cl_deriv[-2]
                        # reapply to Dl's.
                        # note never took 2pi out, so not putting it back either
                        cl_deriv *= cl_norm
                        # also multiply by ell since really wanted ldCl/dl
                        cl_deriv *= ells
                        CL.CL += self.aberration_coeff * cl_deriv

    def write_likelihood_data(self, filename, data_params=None):
        cls = self.init_map_cls(self.nmaps_required, self.required_order)
        self.add_foregrounds(cls, data_params or {})
        with open(filename, 'w', encoding="utf-8") as f:
            cols = []
            for i in range(self.nmaps_required):
                for j in range(i + 1):
                    cols.append(self.Cl_i_j_name(self.map_names, cls[i, j].map_ij))
            f.write('#    L' + ("%17s " * len(cols)) % tuple(cols) + '\n')
            for b in range(self.pcl_lmin, self.pcl_lmax + 1):
                c = [b]
                for i in range(self.nmaps_required):
                    for j in range(i + 1):
                        c.append(cls[i, j].CL[b - self.pcl_lmin])
                f.write(("%I5 " + "%17.8e " * len(cols)) % tuple(c))

    @staticmethod
    def transform(C, Chat, Cfhalf):
        # HL transformation of the matrices
        if C.shape[0] == 1:
            rat = Chat[0, 0] / C[0, 0]
            C[0, 0] = (np.sign(rat - 1) *
                       np.sqrt(2 * np.maximum(0, rat - np.log(rat) - 1)) *
                       Cfhalf[0, 0] ** 2)
            return
        diag, U = np.linalg.eigh(C)
        rot = U.T.dot(Chat).dot(U)
        roots = np.sqrt(diag)
        for i, root in enumerate(roots):
            rot[i, :] /= root
            rot[:, i] /= root
        U.dot(rot.dot(U.T), rot)
        diag, rot = np.linalg.eigh(rot)
        diag = np.sign(diag - 1) * np.sqrt(2 * np.maximum(0, diag - np.log(diag) - 1))
        Cfhalf.dot(rot, U)
        for i, d in enumerate(diag):
            rot[:, i] = U[:, i] * d
        rot.dot(U.T, C)

    def exact_chi_sq(self, C, Chat, L):
        if C.shape[0] == 1:
            return ((2 * L + 1) * self.fsky *
                    (Chat[0, 0] / C[0, 0] - 1 - np.log(Chat[0, 0] / C[0, 0])))
        else:
            M = np.linalg.inv(C).dot(Chat)
            return ((2 * L + 1) * self.fsky *
                    (np.trace(M) - self.nmaps - np.linalg.slogdet(M)[1]))

    def logp(self, **data_params):
        cls = self.provider.get_Cl(ell_factor=True)
        return self.log_likelihood(cls, **data_params)

    # noinspection PyUnboundLocalVariable
    def log_likelihood(self, dls, **data_params):
        r"""
        Get log likelihood from the dls (CMB C_l scaled by L(L+1)/2\pi)

        :param dls: dictionary of d_l ('tt', etc)
        :param data_params: likelihood nuisance parameters
        :return: log likelihood
        """
        self.get_theory_map_cls(dls, data_params)
        C = np.empty((self.nmaps, self.nmaps))
        big_x = np.empty(self.nbins_used * self.ncl_used)
        vecp = np.empty(self.ncl)
        chisq = 0
        if self.binned:
            binned_theory = self.get_binned_map_cls(self.map_cls)
        else:
            Cs = np.zeros((self.nbins_used, self.nmaps, self.nmaps))
            for i in range(self.nmaps):
                for j in range(i + 1):
                    CL = self.map_cls[i, j]
                    if CL is not None:
                        Cs[:, i, j] = CL.CL[self.bin_min - self.pcl_lmin:
                                            self.bin_max - self.pcl_lmin + 1]
                        Cs[:, j, i] = CL.CL[self.bin_min - self.pcl_lmin:
                                            self.bin_max - self.pcl_lmin + 1]
        for b in range(self.nbins_used):
            if self.binned:
                self.elements_to_matrix(binned_theory[b, :], C)
            else:
                C[:, :] = Cs[b, :, :]
            if self.cl_noise is not None:
                C += self.noise_matrix[b]
            if self.like_approx == 'exact':
                chisq += self.exact_chi_sq(
                    C, self.bandpower_matrix[b], self.bin_min + b)
                continue
            elif self.like_approx == 'HL':
                try:
                    self.transform(
                        C, self.bandpower_matrix[b], self.fiducial_sqrt_matrix[b])
                except np.linalg.LinAlgError:
                    self.log.debug("Likelihood computation failed.")
                    return -np.inf
            elif self.like_approx == 'gaussian':
                C -= self.bandpower_matrix[b]
            self.matrix_to_elements(C, vecp)
            big_x[b * self.ncl_used:(b + 1) * self.ncl_used] = vecp[
                self.cl_used_index]
        if self.like_approx == 'exact':
            return -0.5 * chisq
        return -0.5 * self._fast_chi_squared(self.covinv, big_x)


class BinWindows:
    cols_in: np.ndarray
    cols_out: np.ndarray
    binning_matrix: np.ndarray

    def __init__(self, lmin, lmax, nbins, ncl):
        self.lmin = lmin
        self.lmax = lmax
        self.nbins = nbins
        self.ncl = ncl

    def bin(self, theory_cl, cls=None):
        if cls is None:
            cls = np.zeros((self.nbins, self.ncl))
        for i, ((x, y), ix_out) in enumerate(zip(self.cols_in.T, self.cols_out)):
            cl = theory_cl[x, y]
            if cl is not None and ix_out >= 0:
                cls[:, ix_out] += np.dot(self.binning_matrix[i, :, :], cl.CL)
        return cls

    def write(self, froot, stem):
        if not os.path.exists(froot + stem + '_window'):
            os.mkdir(froot + '_window')
        for b in range(self.nbins):
            with open(froot + stem + '_window/window%u.dat' % (b + 1),
                      'w', encoding="utf-8") as f:
                for L in np.arange(self.lmin[b], self.lmax[b] + 1):
                    f.write(
                        ("%5u " + "%10e" * len(self.cols_in) + "\n") %
                        (L, self.binning_matrix[:, b, L]))


def last_top_comment(fname):
    result = None
    with open(fname, encoding="utf-8-sig") as f:
        while x := f.readline():
            if x := x.strip():
                if x[0] != '#':
                    return result
                result = x[1:].strip()
    return None


def white_noise_from_muK_arcmin(noise_muK_arcmin):
    return (noise_muK_arcmin * np.pi / 180 / 60.) ** 2


def save_cl_dict(filename, array_dict, lmin=2, lmax=None,
                 cl_dict_lmin=0):  # pragma: no cover
    """
    Save a Cobaya dict of CL to a text file, with each line starting with L.

    :param filename: filename to save
    :param array_dict: dictionary of power spectra
    :param lmin: minimum L to save
    :param lmax: maximum L to save
    :param cl_dict_lmin: L to start output in file (usually 0 or 2)
    """
    cols = []
    labels = []
    for key in CMB_keys:
        if key in array_dict:
            lmax = lmax or array_dict[key].shape[0] - 1 + cl_dict_lmin
            cols.append(array_dict[key][lmin - cl_dict_lmin:lmax - cl_dict_lmin + 1])
            labels.append(key.upper())
    if 'pp' in array_dict:
        lmax = lmax or array_dict['pp'].shape[0] - 1 + cl_dict_lmin
        cols.append(array_dict['pp'][lmin - cl_dict_lmin:lmax - cl_dict_lmin + 1])
        labels.append('PP')
    ls = np.arange(lmin, lmax + 1)
    np.savetxt(filename, np.vstack((ls,) + tuple(cols)).T,
               fmt=['%4u'] + ['%12.7e'] * len(cols),
               header=' L ' + ' '.join(['{:13s}'.format(lab) for lab in labels]))


def make_forecast_cmb_dataset(fiducial_Cl, output_root, output_dir=None,
                              noise_muK_arcmin_T=None,
                              noise_muK_arcmin_P=None,
                              NoiseVar=None, ENoiseFac=2, fwhm_arcmin=None,
                              lmin=2, lmax=None, fsky=1.0,
                              lens_recon_noise=None, cl_dict_lmin=0):  # pragma: no cover
    """
    Make a simulated .dataset and associated files with 'data' set at the input fiducial
    model. Uses the exact full-sky log-likelihood, scaled by fsky.

    If you want to use numerical N_L CMB noise files, you can just replace the noise
    .dat text file produced by this function.

    :param fiducial_Cl: dictionary of Cls to use, combination of tt, te, ee, bb, pp;
                        note te must be included with tt and ee when using them
    :param output_root: root name for output files, e.g. 'my_sim1'
    :param output_dir: output directory
    :param noise_muK_arcmin_T: temperature noise in muK-arcmin
    :param noise_muK_arcmin_P: polarization noise in muK-arcmin
    :param NoiseVar: alternatively if noise_muK_arcmin_T is None, effective
        isotropic noise variance for the temperature (N_L=NoiseVar with no beam)
    :param ENoiseFac: factor by which polarization noise variance is higher than
                NoiseVar (usually 2, for Planck about 4
                        as only half the detectors polarized)
    :param fwhm_arcmin: beam fwhm in arcminutes
    :param lmin: l_min
    :param lmax: l_max
    :param fsky: sky fraction
    :param lens_recon_noise: optional array, starting at L=0, for the
       pp lensing reconstruction noise, in [L(L+1)]^2C_L^phi/2pi units
    :param cl_dict_lmin: l_min for the arrays in fiducial_Cl
    :return: IniFile that was saved
    """
    ini = IniFile()
    dataset = ini.params

    cl_keys = fiducial_Cl.keys()
    use_CMB = set(cl_keys).intersection(CMB_keys)
    use_lensing = lens_recon_noise is not None

    if use_CMB:
        if NoiseVar is None:
            if noise_muK_arcmin_T is None:
                raise ValueError('Must specify noise')
            NoiseVar = white_noise_from_muK_arcmin(noise_muK_arcmin_T)
            if noise_muK_arcmin_P is not None:
                ENoiseFac = (noise_muK_arcmin_P / noise_muK_arcmin_T) ** 2
        elif noise_muK_arcmin_T is not None or noise_muK_arcmin_P is not None:
            raise ValueError('Specific either noise_muK_arcmin or NoiseVar')
        fields_use = ''
        if 'tt' in cl_keys or 'te' in cl_keys:
            fields_use = 'T'
        if 'ee' in cl_keys or 'te' in cl_keys:
            fields_use += ' E'
        if 'bb' in cl_keys:
            fields_use += ' B'
        if 'pp' in cl_keys and use_lensing:
            fields_use += ' P'
        if 'tt' in cl_keys and 'ee' in cl_keys and 'te' not in cl_keys:
            raise ValueError('Input power spectra should have te if using tt and ee -'
                             'using the exact likelihood requires the full covariance.')
    else:
        fields_use = 'P'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset['fields_use'] = fields_use

    if use_CMB:
        fwhm = fwhm_arcmin / 60
        xlc = 180 * np.sqrt(8. * np.log(2.)) / np.pi
        sigma2 = (fwhm / xlc) ** 2
        noise_cols = 'TT           EE          BB'
        if use_lensing:
            noise_cols += '          PP'
    elif use_lensing:
        noise_cols = 'PP'
    else:
        raise ValueError('Must use CMB or lensing C_L')
    noise_file = output_root + '_Noise.dat'
    with open(os.path.join(output_dir, noise_file), 'w') as f:
        f.write('#L %s\n' % noise_cols)

        for ell in range(lmin, lmax + 1):
            noises = []
            if use_CMB:
                # noinspection PyUnboundLocalVariable
                noise_cl = ell * (ell + 1.) / 2 / np.pi * NoiseVar * np.exp(
                    ell * (ell + 1) * sigma2)
                noises += [noise_cl, ENoiseFac * noise_cl, ENoiseFac * noise_cl]
            if use_lensing:
                noises += [lens_recon_noise[ell]]
            f.write("%d " % ell + " ".join("%E" % elem for elem in noises) + "\n")

    dataset['fullsky_exact_fksy'] = fsky
    dataset['dataset_format'] = 'CMBLike2'
    dataset['like_approx'] = 'exact'

    dataset['cl_lmin'] = lmin
    dataset['cl_lmax'] = lmax

    dataset['binned'] = False

    dataset['cl_hat_includes_noise'] = False

    save_cl_dict(os.path.join(output_dir, output_root + '.dat'),
                 fiducial_Cl, cl_dict_lmin=cl_dict_lmin)
    dataset['cl_hat_file'] = output_root + '.dat'
    dataset['cl_noise_file '] = noise_file

    ini.saveFile(os.path.join(output_dir, output_root + '.dataset'))
    return ini
