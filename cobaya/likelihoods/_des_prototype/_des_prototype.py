"""
.. module:: _des_prototype

:Synopsis: DES likelihood.
           Well tested and agrees with likelihoods in DES chains for fixednu.
:Author: Antony Lewis (little changes for Cobaya by Jesus Torrado)

.. warning::

   This is still in **beta**. Coming soon!

.. |br| raw:: html

   <br />

.. note::

   **If you use any of these likelihoods, please cite them as:**
   |br|
   Abbott, T. M. C. and others,
   `Dark Energy Survey year 1 results: Cosmological constraints from
   galaxy clustering and weak lensing`
   `(arXiv:1708.01530) <https://arxiv.org/abs/1708.01530>`_


Likelihoods of the DES Y1 data release, described in the paper mentioned above:

- ``des_y1_clustering``
- ``des_y1_shear``
- ``des_y1_galaxy_galaxy``
- ``des_y1_joint`` (a shortcut for the combination of the previous three)

Usage
-----

To use any of the DES likelihoods, you simply need to mention them in the
``likelihood`` block, or add them using the :doc:`input generator <cosmo_basic_runs>`.

The corresponding nuisance parameters will be added automatically,
so you don't have to care about listing them in the ``params`` block.

The nuisance parameters and their default priors can be found in the ``defaults.yaml``
files in the folder corresponding to each likelihood. They are not reproduced here because
of their length.


Installation
------------

This likelihood can be installed automatically as explained in :doc:`installation_cosmo`.

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Global
import os
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline
import copy
import logging

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import HandledException
from cobaya.conventions import _path_install, _c_km_s
from cobaya.install import download_github_release

# DES data types
def_DES_types = ['xip', 'xim', 'gammat', 'wtheta']


def get_def_cuts():
    ranges = {}
    for tp in def_DES_types:
        ranges[tp] = np.empty((6, 6), dtype=object)
    ranges['xip'][1][1] = [7.195005, 250.0]
    ranges['xip'][1][2] = [7.195005, 250.0]
    ranges['xip'][1][3] = [5.715196, 250.0]
    ranges['xip'][1][4] = [5.715196, 250.0]
    ranges['xip'][2][1] = [7.195005, 250.0]
    ranges['xip'][2][2] = [4.539741, 250.0]
    ranges['xip'][2][3] = [4.539741, 250.0]
    ranges['xip'][2][4] = [4.539741, 250.0]
    ranges['xip'][3][1] = [5.715196, 250.0]
    ranges['xip'][3][2] = [4.539741, 250.0]
    ranges['xip'][3][3] = [3.606045, 250.0]
    ranges['xip'][3][4] = [3.606045, 250.0]
    ranges['xip'][4][1] = [5.715196, 250.0]
    ranges['xip'][4][2] = [4.539741, 250.0]
    ranges['xip'][4][3] = [3.606045, 250.0]
    ranges['xip'][4][4] = [3.606045, 250.0]
    ranges['xim'][1][1] = [90.579750, 250.0]
    ranges['xim'][1][2] = [71.950053, 250.0]
    ranges['xim'][1][3] = [71.950053, 250.0]
    ranges['xim'][1][4] = [71.950053, 250.0]
    ranges['xim'][2][1] = [71.950053, 250.0]
    ranges['xim'][2][2] = [57.151958, 250.0]
    ranges['xim'][2][3] = [57.151958, 250.0]
    ranges['xim'][2][4] = [45.397414, 250.0]
    ranges['xim'][3][1] = [71.950053, 250.0]
    ranges['xim'][3][2] = [57.151958, 250.0]
    ranges['xim'][3][3] = [45.397414, 250.0]
    ranges['xim'][3][4] = [45.397414, 250.0]
    ranges['xim'][4][1] = [71.950053, 250.0]
    ranges['xim'][4][2] = [45.397414, 250.0]
    ranges['xim'][4][3] = [45.397414, 250.0]
    ranges['xim'][4][4] = [36.060448, 250.0]
    ranges['gammat'][1][1] = [64.0, 250.0]
    ranges['gammat'][1][2] = [64.0, 250.0]
    ranges['gammat'][1][3] = [64.0, 250.0]
    ranges['gammat'][1][4] = [64.0, 250.0]
    ranges['gammat'][2][1] = [40.0, 250.0]
    ranges['gammat'][2][2] = [40.0, 250.0]
    ranges['gammat'][2][3] = [40.0, 250.0]
    ranges['gammat'][2][4] = [40.0, 250.0]
    ranges['gammat'][3][1] = [30.0, 250.0]
    ranges['gammat'][3][2] = [30.0, 250.0]
    ranges['gammat'][3][3] = [30.0, 250.0]
    ranges['gammat'][3][4] = [30.0, 250.0]
    ranges['gammat'][4][1] = [24.0, 250.0]
    ranges['gammat'][4][2] = [24.0, 250.0]
    ranges['gammat'][4][3] = [24.0, 250.0]
    ranges['gammat'][4][4] = [24.0, 250.0]
    ranges['gammat'][5][1] = [21.0, 250.0]
    ranges['gammat'][5][2] = [21.0, 250.0]
    ranges['gammat'][5][3] = [21.0, 250.0]
    ranges['gammat'][5][4] = [21.0, 250.0]
    ranges['wtheta'][1][1] = [43.0, 250.0]
    ranges['wtheta'][2][2] = [27.0, 250.0]
    ranges['wtheta'][3][3] = [20.0, 250.0]
    ranges['wtheta'][4][4] = [16.0, 250.0]
    ranges['wtheta'][5][5] = [14.0, 250.0]
    for tp in def_DES_types:
        ranges[tp] = ranges[tp][1:, 1:]
    return ranges


class _des_prototype(Likelihood):
    def initialize(self):
        self.l_max = self.l_max or int(50000 * self.acc)
        if os.path.isabs(self.data_file):
            data_file = self.data_file
        else:
            # If no path specified, use the modules path
            if self.path:
                data_file_path = self.path
            elif self.path_install:
                data_file_path = os.path.join(self.path_install, "data", des_data_name)
            else:
                self.log.error(
                    "No path given to the DES data. Set the likelihood property 'path' "
                    "or the common property '%s'.", _path_install)
                raise HandledException
            data_file = os.path.normpath(os.path.join(data_file_path, self.data_file))
        try:
            if data_file.endswith(".fits"):
                if self.dataset_params:
                    self.log.error("'dataset_params' can only be specified "
                                   "for .dataset (not .fits) file.")
                    raise HandledException
                self.load_fits_data(data_file)
            else:
                self.load_dataset(data_file, self.dataset_params)
        except IOError:
            self.log.error("The data file '%s' could not be found at '%s'. "
                           "Check your paths!", self.data_file, data_file_path)
            raise HandledException
        self.initialize_postload()

    def load_dataset(self, filename, dataset_params):
        from getdist import IniFile
        ini = IniFile(filename)
        ini.params.update(dataset_params or {})
        self.indices = []
        self.used_indices = []
        self.used_items = []
        self.fullcov = np.loadtxt(ini.relativeFileName('cov_file'))
        ntheta = ini.int('num_theta_bins')
        self.theta_bins = np.loadtxt(ini.relativeFileName('theta_bins_file'))
        self.iintrinsic_alignment_model = ini.string('intrinsic_alignment_model')
        self.data_types = ini.string('data_types').split()
        self.used_types = ini.list('used_data_types', self.data_types)
        with open(ini.relativeFileName('data_selection')) as f:
            header = f.readline()
            assert ('#  type bin1 bin2 theta_min theta_max' == header.strip())
            lines = f.readlines()
        ranges = {}
        for tp in self.data_types:
            ranges[tp] = np.empty((6, 6), dtype=object)
        for line in lines:
            items = line.split()
            if items[0] in self.used_types:
                bin1, bin2 = [int(x) - 1 for x in items[1:3]]
                ranges[items[0]][bin1][bin2] = [np.float64(x) for x in items[3:]]
        self.ranges = ranges
        self.nzbins = ini.int('num_z_bins')  # for lensing sources
        self.nwbins = ini.int('num_gal_bins', 0)  # for galaxies
        maxbin = max(self.nzbins, self.nwbins)
        cov_ix = 0
        self.bin_pairs = []
        self.data_arrays = []
        self.thetas = []
        for i, tp in enumerate(self.data_types):
            xi = np.loadtxt(ini.relativeFileName('measurements[%s]' % tp))
            bin1 = xi[:, 0].astype(np.int) - 1
            bin2 = xi[:, 1].astype(np.int) - 1
            tbin = xi[:, 2].astype(np.int) - 1
            corr = np.empty((maxbin, maxbin), dtype=np.object)
            corr[:, :] = None
            self.data_arrays.append(corr)
            self.bin_pairs.append([])
            for f1, f2, ix, dat in zip(bin1, bin2, tbin, xi[:, 3]):
                self.indices.append((i, f1, f2, ix))
                if not (f1, f2) in self.bin_pairs[i]:
                    self.bin_pairs[i].append((f1, f2))
                    corr[f1, f2] = np.zeros(ntheta)
                corr[f1, f2][ix] = dat
                if ranges[tp][f1, f2] is not None:
                    mn, mx = ranges[tp][f1, f2]
                    if self.theta_bins[ix] > mn and self.theta_bins[ix] < mx:
                        self.thetas.append(self.theta_bins[ix])
                        self.used_indices.append(cov_ix)
                        self.used_items.append(self.indices[-1])
                cov_ix += 1
        nz_source = np.loadtxt(ini.relativeFileName('nz_file'))
        self.zmid = nz_source[:, 1]
        self.zbin_sp = []
        for b in range(self.nzbins):
            self.zbin_sp += [UnivariateSpline(self.zmid, nz_source[:, b + 3], s=0)]
        nz_lens = np.loadtxt(ini.relativeFileName('nz_gal_file'))
        assert (np.array_equal(nz_lens[:, 1], self.zmid))
        self.zbin_w_sp = []
        for b in range(self.nwbins):
            self.zbin_w_sp += [UnivariateSpline(self.zmid, nz_lens[:, b + 3], s=0)]
        self.zmax = self.zmid[-1]
        # k_max actually computed, assumes extrapolated beyond that
        self.k_max = ini.float('kmax', 15)

    def load_fits_data(self, filename, ranges=None):
        import astropy.io.fits as fits
        if ranges is None:
            ranges = get_def_cuts()
        hdulist = fits.open(filename)
        self.indices = []
        self.intrinsic_alignment_model = 'DES1YR'
        self.used_indices = []
        self.used_items = []
        self.ranges = ranges
        self.fullcov = hdulist['COVMAT'].data
        cov_ix = 0
        self.bin_pairs = []
        self.data_types = def_DES_types
        self.used_types = def_DES_types
        for i, tp in enumerate(def_DES_types):
            xi = hdulist[tp].data
            self.bin_pairs.append([])
            for f1, f2, ix, dat, theta in zip(xi.field(0) - 1, xi.field(1) - 1,
                                              xi.field(2), xi.field(3), xi.field(4)):
                self.indices.append((i, f1, f2, ix))
                if not (f1, f2) in self.bin_pairs[i]:
                    self.bin_pairs[i].append((f1, f2))
                mn, mx = ranges[tp][f1, f2]
                if theta > mn and theta < mx:
                    self.used_indices.append(cov_ix)
                    self.used_items.append(self.indices[-1])
                cov_ix += 1
        self.nzbins = 4  # for lensing sources
        corrs_p = np.empty((self.nzbins, self.nzbins), dtype=np.object)
        corrs_p[:, :] = None
        corrs_m = np.empty((self.nzbins, self.nzbins), dtype=np.object)
        corrs_m[:, :] = None
        ntheta = 20
        self.theta_bins = np.empty(ntheta)
        for xi, corrs in zip(
                [hdulist['xip'].data, hdulist['xim'].data], [corrs_p, corrs_m]):
            for f1, f2, ix, dat, theta in zip(xi.field(0) - 1, xi.field(1) - 1,
                                              xi.field(2), xi.field(3), xi.field(4)):
                if corrs[f1, f2] is None:
                    corrs[f1, f2] = np.zeros(ntheta)
                if f1 == 0 and f2 == 0:
                    self.theta_bins[ix] = theta
                assert (np.abs(self.theta_bins[ix] / theta - 1) < 2e-3)
                corrs[f1, f2][ix] = dat
        self.nwbins = 5  # for galaxies
        corrs_w = np.empty((self.nwbins, self.nwbins), dtype=np.object)
        for f1 in range(self.nwbins):
            corrs_w[f1, f1] = np.empty(ntheta)
        wdata = hdulist['wtheta'].data
        for f1, f2, ix, dat, theta in zip(wdata.field(0) - 1, wdata.field(1) - 1,
                                          wdata.field(2), wdata.field(3), wdata.field(4)):
            assert (f1 == f2)
            assert (np.abs(self.theta_bins[ix] / theta - 1) < 2e-3)
            corrs_w[f1, f2][ix] = dat
        corrs_t = np.empty((self.nwbins, self.nwbins), dtype=np.object)
        corrs_t[:, :] = None
        tdata = hdulist['gammat'].data
        for f1, f2, ix, dat, theta in zip(tdata.field(0) - 1, tdata.field(1) - 1,
                                          tdata.field(2), tdata.field(3), tdata.field(4)):
            assert (np.abs(self.theta_bins[ix] / theta - 1) < 2e-3)
            if corrs_t[f1, f2] is None:
                corrs_t[f1, f2] = np.empty(ntheta)
            corrs_t[f1, f2][ix] = dat
        self.data_arrays = [corrs_p, corrs_m, corrs_t, corrs_w]
        self.zmid = hdulist['NZ_SOURCE'].data.field('Z_MID')
        self.zbin_sp = []
        for b in range(self.nzbins):
            self.zbin_sp += [UnivariateSpline(self.zmid,
                                              hdulist['NZ_SOURCE'].data.field(b + 3), s=0)]
        zmid_w = hdulist['NZ_LENS'].data.field('Z_MID')
        assert (np.array_equal(zmid_w, self.zmid))
        self.zbin_w_sp = []
        for b in range(self.nwbins):
            self.zbin_w_sp += [UnivariateSpline(self.zmid,
                                                hdulist['NZ_LENS'].data.field(b + 3), s=0)]
        self.zmax = self.zmid[351]  # last non-zero

    def initialize_postload(self):
        self.covmat = self.fullcov[np.ix_(self.used_indices, self.used_indices)]
        self.covinv = np.linalg.inv(self.covmat)
        self.data_vector = self.make_vector(self.data_arrays)
        self.errors = copy.deepcopy(self.data_arrays)
        cov_ix = 0
        for i, (type_ix, f1, f2, ix) in enumerate(self.indices):
            self.errors[type_ix][f1, f2][ix] = np.sqrt(self.fullcov[cov_ix, cov_ix])
            cov_ix += 1
        self.theta_bins_radians = self.theta_bins / 60 * np.pi / 180
        # Note hankel assumes integral starts at ell=0
        # (though could change spline to zero at zero).
        # At percent level it matters what is assumed
        if self.use_hankel:
            import hankel
            maxx = self.theta_bins_radians[-1] * self.l_max
            h = 3.2 * np.pi / maxx
            N = int(3.2 / h)
            self.hankel0 = hankel.HankelTransform(nu=0, N=N, h=h)
            self.hankel2 = hankel.HankelTransform(nu=2, N=N, h=h)
            self.hankel4 = hankel.HankelTransform(nu=4, N=N, h=h)
        elif self.binned_bessels:
            # Approximate bessel integral as binned smooth C_L against integrals of
            # bessel in each bin. Here we crudely precompute an approximation to the
            # bessel integral by brute force
            dls = np.diff(np.unique((np.exp(np.linspace(
                np.log(1.), np.log(self.l_max), int(500 * self.acc)))).astype(np.int)))
            groups = []
            ell = 2  # ell_min
            self.ls_bessel = np.zeros(dls.size)
            for i, dlx in enumerate(dls):
                self.ls_bessel[i] = (2 * ell + dlx - 1) / 2.
                groups.append(np.arange(ell, ell + dlx))
                ell += dlx
            js = np.empty((3, self.ls_bessel.size, len(self.theta_bins_radians)))
            bigell = np.arange(0, self.l_max + 1, dtype=np.float64)
            for i, theta in enumerate(self.theta_bins_radians):
                bigx = bigell * theta
                for ix, nu in enumerate([0, 2, 4]):
                    bigj = scipy.special.jn(nu, bigx) * bigell / (2 * np.pi)
                    for j, g in enumerate(groups):
                        js[ix, j, i] = np.sum(bigj[g])
            self.bessel_cache = js[0, :, :], js[1, :, :], js[2, :, :]
        else:
            # get ell for bessel transform in dense array,
            # and precompute bessel function matrices
            # Much slower than binned_bessels as many more sampling points
            dl = 4
            self.ls_bessel = np.arange(2 + dl / 2, self.l_max + 1, dl, dtype=np.float64)

            j0s = np.empty((len(self.ls_bessel), len(self.theta_bins_radians)))
            j2s = np.empty((len(self.ls_bessel), len(self.theta_bins_radians)))
            j4s = np.empty((len(self.ls_bessel), len(self.theta_bins_radians)))
            for i, theta in enumerate(self.theta_bins_radians):
                x = self.ls_bessel * theta
                j0s[:, i] = self.ls_bessel * scipy.special.jn(0, x)
                j2s[:, i] = self.ls_bessel * scipy.special.jn(2, x)
                j4s[:, i] = self.ls_bessel * scipy.special.jn(4, x)
            j0s *= dl / (2 * np.pi)
            j2s *= dl / (2 * np.pi)
            j4s *= dl / (2 * np.pi)
            self.bessel_cache = j0s, j2s, j4s
        # Fine z sampling
        if self.acc > 1:
            self.zs = np.linspace(0.005, self.zmax, int(350 * self.acc))
        else:
            self.zs = self.zmid[self.zmid <= self.zmax]
        # Interpolator z sampling
        assert self.zmax <= 5, "z max too large!"
        self.zs_interp = np.linspace(0, self.zmax, 100)

    def add_theory(self):
        if self.theory.__class__ == "classy":
            self.log.error(
                "DES likelihood not yet compatible with CLASS (help appreciated!)")
            raise HandledException
        # Requisites for the theory code
        self.theory.needs(**{
            "omegan": None,
            "omegam": None,
            "Pk_interpolator": {
                "z": self.zs_interp, "k_max": 15 * self.acc,
                "extrap_kmax": 500 * self.acc, "nonlinear": True,
                "vars_pairs": ([["delta_tot", "delta_tot"]] +
                               ([["Weyl", "Weyl"]] if self.use_Weyl else []))},
            "comoving_radial_distance": {"z": self.zs},
            "H": {"z": self.zs, "units": "km/s/Mpc"}})

    def get_theory(self, PKdelta, PKWeyl, bin_bias, shear_calibration_parameters,
                   intrinsic_alignment_A, intrinsic_alignment_alpha,
                   intrinsic_alignment_z0, wl_photoz_errors, lens_photoz_errors):
        h2 = (self.theory.get_param("H0") / 100) ** 2
        omegam = self.theory.get_param("omegam")
        chis = self.theory.get_comoving_radial_distance(self.zs)
        Hs = self.theory.get_H(self.zs, units="1/Mpc")
        dchis = np.hstack(
            ((chis[1] + chis[0]) / 2, (chis[2:] - chis[:-2]) / 2, (chis[-1] - chis[-2])))
        D_growth = PKdelta.P(self.zs, 0.001)
        D_growth = np.sqrt(D_growth / PKdelta.P(0, 0.001))
        c = _c_km_s * 1e3  # m/s

        def get_wq():
            Alignment_z = (intrinsic_alignment_A *
                           (((1 + self.zs) / (1 + intrinsic_alignment_z0)) **
                            intrinsic_alignment_alpha) *
                           0.0134 / D_growth)
            w = []
            for b in range(self.nzbins):
                w.append([])
                zshift = self.zs - wl_photoz_errors[b]
                n_chi = Hs * self.zbin_sp[b](zshift)
                n_chi[zshift < 0] = 0
                fac = n_chi * dchis
                for i, chi in enumerate(chis):
                    w[b].append(np.dot(fac[i:], (1 - chi / chis[i:])))
                w_align = (Alignment_z * n_chi /
                           (chis * (1 + self.zs) * 3 * h2 * (1e5 / c) ** 2 / 2))
                w[b] = np.asarray(w[b]) - w_align
            return w

        def get_qgal():
            w = []
            for b in range(self.nwbins):
                w.append([])
                zshift = self.zs - lens_photoz_errors[b]
                n_chi = Hs * self.zbin_w_sp[b](zshift)
                n_chi[zshift < 0] = 0
                w[b] = n_chi * bin_bias[b]
            return w

        if any([t in self.used_types for t in ["gammat", "wtheta"]]):
            qgal = get_qgal()
        if any([t in self.used_types for t in ["gammat", "xim", "xip"]]):
            wq = get_wq()
            if PKWeyl is not None:
                if 'gammat' in self.used_types:
                    self.log.error(
                        'DES currently only supports Weyl potential for lensing only')
                    raise HandledException
                qs = chis * wq
            else:
                qs = 3 * omegam * h2 * (1e5 / c) ** 2 * chis * (1 + self.zs) / 2 * wq
        ls_cl = np.array(np.hstack(
            (np.arange(2., 100 - 4 / self.acc, 4 / self.acc),
             np.exp(np.linspace(np.log(100.), np.log(self.l_max), int(50 * self.acc))))))
        # Get the angular power spectra and transform back
        weight = np.empty(chis.shape)
        dchifac = dchis / chis ** 2
        tmp = np.empty((ls_cl.shape[0], chis.shape[0]))
        for ix, l in enumerate(ls_cl):
            k = (l + 0.5) / chis
            weight[:] = dchifac
            weight[k < 1e-4] = 0
            weight[k >= PKdelta.kmax] = 0
            tmp[ix, :] = weight * PKdelta.P(self.zs, k, grid=False)
        if PKWeyl is not None:
            tmplens = np.empty((ls_cl.shape[0], chis.shape[0]))
            for ix, l in enumerate(ls_cl):
                k = (l + 0.5) / chis
                weight[:] = dchifac
                weight[k < 1e-4] = 0
                weight[k >= PKWeyl.kmax] = 0
                tmplens[ix, :] = weight * PKWeyl.P(self.zs, k, grid=False)
        else:
            tmplens = tmp
        corrs_th_p = np.empty((self.nzbins, self.nzbins), dtype=np.object)
        corrs_th_m = np.empty((self.nzbins, self.nzbins), dtype=np.object)
        corrs_th_w = np.empty((self.nwbins, self.nwbins), dtype=np.object)
        corrs_th_t = np.empty((self.nwbins, self.nzbins), dtype=np.object)
        if self.use_hankel:
            # Note that the absolute value of the correlation depends
            # on what you do about L_min (e.g. 1 vs 2 vs 0 makes a difference).
            if 'xip' in self.used_types or 'xim' in self.used_types:
                for (f1, f2) in self.bin_pairs[self.data_types.index('xip')]:
                    cl = UnivariateSpline(ls_cl, np.dot(tmplens, qs[f1] * qs[f2]), s=0)
                    fac = ((1 + shear_calibration_parameters[f1]) *
                           (1 + shear_calibration_parameters[f2]) / 2 / np.pi)
                    corrs_th_p[f1, f2] = self.hankel0.transform(
                        cl, self.theta_bins_radians, ret_err=False) * fac
                    corrs_th_m[f1, f2] = self.hankel4.transform(
                        cl, self.theta_bins_radians, ret_err=False) * fac
            if 'gammat' in self.used_types:
                for (f1, f2) in self.bin_pairs[self.data_types.index('gammat')]:
                    cl = UnivariateSpline(ls_cl, np.dot(tmp, qgal[f1] * qs[f2]), s=0)
                    fac = (1 + shear_calibration_parameters[f2]) / 2 / np.pi
                    corrs_th_t[f1, f2] = self.hankel2.transform(
                        cl, self.theta_bins_radians, ret_err=False) * fac
            if 'wtheta' in self.used_types:
                for (f1, f2) in self.bin_pairs[self.data_types.index('wtheta')]:
                    cl = UnivariateSpline(ls_cl, np.dot(tmp, qgal[f1] * qgal[f2]), s=0)
                    corrs_th_w[f1, f2] = self.hankel0.transform(
                        cl, self.theta_bins_radians, ret_err=False) / 2 / np.pi
        else:
            j0s, j2s, j4s = self.bessel_cache
            ls_bessel = self.ls_bessel
            if 'xip' in self.used_types or 'xim' in self.used_types:
                for (f1, f2) in self.bin_pairs[self.data_types.index('xip')]:
                    cl = UnivariateSpline(
                        ls_cl, np.dot(tmplens, qs[f1] * qs[f2]), s=0)(ls_bessel)
                    fac = ((1 + shear_calibration_parameters[f1]) *
                           (1 + shear_calibration_parameters[f2]))
                    corrs_th_p[f1, f2] = np.dot(cl, j0s) * fac
                    corrs_th_m[f1, f2] = np.dot(cl, j4s) * fac
            if 'gammat' in self.used_types:
                for (f1, f2) in self.bin_pairs[self.data_types.index('gammat')]:
                    cl = UnivariateSpline(
                        ls_cl, np.dot(tmp, qgal[f1] * qs[f2]), s=0)(ls_bessel)
                    corrs_th_t[f1, f2] = (np.dot(cl, j2s) *
                                          (1 + shear_calibration_parameters[f2]))
            if 'wtheta' in self.used_types:
                for (f1, f2) in self.bin_pairs[self.data_types.index('wtheta')]:
                    cl = UnivariateSpline(
                        ls_cl, np.dot(tmp, qgal[f1] * qgal[f2]), s=0)(ls_bessel)
                    corrs_th_w[f1, f2] = np.dot(cl, j0s)
        return [corrs_th_p, corrs_th_m, corrs_th_t, corrs_th_w]

    def make_vector(self, arrays):
        nused = len(self.used_items)
        data = np.empty(nused)
        for i, (type_ix, f1, f2, theta_ix) in enumerate(self.used_items):
            data[i] = arrays[type_ix][f1, f2][theta_ix]
        return data

    def make_thetas(self, arrays):
        nused = len(self.used_items)
        data = np.empty(nused)
        for i, (type_ix, f1, f2, theta_ix) in enumerate(self.used_items):
            data[i] = self.theta_bins[theta_ix]
        return data

    def chi_squared(self, theory, return_theory_vector=False):
        theory_vec = self.make_vector(theory)
        delta = self.data_vector - theory_vec
        chi2 = self.covinv.dot(delta).dot(delta)
        if return_theory_vector:
            return theory_vec, chi2
        else:
            return chi2

    def logp(self, **params_values):
        Pk_interpolators = self.theory.get_Pk_interpolator()
        PKdelta = Pk_interpolators["delta_tot_delta_tot"]
        PKWeyl = Pk_interpolators.get("Weyl_Weyl", None)
        wl_photoz_errors = [
            params_values.get(p, None) for p in
            ['DES_DzS1', 'DES_DzS2', 'DES_DzS3', 'DES_DzS4']]
        lens_photoz_errors = [
            params_values.get(p, None) for p in
            ['DES_DzL1', 'DES_DzL2', 'DES_DzL3', 'DES_DzL4', 'DES_DzL5']]
        bin_bias = [
            params_values.get(p, None) for p in
            ['DES_b1', 'DES_b2', 'DES_b3', 'DES_b4', 'DES_b5']]
        shear_calibration_parameters = [
            params_values.get(p, None) for p in
            ['DES_m1', 'DES_m2', 'DES_m3', 'DES_m4']]
        theory = self.get_theory(
            PKdelta, PKWeyl,
            bin_bias=bin_bias,
            wl_photoz_errors=wl_photoz_errors,
            lens_photoz_errors=lens_photoz_errors,
            shear_calibration_parameters=shear_calibration_parameters,
            intrinsic_alignment_A=params_values.get('DES_AIA'),
            intrinsic_alignment_alpha=params_values.get('DES_alphaIA'),
            intrinsic_alignment_z0=params_values.get('DES_z0IA'))
        return -0.5 * self.chi_squared(theory, return_theory_vector=False)

    # Plotting methods ###################################################################

    def plot_source_windows(self):
        import matplotlib.pyplot as plt
        for b in range(self.nzbins):
            plt.plot(self.zmid, self.zbin_sp[b](self.zmid))
        plt.xlim([0, 1.8])

    def plot_gal_windows(self):
        import matplotlib.pyplot as plt
        for b in range(self.nwbins):
            plt.plot(self.zmid, self.zbin_w_sp[b](self.zmid))
        plt.xlim([0, 1.2])

    def plot_w(self, corrs_w=None, errors=True, diff=False, axs=None, ls='-'):
        if "wtheta" not in self.used_types:
            self.log.warning("Clustering not computed. Nothing to plot.")
            return
        import matplotlib.pyplot as plt
        if axs is None:
            _, axs = plt.subplots(1, self.nwbins, figsize=(16, 3))
        for f1 in range(self.nwbins):
            ax = axs[f1]
            data = self.data_arrays[3][f1, f1]
            if diff:
                if errors:
                    data = (data - corrs_w[f1, f1]) / self.errors[3][f1, f1]
                    fac = 1
                else:
                    data = data / corrs_w[f1, f1] - 1
                    fac = 1 / corrs_w[f1, f1]
            else:
                fac = self.theta_bins
                data = data * fac
            if errors and not diff:
                ax.errorbar(self.theta_bins, data, fac * self.errors[3][f1, f1], ls=ls)
            else:
                ax.semilogx(self.theta_bins, data, ls=ls)
            if corrs_w is not None and not diff:
                ax.semilogx(self.theta_bins, fac * corrs_w[f1, f1], ls=ls)
            ax.axvspan(*self.ranges['wtheta'][f1][f1], color='gray', alpha=0.1)
            ax.set_title(f1 + 1)
        return axs

    def plot_lensing(self, corrs_p=None, corrs_m=None, errors=True,
                     diff=False, axs=None, ls='-'):
        if any([t not in self.used_types for t in ["xip", "xim"]]):
            self.log.warning("Shear not computed. Nothing to plot.")
            return
        import matplotlib.pyplot as plt
        if axs is None:
            _, axs = plt.subplots(self.nzbins, self.nzbins, figsize=(14, 14))
        for f1, f2 in self.bin_pairs[0]:
            ax = axs[f1, f2]
            ax.axvspan(*self.ranges['xip'][f1][f2], color='gray', alpha=0.1)
            xip = self.data_arrays[0][f1, f2]
            xim = self.data_arrays[1][f1, f2]
            if diff:
                if errors:
                    xip = (xip - corrs_p[f1, f2]) / self.errors[0][f1, f2]
                    xim = (xim - corrs_m[f1, f2]) / self.errors[1][f1, f2]
                    fac = 1
                    facm = 1
                else:
                    xip = xip / corrs_p[f1, f2] - 1
                    xim = xim / corrs_m[f1, f2] - 1
                    fac = 1 / corrs_p[f1, f2]
                    facm = 1 / corrs_m[f1, f2]
            else:
                fac = 1e4 * self.theta_bins
                facm = fac
                xip = xip * fac
                xim = xim * fac
            if errors and not diff:
                ax.errorbar(self.theta_bins, xip, fac * self.errors[0][f1, f2],
                            color='C0', ls=ls)
                ax.errorbar(self.theta_bins, xim, facm * self.errors[1][f1, f2],
                            color='C1', ls=ls)
            else:
                ax.semilogx(self.theta_bins, xip, color='C0', ls=ls)
                ax.semilogx(self.theta_bins, xim, color='C1', ls=ls)
            if corrs_p is not None and not diff:
                ax.semilogx(self.theta_bins, fac * corrs_p[f1, f2], color='C0', ls='--')
                ax.semilogx(self.theta_bins, facm * corrs_m[f1, f2], color='C1', ls='--')
            ax.set_title('%s-%s' % (f1 + 1, f2 + 1))
        return axs

    def plot_cross(self, corrs_t=None, errors=True, diff=False, axs=None, ls='-'):
        if "gammat" not in self.used_types:
            self.log.warning("Galaxy x galaxy-lensing not computed. Nothing to plot.")
            return
        import matplotlib.pyplot as plt
        if axs is None:
            _, axs = plt.subplots(self.nzbins, self.nwbins, figsize=(16, 14))
        for f1 in range(self.nzbins):
            for f2 in range(self.nwbins):
                ax = axs[f1, f2]
                data = self.data_arrays[2][f2, f1]
                if diff:
                    if errors:
                        data = (data - corrs_t[f2, f1]) / self.errors[2][f2, f1]
                        fac = 1
                    else:
                        data = data / corrs_t[f2, f1] - 1
                        fac = 1. / corrs_t[f2, f1]
                else:
                    fac = 100 * self.theta_bins
                    data = data * fac
                if errors and not diff:
                    ax.errorbar(self.theta_bins, data, fac * self.errors[2][f2, f1], ls=ls)
                else:
                    ax.semilogx(self.theta_bins, data, ls=ls)
                if corrs_t is not None and not diff:
                    ax.semilogx(self.theta_bins, fac * corrs_t[f2, f1], ls=ls)
                ax.axvspan(*self.ranges['gammat'][f2][f1], color='gray', alpha=0.1)
                ax.set_title('%s-%s' % (f2 + 1, f1 + 1))
        return axs


# Conversion .fits --> .dataset  #########################################################

def convert_txt(filename, root, outdir, ranges=None):
    import astropy.io.fits as fits
    if ranges is None:
        ranges = get_def_cuts()
    hdulist = fits.open(filename)
    outlines = []
    outlines += ['measurements_format = DES']
    outlines += ['kmax = 10']  # maches what DES used and is good enough for likelihood
    outlines += ['intrinsic_alignment_model = DES1YR']
    outlines += ['data_types = xip xim gammat wtheta']
    outlines += ['used_data_types = xip xim gammat wtheta']
    outlines += ['num_z_bins = %s' % (max(hdulist['xip'].data['BIN1']))]
    outlines += ['num_gal_bins = %s' % (max(hdulist['wtheta'].data['BIN1']))]
    ntheta = max(hdulist['wtheta'].data['ANGBIN']) + 1
    outlines += ['num_theta_bins = %s' % (ntheta)]
    thetas = hdulist['xip'].data['ANG'][:ntheta]
    np.savetxt(outdir + root + '_theta_bins.dat', thetas, header='theta_arcmin')
    outlines += ['theta_bins_file = %s' % (root + '_theta_bins.dat')]
    np.savetxt(outdir + root + '_cov.dat', hdulist['COVMAT'].data, fmt='%.4e')
    outlines += ['cov_file = %s' % (root + '_cov.dat')]
    out_ranges = []
    for i, tp in enumerate(def_DES_types):
        pairs = []
        for b1, b2 in zip(hdulist[tp].data['BIN1'], hdulist[tp].data['BIN2']):
            if not (b1, b2) in pairs:
                pairs.append((b1, b2))
        for x, y in pairs:
            out_ranges += ['%s %s %s %s %s' % (
                tp, x, y, ranges[tp][x - 1][y - 1][0], ranges[tp][x - 1][y - 1][1])]
        # drop theta value, as assuming shared to all data
        dat = np.asarray(zip(*[hdulist[tp].data[n]
                               for n in list(hdulist[tp].data.names)[:-2]]))
        # fix anomaly that z bins are 1 based but theta bins zero based
        dat[:, 2] += 1
        np.savetxt(outdir + root + '_%s.dat' % tp, dat, fmt=['%u', '%u', '%u', '%.8e'],
                   header=" ".join(list(hdulist[tp].data.dtype.names)[:-2]))
        outlines += ['measurements[%s] = %s_%s.dat' % (tp, root, tp)]
    sourcedata = hdulist['NZ_SOURCE'].data
    maxi = sourcedata.shape[0] - 1
    while np.all(np.asarray(sourcedata[maxi][3:]) == 0):
        maxi -= 1
    np.savetxt(outdir + root + '_nz_source.dat', sourcedata[:maxi + 1], fmt='%.6e',
               header=" ".join(hdulist['NZ_SOURCE'].data.dtype.names))
    outlines += ['nz_file = %s_nz_source.dat' % (root)]
    assert (np.all(np.asarray(hdulist['NZ_LENS'].data[maxi + 1][3:]) == 0))
    np.savetxt(outdir + root + '_nz_lens.dat', hdulist['NZ_LENS'].data[:maxi + 1],
               fmt='%.6e', header=" ".join(hdulist['NZ_LENS'].data.dtype.names))
    outlines += ['nz_gal_file = %s_nz_lens.dat' % (root)]
    with open(outdir + root + '_selection.dat', 'w') as f:
        f.write('#  type bin1 bin2 theta_min theta_max\n')
        f.write("\n".join(out_ranges))
    outlines += ['data_selection = %s_selection.dat' % (root)]
    outlines += ['nuisance_params = DES.paramnames']
    with open(outdir + root + '.dataset', 'w') as f:
        f.write("\n".join(outlines))


# Installation routines ##################################################################

# name of the data and covmats repo/folder
des_data_name = "des_data"
des_data_version = "v1.0"


def get_path(path):
    return os.path.realpath(os.path.join(path, "data", des_data_name))


def is_installed(**kwargs):
    return os.path.exists(os.path.realpath(
        os.path.join(kwargs["path"], "data", des_data_name)))


def install(path=None, force=False, code=False, data=True, no_progress_bars=False):
    if not data:
        return True
    log = logging.getLogger(__name__.split(".")[-1])
    log.info("Downloading DES data...")
    return download_github_release(os.path.join(path, "data"), des_data_name,
                                   des_data_version, no_progress_bars=no_progress_bars)
