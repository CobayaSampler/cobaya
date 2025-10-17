"""
.. module:: des

:Synopsis: DES likelihood, independent Python implementation.
           Well tested and agrees with likelihoods in DES chains for fixed nu mass.
:Author: Antony Lewis (little changes for Cobaya by Jesus Torrado)

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

- ``des_y1.clustering``
- ``des_y1.shear``
- ``des_y1.galaxy_galaxy``
- ``des_y1.joint`` (a shortcut for the combination of the previous three)

Usage
-----

To use any of the DES likelihoods, you simply need to mention them in the
``likelihood`` block, or add them using the :doc:`input generator <cosmo_basic_runs>`.

The corresponding nuisance parameters will be added automatically,
so you don't have to care about listing them in the ``params`` block.

The nuisance parameters and their default priors can be obtained as explained in
:ref:`citations`.


Installation
------------

This likelihood can be installed automatically as explained in :doc:`installation_cosmo`.

"""

import copy

import numpy as np
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline

from cobaya.conventions import Const
from cobaya.functions import numba
from cobaya.likelihoods.base_classes.DataSetLikelihood import DataSetLikelihood
from cobaya.log import LoggedError

# DES data types
def_DES_types = ["xip", "xim", "gammat", "wtheta"]

_spline = InterpolatedUnivariateSpline


def get_def_cuts():  # pragma: no cover
    ranges = {}
    for tp in def_DES_types:
        ranges[tp] = np.empty((6, 6), dtype=object)
    ranges["xip"][1][1] = [7.195005, 250.0]
    ranges["xip"][1][2] = [7.195005, 250.0]
    ranges["xip"][1][3] = [5.715196, 250.0]
    ranges["xip"][1][4] = [5.715196, 250.0]
    ranges["xip"][2][1] = [7.195005, 250.0]
    ranges["xip"][2][2] = [4.539741, 250.0]
    ranges["xip"][2][3] = [4.539741, 250.0]
    ranges["xip"][2][4] = [4.539741, 250.0]
    ranges["xip"][3][1] = [5.715196, 250.0]
    ranges["xip"][3][2] = [4.539741, 250.0]
    ranges["xip"][3][3] = [3.606045, 250.0]
    ranges["xip"][3][4] = [3.606045, 250.0]
    ranges["xip"][4][1] = [5.715196, 250.0]
    ranges["xip"][4][2] = [4.539741, 250.0]
    ranges["xip"][4][3] = [3.606045, 250.0]
    ranges["xip"][4][4] = [3.606045, 250.0]
    ranges["xim"][1][1] = [90.579750, 250.0]
    ranges["xim"][1][2] = [71.950053, 250.0]
    ranges["xim"][1][3] = [71.950053, 250.0]
    ranges["xim"][1][4] = [71.950053, 250.0]
    ranges["xim"][2][1] = [71.950053, 250.0]
    ranges["xim"][2][2] = [57.151958, 250.0]
    ranges["xim"][2][3] = [57.151958, 250.0]
    ranges["xim"][2][4] = [45.397414, 250.0]
    ranges["xim"][3][1] = [71.950053, 250.0]
    ranges["xim"][3][2] = [57.151958, 250.0]
    ranges["xim"][3][3] = [45.397414, 250.0]
    ranges["xim"][3][4] = [45.397414, 250.0]
    ranges["xim"][4][1] = [71.950053, 250.0]
    ranges["xim"][4][2] = [45.397414, 250.0]
    ranges["xim"][4][3] = [45.397414, 250.0]
    ranges["xim"][4][4] = [36.060448, 250.0]
    ranges["gammat"][1][1] = [64.0, 250.0]
    ranges["gammat"][1][2] = [64.0, 250.0]
    ranges["gammat"][1][3] = [64.0, 250.0]
    ranges["gammat"][1][4] = [64.0, 250.0]
    ranges["gammat"][2][1] = [40.0, 250.0]
    ranges["gammat"][2][2] = [40.0, 250.0]
    ranges["gammat"][2][3] = [40.0, 250.0]
    ranges["gammat"][2][4] = [40.0, 250.0]
    ranges["gammat"][3][1] = [30.0, 250.0]
    ranges["gammat"][3][2] = [30.0, 250.0]
    ranges["gammat"][3][3] = [30.0, 250.0]
    ranges["gammat"][3][4] = [30.0, 250.0]
    ranges["gammat"][4][1] = [24.0, 250.0]
    ranges["gammat"][4][2] = [24.0, 250.0]
    ranges["gammat"][4][3] = [24.0, 250.0]
    ranges["gammat"][4][4] = [24.0, 250.0]
    ranges["gammat"][5][1] = [21.0, 250.0]
    ranges["gammat"][5][2] = [21.0, 250.0]
    ranges["gammat"][5][3] = [21.0, 250.0]
    ranges["gammat"][5][4] = [21.0, 250.0]
    ranges["wtheta"][1][1] = [43.0, 250.0]
    ranges["wtheta"][2][2] = [27.0, 250.0]
    ranges["wtheta"][3][3] = [20.0, 250.0]
    ranges["wtheta"][4][4] = [16.0, 250.0]
    ranges["wtheta"][5][5] = [14.0, 250.0]
    for tp in def_DES_types:
        ranges[tp] = ranges[tp][1:, 1:]
    return ranges


if numba:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        @numba.njit("void(float64[::1],float64[::1],float64[::1],float64[::1])")
        def _get_lensing_dots(wq_b, chis, n_chi, dchis):
            for i, chi in enumerate(chis):
                wq_b[i] = np.dot(n_chi[i:], (1 - chi / chis[i:]) * dchis[i:])

        @numba.njit("void(float64[:,::1],float64[:,::1],float64[::1],float64)")
        def _limber_PK_terms(powers, ks, dchifac, kmax):
            for ix in range(powers.shape[0]):
                for i in range(powers.shape[1]):
                    if 1e-4 <= ks[ix, i] < kmax:
                        powers[ix, i] *= dchifac[i]
                    else:
                        powers[ix, i] = 0


class DES(DataSetLikelihood):
    install_options = {
        "github_repository": "CobayaSampler/des_data",
        "github_release": "v1.0",
    }

    # variables defined in yaml
    acc: float
    binned_bessels: bool
    use_hankel: bool
    use_Weyl: bool
    l_max: int

    def load_dataset_file(self, filename, dataset_params=None):
        self.l_max = self.l_max or int(50000 * self.acc)
        # lmax here is an internal parameter for transforms
        if filename.endswith(".fits"):
            if dataset_params:
                raise LoggedError(
                    self.log,
                    "'dataset_params' can only be specified "
                    "for .dataset (not .fits) file.",
                )
            try:
                self.load_fits_data(filename)
            except OSError:
                raise LoggedError(
                    self.log,
                    "The data file '%s' could not be found'. Check your paths!",
                    filename,
                )

        else:
            super().load_dataset_file(filename, dataset_params)
        self.initialize_postload()

    def init_params(self, ini):
        self.indices = []
        self.used_indices = []
        self.used_items = []
        self.fullcov = np.loadtxt(ini.relativeFileName("cov_file"))
        ntheta = ini.int("num_theta_bins")
        self.theta_bins = np.loadtxt(ini.relativeFileName("theta_bins_file"))
        self.iintrinsic_alignment_model = ini.string("intrinsic_alignment_model")
        self.data_types = ini.string("data_types").split()
        self.used_types = ini.list("used_data_types", self.data_types)
        with open(ini.relativeFileName("data_selection"), encoding="utf-8") as f:
            header = f.readline()
            assert "#  type bin1 bin2 theta_min theta_max" == header.strip()
            lines = f.readlines()
        ranges = {}
        for tp in self.data_types:
            ranges[tp] = np.empty((6, 6), dtype=object)
        for line in lines:
            items = line.split()
            if items[0] in self.used_types:
                bin1, bin2 = (int(x) - 1 for x in items[1:3])
                ranges[items[0]][bin1][bin2] = [np.float64(x) for x in items[3:]]
        self.ranges = ranges
        self.nzbins = ini.int("num_z_bins")  # for lensing sources
        self.nwbins = ini.int("num_gal_bins", 0)  # for galaxies
        maxbin = max(self.nzbins, self.nwbins)
        cov_ix = 0
        self.bin_pairs: list[list[tuple]] = []
        self.data_arrays = []
        self.thetas = []
        for i, tp in enumerate(self.data_types):
            xi = np.loadtxt(ini.relativeFileName("measurements[%s]" % tp))
            bin1 = xi[:, 0].astype(int) - 1
            bin2 = xi[:, 1].astype(int) - 1
            tbin = xi[:, 2].astype(int) - 1
            corr = np.empty((maxbin, maxbin), dtype=object)
            corr[:, :] = None
            self.data_arrays.append(corr)
            self.bin_pairs.append([])
            for f1, f2, ix, dat in zip(bin1, bin2, tbin, xi[:, 3]):
                self.indices.append((i, f1, f2, ix))
                if (f1, f2) not in self.bin_pairs[i]:
                    self.bin_pairs[i].append((f1, f2))
                    corr[f1, f2] = np.zeros(ntheta)
                corr[f1, f2][ix] = dat
                if ranges[tp][f1, f2] is not None:
                    mn, mx = ranges[tp][f1, f2]
                    if mn < self.theta_bins[ix] < mx:
                        self.thetas.append(self.theta_bins[ix])
                        self.used_indices.append(cov_ix)
                        self.used_items.append(self.indices[-1])
                cov_ix += 1
        nz_source = np.loadtxt(ini.relativeFileName("nz_file"))
        self.zmid = nz_source[:, 1]
        self.zbin_sp = []
        for b in range(self.nzbins):
            self.zbin_sp += [InterpolatedUnivariateSpline(self.zmid, nz_source[:, b + 3])]
        nz_lens = np.loadtxt(ini.relativeFileName("nz_gal_file"))
        assert np.array_equal(nz_lens[:, 1], self.zmid)
        self.zbin_w_sp = []
        for b in range(self.nwbins):
            self.zbin_w_sp += [InterpolatedUnivariateSpline(self.zmid, nz_lens[:, b + 3])]
        self.zmax = self.zmid[-1]
        # k_max actually computed, assumes extrapolated beyond that
        self.k_max = ini.float("kmax", 15)

    def load_fits_data(self, filename, ranges=None):  # pragma: no cover
        import astropy.io.fits as fits  # type: ignore

        if ranges is None:
            ranges = get_def_cuts()
        hdulist = fits.open(filename)
        self.indices = []
        self.intrinsic_alignment_model = "DES1YR"
        self.used_indices = []
        self.used_items = []
        self.ranges = ranges
        self.fullcov = hdulist["COVMAT"].data
        cov_ix = 0
        self.bin_pairs: list[list[tuple]] = []
        self.data_types = def_DES_types
        self.used_types = def_DES_types
        for i, tp in enumerate(def_DES_types):
            xi = hdulist[tp].data
            self.bin_pairs.append([])
            for f1, f2, ix, dat, theta in zip(
                xi.field(0) - 1, xi.field(1) - 1, xi.field(2), xi.field(3), xi.field(4)
            ):
                self.indices.append((i, f1, f2, ix))
                if (f1, f2) not in self.bin_pairs[i]:
                    self.bin_pairs[i].append((f1, f2))
                mn, mx = ranges[tp][f1, f2]
                if mn < theta < mx:
                    self.used_indices.append(cov_ix)
                    self.used_items.append(self.indices[-1])
                cov_ix += 1
        self.nzbins = 4  # for lensing sources
        corrs_p = np.empty((self.nzbins, self.nzbins), dtype=object)
        corrs_p[:, :] = None
        corrs_m = np.empty((self.nzbins, self.nzbins), dtype=object)
        corrs_m[:, :] = None
        ntheta = 20
        self.theta_bins = np.empty(ntheta)
        for xi, corrs in zip(
            [hdulist["xip"].data, hdulist["xim"].data], [corrs_p, corrs_m]
        ):
            for f1, f2, ix, dat, theta in zip(
                xi.field(0) - 1, xi.field(1) - 1, xi.field(2), xi.field(3), xi.field(4)
            ):
                if corrs[f1, f2] is None:
                    corrs[f1, f2] = np.zeros(ntheta)
                if f1 == 0 and f2 == 0:
                    self.theta_bins[ix] = theta
                assert np.abs(self.theta_bins[ix] / theta - 1) < 2e-3
                corrs[f1, f2][ix] = dat
        self.nwbins = 5  # for galaxies
        corrs_w = np.empty((self.nwbins, self.nwbins), dtype=object)
        for f1 in range(self.nwbins):
            corrs_w[f1, f1] = np.empty(ntheta)
        wdata = hdulist["wtheta"].data
        for f1, f2, ix, dat, theta in zip(
            wdata.field(0) - 1,
            wdata.field(1) - 1,
            wdata.field(2),
            wdata.field(3),
            wdata.field(4),
        ):
            assert f1 == f2
            assert np.abs(self.theta_bins[ix] / theta - 1) < 2e-3
            corrs_w[f1, f2][ix] = dat
        corrs_t = np.empty((self.nwbins, self.nwbins), dtype=object)
        corrs_t[:, :] = None
        tdata = hdulist["gammat"].data
        for f1, f2, ix, dat, theta in zip(
            tdata.field(0) - 1,
            tdata.field(1) - 1,
            tdata.field(2),
            tdata.field(3),
            tdata.field(4),
        ):
            assert np.abs(self.theta_bins[ix] / theta - 1) < 2e-3
            if corrs_t[f1, f2] is None:
                corrs_t[f1, f2] = np.empty(ntheta)
            corrs_t[f1, f2][ix] = dat
        self.data_arrays = [corrs_p, corrs_m, corrs_t, corrs_w]
        self.zmid = hdulist["NZ_SOURCE"].data.field("Z_MID")
        self.zbin_sp = []
        for b in range(self.nzbins):
            self.zbin_sp += [_spline(self.zmid, hdulist["NZ_SOURCE"].data.field(b + 3))]
        zmid_w = hdulist["NZ_LENS"].data.field("Z_MID")
        assert np.array_equal(zmid_w, self.zmid)
        self.zbin_w_sp = []
        for b in range(self.nwbins):
            self.zbin_w_sp += [_spline(self.zmid, hdulist["NZ_LENS"].data.field(b + 3))]
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
        if self.use_hankel:  # pragma: no cover
            import hankel  # type: ignore

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
            dls = np.diff(
                np.unique(
                    (
                        np.exp(
                            np.linspace(
                                np.log(1.0), np.log(self.l_max), int(500 * self.acc)
                            )
                        )
                    ).astype(int)
                )
            )
            groups = []
            ell = 2  # ell_min
            self.ls_bessel = np.zeros(dls.size)
            for i, dlx in enumerate(dls):
                self.ls_bessel[i] = (2 * ell + dlx - 1) / 2.0
                groups.append(np.arange(ell, ell + dlx))
                ell += dlx
            js = np.empty((3, self.ls_bessel.size, len(self.theta_bins_radians)))
            bigell = np.arange(0, self.l_max + 1, dtype=np.float64)
            for i, theta in enumerate(self.theta_bins_radians):
                bigx = bigell * theta
                for ix, nu in enumerate([0, 2, 4]):
                    bigj = special.jn(nu, bigx) * bigell / (2 * np.pi)
                    for j, g in enumerate(groups):
                        js[ix, j, i] = np.sum(bigj[g])
            self.bessel_cache = js[0, :, :], js[1, :, :], js[2, :, :]
        else:  # pragma: no cover
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
                j0s[:, i] = self.ls_bessel * special.jn(0, x)
                j2s[:, i] = self.ls_bessel * special.jn(2, x)
                j4s[:, i] = self.ls_bessel * special.jn(4, x)
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

    def get_requirements(self):
        return {
            "H0": None,
            "omegam": None,
            "Pk_interpolator": {
                "z": self.zs_interp,
                "k_max": 15 * self.acc,
                "nonlinear": True,
                "vars_pairs": (
                    [("delta_tot", "delta_tot")]
                    + ([("Weyl", "Weyl")] if self.use_Weyl else [])
                ),
            },
            "comoving_radial_distance": {"z": self.zs},
            "Hubble": {"z": self.zs},
        }

    def get_theory(
        self,
        PKdelta,
        PKWeyl,
        bin_bias,
        shear_calibration_parameters,
        intrinsic_alignment_A,
        intrinsic_alignment_alpha,
        intrinsic_alignment_z0,
        wl_photoz_errors,
        lens_photoz_errors,
    ):
        h2 = (self.provider.get_param("H0") / 100) ** 2
        omegam = self.provider.get_param("omegam")
        chis = self.provider.get_comoving_radial_distance(self.zs)
        Hs = self.provider.get_Hubble(self.zs, units="1/Mpc")
        dchis = np.hstack(
            ((chis[1] + chis[0]) / 2, (chis[2:] - chis[:-2]) / 2, (chis[-1] - chis[-2]))
        )
        D_growth = PKdelta.P(self.zs, 0.001)
        D_growth = np.sqrt(D_growth / PKdelta.P(0, 0.001))
        c = Const.c_km_s * 1e3  # m/s

        if any(t in self.used_types for t in ["gammat", "wtheta"]):
            qgal = []
            for b in range(self.nwbins):
                qgal.append([])
                zshift = self.zs - lens_photoz_errors[b]
                n_chi = Hs * self.zbin_w_sp[b](zshift)
                n_chi[zshift < 0] = 0
                qgal[b] = n_chi * bin_bias[b]
        if any(t in self.used_types for t in ["gammat", "xim", "xip"]):
            Alignment_z = (
                intrinsic_alignment_A
                * (
                    ((1 + self.zs) / (1 + intrinsic_alignment_z0))
                    ** intrinsic_alignment_alpha
                )
                * 0.0134
                / D_growth
            )
            Alignment_z /= chis * (1 + self.zs) * 3 * h2 * (1e5 / c) ** 2 / 2

            wq = np.empty((self.nzbins, len(chis)))
            wq_b = np.empty(chis.shape)

            for b in range(self.nzbins):
                zshift = self.zs - wl_photoz_errors[b]
                n_chi = Hs * self.zbin_sp[b](zshift)
                n_chi[zshift < 0] = 0
                if numba:
                    _get_lensing_dots(wq_b, chis, n_chi, dchis)
                else:
                    for i, chi in enumerate(chis):
                        wq_b[i] = np.dot(n_chi[i:], (1 - chi / chis[i:]) * dchis[i:])
                wq[b] = wq_b - Alignment_z * n_chi

            if PKWeyl is not None:
                if "gammat" in self.used_types:
                    raise LoggedError(
                        self.log,
                        "DES currently only supports Weyl potential for lensing only",
                    )
                qs = chis * wq
            else:
                qs = 3 * omegam * h2 * (1e5 / c) ** 2 * chis * (1 + self.zs) / 2 * wq
        ls_cl = np.hstack(
            (
                np.arange(2.0, 100 - 4 / self.acc, 4 / self.acc),
                np.exp(
                    np.linspace(np.log(100.0), np.log(self.l_max), int(50 * self.acc))
                ),
            )
        )
        # Get the angular power spectra and transform back
        dchifac = dchis / chis**2
        if numba:
            ks = np.outer(ls_cl + 0.5, 1 / chis)
            tmp = PKdelta.P(self.zs, ks, grid=False)
            _limber_PK_terms(tmp, ks, dchifac, PKdelta.kmax)
        else:
            tmp = np.empty((ls_cl.shape[0], chis.shape[0]))
            weight = np.empty(chis.shape)
            for ix, ell in enumerate(ls_cl):
                k = (ell + 0.5) / chis
                weight[:] = dchifac
                weight[k < 1e-4] = 0
                weight[k >= PKdelta.kmax] = 0
                tmp[ix, :] = weight * PKdelta.P(self.zs, k, grid=False)

        if PKWeyl is not None:
            if numba:
                tmplens = PKWeyl.P(self.zs, ks, grid=False)
                _limber_PK_terms(tmplens, ks, dchifac, PKWeyl.kmax)
            else:
                tmplens = np.empty((ls_cl.shape[0], chis.shape[0]))
                for ix, ell in enumerate(ls_cl):
                    k = (ell + 0.5) / chis
                    weight[:] = dchifac
                    weight[k < 1e-4] = 0
                    weight[k >= PKWeyl.kmax] = 0
                    tmplens[ix, :] = weight * PKWeyl.P(self.zs, k, grid=False)
        else:
            tmplens = tmp
        corrs_th_p = np.empty((self.nzbins, self.nzbins), dtype=object)
        corrs_th_m = np.empty((self.nzbins, self.nzbins), dtype=object)
        corrs_th_w = np.empty((self.nwbins, self.nwbins), dtype=object)
        corrs_th_t = np.empty((self.nwbins, self.nzbins), dtype=object)
        if self.use_hankel:  # pragma: no cover
            # Note that the absolute value of the correlation depends
            # on what you do about L_min (e.g. 1 vs 2 vs 0 makes a difference).
            if "xip" in self.used_types or "xim" in self.used_types:
                for f1, f2 in self.bin_pairs[self.data_types.index("xip")]:
                    cl = _spline(ls_cl, np.dot(tmplens, qs[f1] * qs[f2]))
                    fac = (
                        (1 + shear_calibration_parameters[f1])
                        * (1 + shear_calibration_parameters[f2])
                        / 2
                        / np.pi
                    )
                    corrs_th_p[f1, f2] = (
                        self.hankel0.transform(cl, self.theta_bins_radians, ret_err=False)
                        * fac
                    )
                    corrs_th_m[f1, f2] = (
                        self.hankel4.transform(cl, self.theta_bins_radians, ret_err=False)
                        * fac
                    )
            if "gammat" in self.used_types:
                for f1, f2 in self.bin_pairs[self.data_types.index("gammat")]:
                    cl = _spline(ls_cl, np.dot(tmp, qgal[f1] * qs[f2]))
                    fac = (1 + shear_calibration_parameters[f2]) / 2 / np.pi
                    corrs_th_t[f1, f2] = (
                        self.hankel2.transform(cl, self.theta_bins_radians, ret_err=False)
                        * fac
                    )
            if "wtheta" in self.used_types:
                for f1, f2 in self.bin_pairs[self.data_types.index("wtheta")]:
                    cl = _spline(ls_cl, np.dot(tmp, qgal[f1] * qgal[f2]))
                    corrs_th_w[f1, f2] = (
                        self.hankel0.transform(cl, self.theta_bins_radians, ret_err=False)
                        / 2
                        / np.pi
                    )
        else:
            j0s, j2s, j4s = self.bessel_cache
            ls_bessel = self.ls_bessel
            if "xip" in self.used_types or "xim" in self.used_types:
                for f1, f2 in self.bin_pairs[self.data_types.index("xip")]:
                    cl = _spline(ls_cl, np.dot(tmplens, qs[f1] * qs[f2]))(ls_bessel)
                    fac = (1 + shear_calibration_parameters[f1]) * (
                        1 + shear_calibration_parameters[f2]
                    )
                    corrs_th_p[f1, f2] = np.dot(cl, j0s) * fac
                    corrs_th_m[f1, f2] = np.dot(cl, j4s) * fac
            if "gammat" in self.used_types:
                for f1, f2 in self.bin_pairs[self.data_types.index("gammat")]:
                    cl = _spline(ls_cl, np.dot(tmp, qgal[f1] * qs[f2]))(ls_bessel)
                    corrs_th_t[f1, f2] = np.dot(cl, j2s) * (
                        1 + shear_calibration_parameters[f2]
                    )
            if "wtheta" in self.used_types:
                for f1, f2 in self.bin_pairs[self.data_types.index("wtheta")]:
                    cl = _spline(ls_cl, np.dot(tmp, qgal[f1] * qgal[f2]))(ls_bessel)
                    corrs_th_w[f1, f2] = np.dot(cl, j0s)
        return [corrs_th_p, corrs_th_m, corrs_th_t, corrs_th_w]

    def make_vector(self, arrays):
        nused = len(self.used_items)
        data = np.empty(nused)
        for i, (type_ix, f1, f2, theta_ix) in enumerate(self.used_items):
            data[i] = arrays[type_ix][f1, f2][theta_ix]
        return data

    def make_thetas(self):
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
        PKdelta = self.provider.get_Pk_interpolator(
            ("delta_tot", "delta_tot"), extrap_kmax=3000 * self.acc
        )
        if self.use_Weyl:
            PKWeyl = self.provider.get_Pk_interpolator(
                ("Weyl", "Weyl"), extrap_kmax=3000 * self.acc
            )
        else:
            PKWeyl = None

        wl_photoz_errors = [
            params_values.get(p, None)
            for p in ["DES_DzS1", "DES_DzS2", "DES_DzS3", "DES_DzS4"]
        ]
        lens_photoz_errors = [
            params_values.get(p, None)
            for p in ["DES_DzL1", "DES_DzL2", "DES_DzL3", "DES_DzL4", "DES_DzL5"]
        ]
        bin_bias = [
            params_values.get(p, None)
            for p in ["DES_b1", "DES_b2", "DES_b3", "DES_b4", "DES_b5"]
        ]
        shear_calibration_parameters = [
            params_values.get(p, None) for p in ["DES_m1", "DES_m2", "DES_m3", "DES_m4"]
        ]
        theory = self.get_theory(
            PKdelta,
            PKWeyl,
            bin_bias=bin_bias,
            wl_photoz_errors=wl_photoz_errors,
            lens_photoz_errors=lens_photoz_errors,
            shear_calibration_parameters=shear_calibration_parameters,
            intrinsic_alignment_A=params_values.get("DES_AIA"),
            intrinsic_alignment_alpha=params_values.get("DES_alphaIA"),
            intrinsic_alignment_z0=params_values.get("DES_z0IA"),
        )
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

    def plot_w(self, corrs_w=None, errors=True, diff=False, axs=None, ls="-"):
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
            ax.axvspan(*self.ranges["wtheta"][f1][f1], color="gray", alpha=0.1)
            ax.set_title(f1 + 1)
        return axs

    def plot_lensing(
        self, corrs_p=None, corrs_m=None, errors=True, diff=False, axs=None, ls="-"
    ):  # pragma: no cover
        if any(t not in self.used_types for t in ["xip", "xim"]):
            self.log.warning("Shear not computed. Nothing to plot.")
            return
        import matplotlib.pyplot as plt

        if axs is None:
            _, axs = plt.subplots(self.nzbins, self.nzbins, figsize=(14, 14))
        for f1, f2 in self.bin_pairs[0]:
            ax = axs[f1, f2]
            ax.axvspan(*self.ranges["xip"][f1][f2], color="gray", alpha=0.1)
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
                ax.errorbar(
                    self.theta_bins, xip, fac * self.errors[0][f1, f2], color="C0", ls=ls
                )
                ax.errorbar(
                    self.theta_bins, xim, facm * self.errors[1][f1, f2], color="C1", ls=ls
                )
            else:
                ax.semilogx(self.theta_bins, xip, color="C0", ls=ls)
                ax.semilogx(self.theta_bins, xim, color="C1", ls=ls)
            if corrs_p is not None and not diff:
                ax.semilogx(self.theta_bins, fac * corrs_p[f1, f2], color="C0", ls="--")
                ax.semilogx(self.theta_bins, facm * corrs_m[f1, f2], color="C1", ls="--")
            ax.set_title(f"{f1 + 1}-{f2 + 1}")
        return axs

    def plot_cross(self, corrs_t=None, errors=True, diff=False, axs=None, ls="-"):
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
                        fac = 1.0 / corrs_t[f2, f1]
                else:
                    fac = 100 * self.theta_bins
                    data = data * fac
                if errors and not diff:
                    ax.errorbar(
                        self.theta_bins, data, fac * self.errors[2][f2, f1], ls=ls
                    )
                else:
                    ax.semilogx(self.theta_bins, data, ls=ls)
                if corrs_t is not None and not diff:
                    ax.semilogx(self.theta_bins, fac * corrs_t[f2, f1], ls=ls)
                ax.axvspan(*self.ranges["gammat"][f2][f1], color="gray", alpha=0.1)
                ax.set_title(f"{f2 + 1}-{f1 + 1}")
        return axs


# Conversion .fits --> .dataset  #########################################################


def convert_txt(filename, root, outdir, ranges=None):  # pragma: no cover
    import astropy.io.fits as fits  # type: ignore

    if ranges is None:
        ranges = get_def_cuts()
    hdulist = fits.open(filename)
    outlines = []
    outlines += ["measurements_format = DES"]
    outlines += ["kmax = 10"]  # matches what DES used and is good enough for likelihood
    outlines += ["intrinsic_alignment_model = DES1YR"]
    outlines += ["data_types = xip xim gammat wtheta"]
    outlines += ["used_data_types = xip xim gammat wtheta"]
    outlines += ["num_z_bins = %s" % (max(hdulist["xip"].data["BIN1"]))]
    outlines += ["num_gal_bins = %s" % (max(hdulist["wtheta"].data["BIN1"]))]
    ntheta = max(hdulist["wtheta"].data["ANGBIN"]) + 1
    outlines += ["num_theta_bins = %s" % ntheta]
    thetas = hdulist["xip"].data["ANG"][:ntheta]
    np.savetxt(outdir + root + "_theta_bins.dat", thetas, header="theta_arcmin")
    outlines += ["theta_bins_file = %s" % (root + "_theta_bins.dat")]
    np.savetxt(outdir + root + "_cov.dat", hdulist["COVMAT"].data, fmt="%.4e")
    outlines += ["cov_file = %s" % (root + "_cov.dat")]
    out_ranges = []
    for i, tp in enumerate(def_DES_types):
        pairs = []
        for b1, b2 in zip(hdulist[tp].data["BIN1"], hdulist[tp].data["BIN2"]):
            if (b1, b2) not in pairs:
                pairs.append((b1, b2))
        for x, y in pairs:
            out_ranges += [
                "{} {} {} {} {}".format(
                    tp, x, y, ranges[tp][x - 1][y - 1][0], ranges[tp][x - 1][y - 1][1]
                )
            ]
        # drop theta value, as assuming shared to all data
        dat = np.asarray(
            zip(*[hdulist[tp].data[n] for n in list(hdulist[tp].data.names)[:-2]])
        )
        # fix anomaly that z bins are 1 based but theta bins zero based
        dat[:, 2] += 1
        np.savetxt(
            outdir + root + "_%s.dat" % tp,
            dat,
            fmt=["%u", "%u", "%u", "%.8e"],
            header=" ".join(list(hdulist[tp].data.dtype.names)[:-2]),
        )
        outlines += [f"measurements[{tp}] = {root}_{tp}.dat"]
    sourcedata = hdulist["NZ_SOURCE"].data
    maxi = sourcedata.shape[0] - 1
    while np.all(np.asarray(sourcedata[maxi][3:]) == 0):
        maxi -= 1
    np.savetxt(
        outdir + root + "_nz_source.dat",
        sourcedata[: maxi + 1],
        fmt="%.6e",
        header=" ".join(hdulist["NZ_SOURCE"].data.dtype.names),
    )
    outlines += ["nz_file = %s_nz_source.dat" % root]
    assert np.all(np.asarray(hdulist["NZ_LENS"].data[maxi + 1][3:]) == 0)
    np.savetxt(
        outdir + root + "_nz_lens.dat",
        hdulist["NZ_LENS"].data[: maxi + 1],
        fmt="%.6e",
        header=" ".join(hdulist["NZ_LENS"].data.dtype.names),
    )
    outlines += ["nz_gal_file = %s_nz_lens.dat" % root]
    with open(outdir + root + "_selection.dat", "w", encoding="utf-8") as f:
        f.write("#  type bin1 bin2 theta_min theta_max\n")
        f.write("\n".join(out_ranges))
    outlines += ["data_selection = %s_selection.dat" % root]
    outlines += ["nuisance_params = DES.paramnames"]
    with open(outdir + root + ".dataset", "w", encoding="utf-8") as f:
        f.write("\n".join(outlines))
