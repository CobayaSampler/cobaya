import os

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError


class TT(InstallableLikelihood):
    """
    Python translation of the Planck 2018 Gibbs TT likelihood
    (python Eirik Gjerløw, Feb 2023)
    See https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code

    """

    install_options = {
        "github_repository": "CobayaSampler/planck_native_data",
        "github_release": "v1",
        "asset": "planck_2018_lowT.zip",
        "directory": "planck_2018_lowT_native",
    }
    lmin: int = 2
    lmax: int = 29
    type = "CMB"
    aliases = ["lowT"]

    @classmethod
    def get_bibtex(cls):
        from cobaya.likelihoods.base_classes import Planck2018Clik

        return Planck2018Clik.get_bibtex()

    def get_requirements(self):
        return {"Cl": {"tt": self.lmax}}

    def get_can_support_params(self):
        return ["A_planck"]

    def initialize(self):
        if self.get_install_options() and self.packages_path:
            if self.lmin < 2 or self.lmax > 200 or self.lmin >= self.lmax:
                raise LoggedError(
                    self.log,
                    "lmin must be >= 2, lmax must be <= 200,\n"
                    "and lmin must be less than lmax.",
                )

            path = self.get_path(self.packages_path)

            # The txt files start at l=2, hence the index gymnastics
            cov = np.loadtxt(os.path.join(path, "cov.txt"))[
                self.lmin - 2 : self.lmax + 1 - 2, self.lmin - 2 : self.lmax + 1 - 2
            ]
            # The inverse covariance matrix for the gaussian likelihood calculation
            self._covinv = np.linalg.inv(cov)
            # The average cl's for the gaussian likelihood calculation
            self._mu = np.ascontiguousarray(
                np.loadtxt(os.path.join(path, "mu.txt"))[
                    self.lmin - 2 : self.lmax + 1 - 2
                ]
            )
            # The cl's used for offset calculation - hence the full range of ells
            mu_sigma = np.zeros(self.lmax + 1)
            mu_sigma[self.lmin :] = np.loadtxt(os.path.join(path, "mu_sigma.txt"))[
                self.lmin - 2 : self.lmax + 1 - 2
            ]

            # Spline info
            nbins = 1000
            spline_cl = np.loadtxt(os.path.join(path, "cl2x_1.txt"))[
                :, self.lmin - 2 : self.lmax + 1 - 2
            ]
            spline_val = np.loadtxt(os.path.join(path, "cl2x_2.txt"))[
                :, self.lmin - 2 : self.lmax + 1 - 2
            ]
            self._spline = []
            self._spline_derivative = []

            # Set up prior and spline
            self._prior_bounds = np.zeros((self.lmax + 1 - self.lmin, 2))
            for i in range(self.lmax - self.lmin + 1):
                j = 0
                while abs(spline_val[j, i] + 5) < 1e-4:
                    j += 1
                self._prior_bounds[i, 0] = spline_cl[j + 2, i]
                j = nbins - 1
                while abs(spline_val[j, i] - 5) < 1e-4:
                    j -= 1
                self._prior_bounds[i, 1] = spline_cl[j - 2, i]
                self._spline.append(
                    InterpolatedUnivariateSpline(spline_cl[:, i], spline_val[:, i])
                )
                self._spline_derivative.append(self._spline[-1].derivative())
            # initialize offset to normalize like a chi-squared
            self._offset = 0
            self._offset = self.log_likelihood(mu_sigma)

    def log_likelihood(self, cls_TT, calib=1):
        r"""
        Calculate log likelihood from CMB TT spectrum

        :param cls_TT: L(L+1)C_L/2pi zero-based array in muK^2 units
        :param calib: optional calibration parameter
        :return: log likelihood
        """

        theory: np.ndarray = cls_TT[self.lmin : self.lmax + 1] / calib**2

        if any(theory < self._prior_bounds[:, 0]) or any(
            theory > self._prior_bounds[:, 1]
        ):
            return -np.inf

        logl = 0.0
        # Convert the cl's to Gaussianized variables
        x = np.zeros_like(theory)
        for i, (spline, diff_spline, cl) in enumerate(
            zip(self._spline, self._spline_derivative, theory)
        ):
            dxdCl = diff_spline(cl)
            if dxdCl < 0:
                return -np.inf
            logl += np.log(dxdCl)
            x[i] = spline(cl)

        delta = x - self._mu
        logl += -0.5 * self._covinv.dot(delta).dot(delta)
        logl -= self._offset

        return logl

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)["tt"]
        return self.log_likelihood(cls, params_values.get("A_planck", 1))
