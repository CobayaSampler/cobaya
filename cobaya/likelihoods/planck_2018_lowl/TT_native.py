import os
import numpy as np
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError
from scipy.interpolate import InterpolatedUnivariateSpline


class TT_native(InstallableLikelihood):
    """
    Python translation of the Planck 2018 Gibbs TT likelihood (python Eirik Gjerløw, Feb 2023)
    See https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code

    """

    install_options = {"github_repository": "CobayaSampler/planck_native_data",
                       "github_release": "v1",
                       "asset": "planck_2018_lowT.zip",
                       "directory": "planck_2018_lowT_native"}
    lmin = 2
    lmax = 29
    type = "CMB"
    aliases = ["lowT"]

    @classmethod
    def get_bibtex(cls):
        from cobaya.likelihoods.planck_2018_lowl.TT import TT
        return TT.get_bibtex()

    def get_requirements(self):
        return {'Cl': {'tt': self.lmax}}

    def get_can_support_params(self):
        return ['A_planck']

    def initialize(self):
        if self.get_install_options() and self.packages_path:
            if self.lmin < 2 or self.lmax > 200 or self.lmin >= self.lmax:
                raise LoggedError(
                    self.log, "lmin must be >= 2, lmax must be <= 200,\n"
                              "and lmin must be less than lmax.")
            #The inverse covariance matrix for the gaussian likelihood calculation
            self._covinv = np.zeros((self.lmax - self.lmin + 1,
                                     self.lmax - self.lmin + 1))
            # The average cl's for the gaussian likelihood calculation
            self._mu = np.zeros(self.lmax - self.lmin + 1)
            # The cl's used for offset calculation - hence the full range of ells
            mu_sigma = np.zeros(self.lmax + 1)

            # The txt files start at l=2, hence the index gymnastics
            path = self.get_path(self.packages_path)

            cov = np.loadtxt(
                os.path.join(path, 'cov.txt'))[
                    self.lmin - 2:self.lmax + 1 - 2,
                    self.lmin - 2:self.lmax + 1 - 2]
            self._covinv[:, :] = np.linalg.inv(cov)
            self._mu[:] = np.loadtxt(
                os.path.join(path, 'mu.txt'))[self.lmin - 2:self.lmax + 1 - 2]
            mu_sigma[self.lmin:] = np.loadtxt(
                os.path.join(path, 'mu_sigma.txt'))[self.lmin - 2:self.lmax + 1 - 2]

            # Spline info
            nbins = 1000
            cl2x = np.zeros((nbins, self.lmax - self.lmin + 1, 2))
            cl2x[:, :, 0] = np.loadtxt(os.path.join(path, 'cl2x_1.txt'))[:, self.lmin - 2:self.lmax + 1 - 2]
            cl2x[:, :, 1] = np.loadtxt(os.path.join(path, 'cl2x_2.txt'))[:, self.lmin - 2:self.lmax + 1 - 2]
            self._spline = []
            self._spline_derivative = []

            # Set up prior and spline
            self._prior_bounds = np.zeros((self.lmax + 1 - self.lmin, 2))
            for i in range(self.lmax - self.lmin + 1):
                j = 0
                while abs(cl2x[j, i, 1] + 5) < 1e-4:
                    j += 1
                self._prior_bounds[i, 0] = cl2x[j + 2, i, 0]
                j = nbins - 1
                while abs(cl2x[j, i, 1] - 5) < 1e-4:
                    j -= 1
                self._prior_bounds[i, 1] = cl2x[j - 2, i, 0]
                self._spline.append(
                    InterpolatedUnivariateSpline(cl2x[:, i, 0], cl2x[:, i, 1]))
                self._spline_derivative.append(self._spline[-1].derivative())

            self._offset = self.log_likelihood(mu_sigma, init=True)


    def log_likelihood(self, cls_TT, calib=1, init=False):
        theory = cls_TT[self.lmin:self.lmax + 1] / calib ** 2

        if any(theory < self._prior_bounds[:, 0]) or any(theory > self._prior_bounds[:, 1]):
            return - np.inf

        logl = 0
        # Convert the cl's to Gaussianized variables
        x = np.zeros_like(theory)
        for i, (spline, diff_spline, cl) in enumerate(
                zip(self._spline, self._spline_derivative, theory)):
            x[i] = spline(cl)
            dxdCl = diff_spline(cl)
            if dxdCl < 0:
                return -np.inf
            logl += np.log(dxdCl)
        delta = x - self._mu
        logl += -0.5 * self._covinv.dot(delta).dot(delta)

        if not init:
            logl -= self._offset

        return logl

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)['tt']
        return self.log_likelihood(cls, params_values.get('A_planck', 1))
