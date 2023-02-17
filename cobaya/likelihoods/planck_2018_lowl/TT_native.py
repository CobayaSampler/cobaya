import os
import numpy as np
from cobaya.likelihoods.base_classes import InstallableLikelihood

#class TT_native(InstallableLikelihood):
class TT_native():
    """
    Python translation of the Planck 2018 Gibbs TT likelihood (python Eirik Gjerløw, Feb 2023)
    See https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code

    """

    install_options = {"github_repository": "CobayaSampler/planck_native_data",
                       "github_release": "v1",
                       "asset": "planck_2018_lowT.zip",
                       "directory": "planck_2018_lowT_native"}

    type = "CMB"
    aliases = ["lowT"]

    _lmin = 2
    _lmax = 29 # Could in principle be extended to 200
    _delta_l = 1000
    _nbins = 1000

    #The inverse covariance matrix for the gaussian likelihood calculation
    _covinv = np.zeros((_lmax-_lmin+1, _lmax-_lmin+1))
    # The spline coefficients for the cl's
    _cl2x = np.zeros((_nbins, _lmax-_lmin+1, 3))
    # The average cl's for the gaussian likelihood calculation
    _mu = np.zeros(_lmax-_lmin+1)
    # The cl's used for offset calculation - hence the full range of ells
    _mu_sigma = np.zeros(_lmax+1)

    @classmethod
    def get_bibtex(cls):
        from cobaya.likelihoods.planck_2018_lowl.TT import TT
        return TT.get_bibtex()

    def get_requirements(self):
        return {'Cl': {'tt': self._lmax}}

    def initialize(self):
#        if self.get_install_options() and self.packages_path:
        if True:
            path = '/home/eirik/data/clik_antony/'
#            path = self.get_path(self.packages_path)
            # The txt files start at l=2, hence the index gymnastics
            self._covinv[:, :] = np.loadtxt(os.path.join(path, 'covinv.txt'))[self._lmin-2:self._lmax+1-2, self._lmin-2:self._lmax+1-2]
            self._cl2x[:, :, 0] = np.loadtxt(os.path.join(path, 'cl2x_1.txt'))[:, self._lmin-2:self._lmax+1-2]
            self._cl2x[:, :, 1] = np.loadtxt(os.path.join(path, 'cl2x_2.txt'))[:, self._lmin-2:self._lmax+1-2]
            self._cl2x[:, :, 2] = np.loadtxt(os.path.join(path, 'cl2x_3.txt'))[:, self._lmin-2:self._lmax+1-2]
            self._mu[:] = np.loadtxt(os.path.join(path, 'mu.txt'))[self._lmin-2:self._lmax+1-2]
            self._mu_sigma[self._lmin:] = np.loadtxt(os.path.join(path, 'mu_sigma.txt'))[self._lmin-2:self._lmax+1-2]

            # Set up prior
            self._prior = np.zeros((self._lmax+1-self._lmin, 2))
            for l in range(self._lmax-self._lmin+1):
                j = 0
                while abs(self._cl2x[j, l, 1] + 5) < 1e-4:
                    j += 1
                self._prior[l, 0] = self._cl2x[j+2, l, 0]
                j = self._nbins-1
                while abs(self._cl2x[j, l, 1] - 5) < 1e-4:
                    j -= 1
                self._prior[l, 1] = self._cl2x[j-2, l, 0]

            self._offset = self.log_likelihood(self._mu_sigma, init=True)
                

    def get_requirements(self):
        return {'Cl': {'tt': self._lmax}}


    def log_likelihood(self, cls_TT, init=False):
        cls_eval = cls_TT[self._lmin:self._lmax+1]

        if any(cls_eval < self._prior[:, 0]) or any(cls_eval > self._prior[:, 1]):
            return - np.inf

        # Convert the cl's to Gaussianized variables
        x = np.zeros(self._lmax+1-self._lmin)
        for l in range(self._lmax+1-self._lmin):
            x[l] = self._splint_gauss_br(self._cl2x[:, l, 0],
                                         self._cl2x[:, l, 1],
                                         self._cl2x[:, l, 2],
                                         cls_eval[l])
            delta = x - self._mu
        logl = -0.5 * self._covinv.dot(delta).dot(delta) 

        # Add Jacobian term
        for l in range(self._lmax+1-self._lmin):
            dxdCl = self._splint_deriv_gauss_br(self._cl2x[:, l, 0],
                                                self._cl2x[:, l, 1],
                                                self._cl2x[:, l, 2],
                                                cls_eval[l])
            if dxdCl < 0:
                return -np.inf
            else:
                logl += np.log(dxdCl)
        if not init:
            logl -= self._offset

        return logl

    def _splint_gauss_br(self, xa, ya, y2a, x):
        #Taken from numerical recipes
        n = len(xa)
        klo = max(min(self._locate_gauss_br(xa, x),n-2),0)
        khi = klo + 1
        h = xa[khi] - xa[klo]
        a = (xa[khi] - x) / h
        b = (x - xa[klo]) / h
        return a * ya[klo] + b * ya[khi] + ((a**3 - a) * y2a[klo] + (b**3 - b) * y2a[khi]) * (h**2) / 6.0


    def _splint_deriv_gauss_br(self, xa, ya, y2a, x):
        #Taken from numerical recipes
        n = len(xa)
        klo = max(min(self._locate_gauss_br(xa, x), n-2), 0)
        khi = klo + 1
        h = xa[khi] - xa[klo]
        a = (xa[khi] - x) / h
        b = (x - xa[klo]) / h
        return (ya[khi] - ya[klo]) / h - (3.0 * a ** 2 - 1.0) / 6.0 * h * y2a[klo] + (3.0 * b ** 2 - 1.0) / 6.0 * h * y2a[khi]


    def _locate_gauss_br(self, xx, x):
        #Taken from numerical recipes
        
        n = len(xx)
        ascnd = (xx[n-1] >= xx[0])
        jl = -1
        ju = n

        while True:
            if (ju - jl <= 1): break
            jm = int((ju + jl) / 2)
            if (ascnd == (x >= xx[jm])):
                jl = jm
            else:
                ju = jm
        if x == xx[0]:
            return 0
        elif x == xx[n-1]:
            return n - 1
        else:
            return jl


    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)['tt']
        return self.log_likelihood(cls)
