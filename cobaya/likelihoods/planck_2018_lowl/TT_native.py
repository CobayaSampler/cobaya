import os
import numpy as np
from cobaya.likelihoods.base_classes import InstallableLikelihood

class TT_native(InstallableLikelihood):
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
    _nbins = 0


    _data = {}
    _data['cov'] = np.zeros((251, 251))
    _data['cl2x'] = np.zeros((1000, 251, 3))
    _data['mu'] = np.zeros(251)
    _data['mu_sigma'] = np.zeros(251)

    @classmethod
    def get_bibtex(cls):
        from cobaya.likelihoods.planck_2018_lowl.TT import TT
        return TT.get_bibtex()

    def get_requirements(self):
        return {'Cl': {'tt': self._lmax}}

    def initialize(self):
        if self.get_install_options() and self.packages_path:
            path = self.get_path(self.packages_path)
            self._data['cov'][2:, 2:] = np.loadtxt(os.path.join(path, 'cov.txt'))
            self._data['cl2x'][:, 2:, 0] = np.loadtxt(os.path.join(path, 'cl2x_1.txt')) 
            self._data['cl2x'][:, 2:, 1] = np.loadtxt(os.path.join(path, 'cl2x_2.txt')) 
            self._data['cl2x'][:, 2:, 2] = np.loadtxt(os.path.join(path, 'cl2x_3.txt')) 
            self._data['mu'][2:] = np.loadtxt(os.path.join(path, 'mu.txt'))
            self._data['mu_sigma'][2:] = np.loadtxt(os.path.join(path, 'mu_sigma.txt'))
            _nbins = len(self._data['cl2x'][:, 0, 0])

            # Bandlimit covariance matrix
            for l in range(self._lmin, self._lmax+1):
                for k in range(self._lmin, self._lmax+1):
                    if abs(l-k) > self._delta_l:
                        self._data[cov][l, k] = 0

            # Set up prior
            self._data['prior'] = np.zeros((251, 2))
            for l in range(self._lmin, self._lmax+1):
                j = 0
                while abs(self._data['cl2x'][j, l, 1] + 5) < 1e-4:
                    j += 1
                self._data['prior'][l, 0] = self._data['cl2x'][j+2, l, 0]
                j = _nbins-1
                while abs(self._data['cl2x'][j, l, 1] - 5) < 1e-4:
                    j -= 1
                self._data['prior'][l, 1] = self._data['cl2x'][j-2, l, 0]

            np.savetxt('prior.txt', self._data['prior'])

            self._data['offset'] = self.log_likelihood(self._data['mu_sigma'], init=True)
                

    def get_requirements(self):
        return {'Cl': {'tt': self._lmax}}


    def log_likelihood(self, cls_TT, init=False):

        if (any(cls_TT[self._lmin:self._lmax+1] < self._data['prior'][self._lmin:self._lmax+1, 0]) or
                any(cls_TT[self._lmin:self._lmax+1] > self._data['prior'][self._lmin:self._lmax+1, 1])):
            return -1e30

        # Convert the cl's to Gaussianized variables
        x = np.zeros(self._lmax+1)
        for l in range(self._lmin, self._lmax+1):
            x[l] = self._splint_gauss_br(self._data['cl2x'][:, l, 0],
                                        self._data['cl2x'][:, l, 1],
                                        self._data['cl2x'][:, l, 2],
                                        cls_TT[l])
        logl = -0.5 * sum((x[self._lmin:self._lmax+1] - self._data['mu'][self._lmin:self._lmax+1]) * 
                          np.dot(self._data['cov'][self._lmin:self._lmax+1, self._lmin:self._lmax+1],
                                 (x[self._lmin:self._lmax+1] - self._data['mu'][self._lmin:self._lmax+1])))

        # Add Jacobian term
        for l in range(self._lmin, self._lmax+1):
            dxdCl = self._splint_deriv_gauss_br(self._data['cl2x'][:, l, 0],
                                               self._data['cl2x'][:, l, 1],
                                               self._data['cl2x'][:, l, 2],
                                               cls_TT[l])
            if dxdCl < 0:
                return -1e30
            else:
                logl += np.log(dxdCl)
        if not init:
            logl -= self._data['offset']

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
