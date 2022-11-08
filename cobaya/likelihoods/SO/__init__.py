import numpy as np
from cobaya import Likelihood
from cobaya.likelihoods.base_classes import CMBlikes
import os
from camb.mathutils import chi_squared

dir = 'C:\Tmp\SO\LCDM-likelihoods'
if not os.path.exists(dir):
    dir = os.path.dirname(__file__)


class Var_like(CMBlikes):
    variant = ''

    def initialize(self):
        self.dataset_file = os.path.join(dir, self.dataset_file % self.variant)
        super().initialize()


class SO_lensing_baseline(Var_like):
    dataset_file = 'SO_lensing_baseline%s.dataset'


class SO_ILC_baseline(Var_like):
    dataset_file = 'SO_ILC_baseline%s.dataset'


class PlanckMidT(Var_like):
    dataset_file = 'PlanckMidT%s.dataset'


class PlanckLowT(Var_like):
    dataset_file = 'PlanckLowT.dataset'


class SO_corr_like(Likelihood):
    lmax = 4000
    lmin = 40
    fsky = 0.4
    inv_cov_file = "covinv.npy"
    variant = ""

    def get_requirements(self):
        return {
            'Cl': {'tt': self.lmax, 'ee': self.lmax, 'te': self.lmax, 'pp': self.lmax}}

    def initialize(self):
        import pickle
        phi = np.loadtxt(os.path.join(dir, 'SO_lensing_baseline%s.dat' % self.variant))
        cmb = np.loadtxt(os.path.join(dir, 'SO_ILC_baseline%s.dat' % self.variant))
        res = {'pp': np.concatenate(([0, 0], phi[:, 1])),
               'tt': np.concatenate(([0, 0], cmb[:, 1])) / 2.726e6 ** 2,
               'ee': np.concatenate(([0, 0], cmb[:, 3])) / 2.726e6 ** 2,
               'te': np.concatenate(([0, 0], cmb[:, 2])) / 2.726e6 ** 2}
        self.data_vector = self.make_vec(res)
        with open(os.path.join(dir, self.inv_cov_file), "rb") as f:
            self.covinv = pickle.load(f)

    def make_vec(self, cls):
        num_SO_ls = self.lmax - self.lmin + 1
        vec = np.empty(num_SO_ls * 4)
        vec[:num_SO_ls] = cls['tt'][self.lmin:self.lmax + 1]
        vec[num_SO_ls:num_SO_ls * 2] = cls['ee'][self.lmin:self.lmax + 1]
        vec[num_SO_ls * 2:num_SO_ls * 3] = cls['te'][self.lmin:self.lmax + 1]
        vec[num_SO_ls * 3:num_SO_ls * 4] = cls['pp'][self.lmin:self.lmax + 1]
        return vec

    def log_likelihood(self, cls):
        theory = self.make_vec(cls)
        return -self.fsky * chi_squared(self.covinv, self.data_vector - theory) / 2

    def logp(self, **params_values):
        return self.log_likelihood(self.provider.get_Cl(ell_factor=True, units='1'))


class SO_corr_like_gauss(SO_corr_like):
    inv_cov_file = "covinv_gauss.npy"
