import os
import numpy as np
from .pantheonplus import pantheonplus


class pantheonplusshoes(pantheonplus):
    """
    Likelihood for Pantheon+ (with SH0ES) type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2202.04077
    """

    def init_params(self, ini):
        self.twoscriptmfit = False
        # self.pecz = 0.
        # self.has_third_var = False
        data_file = os.path.normpath(os.path.join(self.path, ini.string("data_file")))
        self._read_data_file(data_file)
        self.covs = {}
        for name in ['mag']:
            self.log.debug('Reading covmat for: %s ' % name)
            self.covs[name] = self._read_covmat(
                os.path.join(self.path, ini.string('%s_covmat_file' % name)))
        self.alphabeta_covmat = False
        zmask = (self.zcmb > 0.01) | self.is_calibrator
        for col in self.cols:
            setattr(self, col, getattr(self, col)[zmask])
        for name, cov in self.covs.items():
            self.covs[name] = cov[np.ix_(zmask, zmask)]
        self.pre_vars = 0.  # diagonal component
        self.inverse_covariance_matrix()
        if not self.use_abs_mag:
            self._marginalize_abs_mag()
        self.marginalize = False
        print(self.zcmb)

    def alpha_beta_logp(self, lumdists, Mb=0):
        if self.use_abs_mag:
            estimated_scriptm = Mb + 25
        else:
            estimated_scriptm = 0.

        # Use Cepheids host distances as theory
        lumdists[self.is_calibrator] = self.ceph_dist[self.is_calibrator]
        diffmag = self.mag - lumdists - estimated_scriptm
        return - diffmag.dot(self.invcov).dot(diffmag) / 2.
