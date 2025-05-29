import os

import numpy as np

from cobaya.likelihoods.base_classes import SN


class PantheonPlus(SN):
    """
    Likelihood for Pantheon+ (without SH0ES) type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2202.04077
    """

    def init_params(self, ini):
        self.twoscriptmfit = False
        data_file = os.path.normpath(os.path.join(self.path, ini.string("data_file")))
        self._read_data_file(data_file)
        self.covs = {}
        for name in ["mag"]:
            self.log.debug("Reading covmat for: %s " % name)
            self.covs[name] = self._read_covmat(
                os.path.join(self.path, ini.string("%s_covmat_file" % name))
            )
        self.alphabeta_covmat = False
        self.configure()
        self.inverse_covariance_matrix()
        if not self.use_abs_mag:
            self._marginalize_abs_mag()
        self.marginalize = False

    def _apply_mask(self, zmask):
        for col in self.cols:
            setattr(self, col, getattr(self, col)[zmask])
        for name, cov in self.covs.items():
            self.covs[name] = cov[np.ix_(zmask, zmask)]

    def configure(self):
        self._apply_mask(zmask=self.zcmb > 0.01)
        self.pre_vars = 0.0  # diagonal component

    def _read_cols(self, data_file, file_cols, sep=None):
        self.log.debug("Reading %s" % data_file)
        with open(data_file) as f:
            lines = f.readlines()
            line = lines[0]
            if line.startswith("#"):
                line = line[1:]
            cols = [col.strip().lower() for col in line.split(sep)]
            assert cols[0].isalpha()
            indices = [cols.index(col) for col in file_cols]
            zeros = np.zeros(len(lines) - 1)
            for col in self.cols:
                setattr(self, col, zeros.astype(dtype="f8", copy=True))
            for ix, line in enumerate(lines[1:]):
                vals = [val.strip() for val in line.split(sep)]
                vals = [vals[i] for i in indices]
                for i, (col, val) in enumerate(zip(self.cols, vals)):
                    tmp = getattr(self, col)
                    tmp[ix] = np.asarray(val, dtype=tmp.dtype)
        self.nsn = ix + 1
        self.log.debug("Number of SN read: %s " % self.nsn)

    def _read_data_file(self, data_file):
        file_cols = ["m_b_corr", "zhd", "zhel"]
        self.cols = ["mag", "zcmb", "zhel"]
        self._read_cols(data_file, file_cols)

    def _marginalize_abs_mag(self):
        deriv = np.ones_like(self.mag)[:, None]
        derivp = self.invcov.dot(deriv)
        fisher = deriv.T.dot(derivp)
        self.invcov = self.invcov - derivp.dot(np.linalg.solve(fisher, derivp.T))

    def alpha_beta_logp(self, lumdists, Mb=0.0, **kwargs):
        if self.use_abs_mag:
            estimated_scriptm = Mb + 25
        else:
            estimated_scriptm = 0.0
        diffmag = self.mag - lumdists - estimated_scriptm
        return -diffmag.dot(self.invcov).dot(diffmag) / 2.0
