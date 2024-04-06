import os
import numpy as np
from .pantheonplus import pantheonplus


class desy5(pantheonplus):
    """
    Likelihood for DES-Y5 type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2401.02929
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
        self.pre_vars = self.mag_err ** 2  # diagonal component
        # argsort = np.argsort(self.zcmb)
        # for col in self.cols:
        #    setattr(self, col, getattr(self, col)[argsort])
        # for name, cov in self.covs.items():
        #    self.covs[name] = cov[np.ix_(argsort, argsort)]
        self.inverse_covariance_matrix()
        if not self.use_abs_mag:
            self._marginalize_abs_mag()
        self.marginalize = False

    def _read_data_file(self, data_file):
        self.log.debug('Reading %s' % data_file)
        sep = ','
        supernovae = {}
        self.names = []
        oldcols = ['zhd', 'zhel', 'mu', 'muerr_final']
        self.cols = ['zcmb', 'zhel', 'mag', 'mag_err']
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for iline, line in enumerate(lines):
                if not line.startswith('#'):
                    break
            lines = lines[iline:]
            line = lines[0]
            cols = [col.strip().lower() for col in line.split(sep)]
            indices = [cols.index(col) for col in oldcols]
            zeros = np.zeros(len(lines) - 1)
            dtypes = {}
            for col in self.cols:
                setattr(self, col, zeros.astype(dtype=dtypes.get(col, 'f8'), copy=True))
            for ix, line in enumerate(lines[1:]):
                vals = [val.strip() for val in line.split(sep)]
                vals = [vals[i] for i in indices]
                for i, (col, val) in enumerate(zip(self.cols, vals)):
                    tmp = getattr(self, col)
                    tmp[ix] = np.asarray(val, dtype=tmp.dtype)
        self.nsn = ix + 1
        self.log.debug('Number of SN read: %s ' % self.nsn)
