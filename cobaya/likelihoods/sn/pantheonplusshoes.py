from .pantheonplus import PantheonPlus


class PantheonPlusShoes(PantheonPlus):
    """
    Likelihood for Pantheon+ (with SH0ES) type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2202.04077
    """

    def configure(self):
        self._apply_mask((self.zcmb > 0.01) | self.is_calibrator)
        self.pre_vars = 0.0  # diagonal component

    def _read_data_file(self, data_file):
        file_cols = ["m_b_corr", "zhd", "zhel", "is_calibrator", "ceph_dist"]
        self.cols = ["mag", "zcmb", "zhel", "is_calibrator", "ceph_dist"]
        self._read_cols(data_file, file_cols)
        self.is_calibrator = self.is_calibrator.astype(dtype="?")

    def alpha_beta_logp(self, lumdists, Mb=0, **kwargs):
        if self.use_abs_mag:
            estimated_scriptm = Mb + 25
        else:
            estimated_scriptm = 0.0

        # Use Cepheids host distances as theory
        lumdists[self.is_calibrator] = self.ceph_dist[self.is_calibrator] - 25.0
        diffmag = self.mag - lumdists - estimated_scriptm
        return -diffmag.dot(self.invcov).dot(diffmag) / 2.0
