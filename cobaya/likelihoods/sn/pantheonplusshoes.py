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
        self.pre_vars = 0.  # diagonal component

    def alpha_beta_logp(self, lumdists, Mb=0, **kwargs):
        if self.use_abs_mag:
            estimated_scriptm = Mb + 25
        else:
            estimated_scriptm = 0.

        # Use Cepheids host distances as theory
        lumdists[self.is_calibrator] = self.ceph_dist[self.is_calibrator]
        diffmag = self.mag - lumdists - estimated_scriptm
        return - diffmag.dot(self.invcov).dot(diffmag) / 2.
