from .pantheonplus import PantheonPlus


class DESDovekie(PantheonPlus):
    """
    Likelihood for DES-Dovekie type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2511.07517
    """

    def _read_data_file(self, data_file):
        file_cols = ["zHD", "zHEL", "MU", "MUERR"]
        self.cols = ["zcmb", "zhel", "mag", "mag_err"]
        self._read_cols(data_file, file_cols, sep=r"\s+")

    def init_params(self, ini):
        self.twoscriptmfit = False
        data_file = os.path.normpath(os.path.join(self.path, ini.string("data_file")))
        self._read_data_file(data_file)
        self.covs = {}
        for name in ["mag"]:
            self.log.debug("Reading covmat for: %s " % name)
            self.covs[name] = self._read_inv_covmat(
                os.path.join(self.path, ini.string("%s_covmat_file" % name))
            )
        self.alphabeta_covmat = False
        self.configure()
        #Don't need to invert the covariance matrix here. 
        if not self.use_abs_mag:
            self._marginalize_abs_mag()
        self.marginalize = False

