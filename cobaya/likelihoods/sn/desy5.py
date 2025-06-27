from .pantheonplus import PantheonPlus


class DESy5(PantheonPlus):
    """
    Likelihood for DES-Y5 type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/abs/2401.02929
    """

    def configure(self):
        self.pre_vars = self.mag_err**2

    def _read_data_file(self, data_file):
        file_cols = ["zhd", "zhel", "mu", "muerr_final"]
        self.cols = ["zcmb", "zhel", "mag", "mag_err"]
        self._read_cols(data_file, file_cols, sep=",")
