from .pantheonplus import PantheonPlus


class Union3(PantheonPlus):
    """
    Likelihood for the Union3 & UNITY1.5 type Ia supernovae sample.

    Reference
    ---------
    https://arxiv.org/pdf/2311.12098.pdf
    """

    def configure(self):
        self.pre_vars = 0.0  # diagonal component

    def _read_data_file(self, data_file):
        file_cols = ["zcmb", "mb"]
        self.cols = ["zcmb", "mag"]
        self._read_cols(data_file, file_cols)
        self.zhel = self.zcmb
