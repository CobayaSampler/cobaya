import os

import numpy as np

from cobaya.likelihoods.base_classes import InstallableLikelihood


class EE(InstallableLikelihood):
    """
    Python translation of the Planck 2018 SimALl EE likelihood (python AL, Oct 2022)
    See https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code
    Equivalent to planck_2018_lowl.EE: the public Planck clik likelihood using file
    simall_100x143_offlike5_EE_Aplanck_B.clik (data converted to prob_table.txt)

    Calibration error is very small compared to EE uncertainty, but calibration can be
    used; otherwise taken to be 1
    """

    install_options = {
        "github_repository": "CobayaSampler/planck_native_data",
        "github_release": "v1",
        "asset": "planck_2018_lowE.zip",
        "directory": "planck_2018_lowE_native",
    }
    type = "CMB"
    aliases = ["lowE"]

    _lmin = 2
    _lmax = 29
    _nstepsEE = 3000
    _stepEE = 0.0001
    _table_file_name = "prob_table.txt"

    @classmethod
    def get_bibtex(cls):
        from cobaya.likelihoods.base_classes import Planck2018Clik

        return Planck2018Clik.get_bibtex()

    def initialize(self):
        if self.get_install_options() and self.packages_path:
            path = self.get_path(self.packages_path)
            self.probEE = np.loadtxt(os.path.join(path, self._table_file_name))

    def get_can_support_params(self):
        return ["A_planck"]

    def get_requirements(self):
        return {"Cl": {"ee": self._lmax}}

    def log_likelihood(self, cls_EE, calib=1):
        r"""
        Calculate log likelihood from CMB EE spectrum by using likelihood table

        :param cls_EE: L(L+1)C_L/2pi zero-based array in muK^2 units
        :param calib: optional calibration parameter
        :return: log likelihood
        """
        EE_index = (
            cls_EE[self._lmin : self._lmax + 1] / (calib**2 * self._stepEE)
        ).astype(int)
        try:
            return np.take_along_axis(self.probEE, EE_index[np.newaxis, :], 0).sum()
        except IndexError:
            self.log.warning("low EE multipole out of range, rejecting point")
            return -np.inf

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)["ee"]
        return self.log_likelihood(cls, params_values.get("A_planck", 1))
