import numpy as np
from cobaya.likelihoods._base_classes._planck_2018_CamSpec_python import _planck_2018_CamSpec_python

class planck_2020_CamSpec_python(_planck_2018_CamSpec_python):
    install_options = {
        "download_url":
            r"https://cdn.cosmologist.info/cosmobox/test2019_kaml/CamSpec2020.zip",
        "data_path": "CamSpec2020"}
    def get_powerlaw_residuals(self, data_params):
        
        amp = np.empty(4)
        amp[0] = data_params['amp_100']
        amp[1] = data_params['amp_143']
        amp[2] = data_params['amp_217']
        amp[3] = data_params['amp_143x217']

        tilt = np.empty(4)
        tilt[0] = data_params['n_100']
        tilt[1] = data_params['n_143']
        tilt[2] = data_params['n_217']
        tilt[3] = data_params['n_143x217']

        powerlaw_pivot=1500
        C_powerlaw = np.array([amp[ii] * (self.ls/powerlaw_pivot)**tilt[ii] for ii in range(4)])

        return C_powerlaw

    def chi_squared(self, CT, CTE, CEE, data_params):
        use_fg_residual_model = data_params['use_fg_residual_model']
        cals = self.get_cals(data_params)
        
        if np.any(self.cl_used[:4]):
            if use_fg_residual_model == 0:
                foregrounds = self.get_powerlaw_residuals(data_params)
            elif use_fg_residual_model == 1:
                foregrounds = self.get_foregrounds(data_params)
            elif use_fg_residual_model == 2:
                foregrounds = self.get_foregrounds(data_params) + self.get_powerlaw_residuals(data_params)
            else:
                raise ValueError("use_fg_residual_model should be 0 (powerlaw), 1 (foregrounds) or 2 (both)")
        delta_vector = self.data_vector.copy()
        ix = 0
        for i, (cal, n) in enumerate(zip(cals, self.used_sizes)):
            if n > 0:
                if i <= 3:
                    delta_vector[ix:ix + n] -= (CT[self.ell_ranges[i]] +
                                                foregrounds[i][self.ell_ranges[i]]) / cal
                elif i == 4:
                    delta_vector[ix:ix + n] -= CTE[self.ell_ranges[i]] / cal
                elif i == 5:
                    delta_vector[ix:ix + n] -= CEE[self.ell_ranges[i]] / cal
                ix += n
        return self._fast_chi_squared(self.covinv, delta_vector)
    
