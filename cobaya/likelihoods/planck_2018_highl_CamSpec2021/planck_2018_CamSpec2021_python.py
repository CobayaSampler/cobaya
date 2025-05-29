import numpy as np

from cobaya.likelihoods.base_classes.planck_2018_CamSpec_python import (
    Planck2018CamSpecPython,
)


class Planck2018CamSpec2021Python(Planck2018CamSpecPython):
    install_options = {
        "github_repository": "CobayaSampler/planck_native_data",
        "github_release": "v1",
        "asset": "CamSpec2021.zip",
        "directory": "planck_2018_CamSpec2021",
    }

    bibtex_file = "CamSpec2021.bibtex"
    _is_abstract = True

    def get_powerlaw_residuals(self, data_params):
        amp = np.empty(4)
        amp[0] = data_params["amp_100"]
        amp[1] = data_params["amp_143"]
        amp[2] = data_params["amp_217"]
        amp[3] = data_params["amp_143x217"]

        tilt = np.empty(4)
        tilt[0] = data_params["n_100"]
        tilt[1] = data_params["n_143"]
        tilt[2] = data_params["n_217"]
        tilt[3] = data_params["n_143x217"]

        powerlaw_pivot = 1500
        C_powerlaw = np.array(
            [amp[ii] * (self.ls / powerlaw_pivot) ** tilt[ii] for ii in range(4)]
        )

        return C_powerlaw

    def get_foregrounds(self, data_params):
        use_fg_residual_model = data_params["use_fg_residual_model"]
        if use_fg_residual_model == 0:
            foregrounds = self.get_powerlaw_residuals(data_params)
        elif use_fg_residual_model == 1:
            foregrounds = super().get_foregrounds(data_params)
        elif use_fg_residual_model == 2:
            foregrounds = super().get_foregrounds(
                data_params
            ) + self.get_powerlaw_residuals(data_params)
        else:
            raise ValueError(
                "use_fg_residual_model should be 0 (powerlaw), "
                "1 (foregrounds) or 2 (both)"
            )
        return foregrounds
