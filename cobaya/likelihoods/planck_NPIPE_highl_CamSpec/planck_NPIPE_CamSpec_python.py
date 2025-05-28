from cobaya.likelihoods.planck_2018_highl_CamSpec2021.planck_2018_CamSpec2021_python import (
    Planck2018CamSpec2021Python,
)


class Planck2020CamSpecPython(Planck2018CamSpec2021Python):
    install_options = {
        "github_repository": "CobayaSampler/planck_native_data",
        "github_release": "v1",
        "asset": "CamSpec_NPIPE.zip",
        "directory": "planck_NPIPE_CamSpec",
    }

    bibtex_file = "CamSpec_NPIPE_2022.bibtex"
    _is_abstract = True
