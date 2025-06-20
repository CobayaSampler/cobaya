from cobaya.likelihoods.base_classes import CMBlikes
from cobaya.likelihoods.base_classes.planck_clik import last_version_supp_data_and_covmats


class native(CMBlikes):
    r"""
    Lensing likelihood of Planck's 2018 data release based on temperature+polarization
    map-based lensing reconstruction \cite{Aghanim:2018oex} (native Python
    re-implementation by A.~Lewis).
    """

    install_options = {
        "github_repository": "CobayaSampler/planck_supp_data_and_covmats",
        "github_release": last_version_supp_data_and_covmats,
    }

    bibtex_file = "PlanckLensing2018.bibtex"


class CMBMarged(native):
    r"""
    Lensing likelihood of Planck's 2018 data release based on temperature+polarization
    map-based lensing reconstruction, marginalized over the CMB power spectra
    \cite{Aghanim:2018oex} (native Python re-implementation by A.~Lewis).
    """

    pass
