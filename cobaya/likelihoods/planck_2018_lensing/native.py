from cobaya.likelihoods._base_classes import _cmblikes_prototype
from cobaya.likelihoods._base_classes._planck_clik_prototype import last_version_supp_data_and_covmats


class native(_cmblikes_prototype):
    install_options = {"github_repository": "CobayaSampler/planck_supp_data_and_covmats",
                       "github_release": last_version_supp_data_and_covmats}
