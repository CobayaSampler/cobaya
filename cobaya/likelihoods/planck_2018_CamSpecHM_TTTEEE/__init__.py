from cobaya.likelihoods._planck_clik_prototype import _planck_clik_prototype
from cobaya.likelihoods.planck_2018_CamSpec import planck_2018_CamSpec_TT, planck_2018_CamSpec_TE, \
    planck_2018_CamSpec_EE


class planck_2018_CamSpecHM_TTTEEE(_planck_clik_prototype, planck_2018_CamSpec_TT, planck_2018_CamSpec_TE, \
                                   planck_2018_CamSpec_EE):
    pass
