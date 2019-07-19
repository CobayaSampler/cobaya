from cobaya.likelihoods.planck_2018_CamSpec import planck_2018_CamSpec_python, planck_2018_CamSpec_TT, \
    planck_2018_CamSpec_TE, planck_2018_CamSpec_EE


class CamSpec_TTTEEE_python(planck_2018_CamSpec_python, planck_2018_CamSpec_TT, planck_2018_CamSpec_TE,
                            planck_2018_CamSpec_EE):
    pass
