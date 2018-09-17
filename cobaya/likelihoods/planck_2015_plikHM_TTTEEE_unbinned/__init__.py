from cobaya.likelihoods._planck_clik_prototype import _planck_clik_prototype


class planck_2015_plikHM_TTTEEE_unbinned(_planck_clik_prototype):
    pass


from cobaya.likelihoods._planck_clik_prototype import install as install_common
from cobaya.likelihoods._planck_clik_prototype import is_installed as is_installed_common


def is_installed(**kwargs):
    kwargs["name"] = __name__.split(".")[-1]
    return is_installed_common(**kwargs)


def install(**kwargs):
    kwargs["name"] = __name__.split(".")[-1]
    return install_common(**kwargs)
