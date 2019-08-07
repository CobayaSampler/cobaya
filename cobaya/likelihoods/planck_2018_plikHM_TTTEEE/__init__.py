from cobaya.likelihoods._planck_clik_prototype import _planck_clik_prototype
from cobaya.likelihoods._planck_clik_prototype import install as install_common
from cobaya.likelihoods._planck_clik_prototype import is_installed as is_installed_common


class planck_2018_plikHM_TTTEEE(_planck_clik_prototype):
    defaults = """
        likelihood:
          planck_2018_plikHM_TTTEEE:
            path:
            clik_file: baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik
            # product_id: "1900"
            # Aliases for automatic covariance matrix
            renames: [plikHM_TTTEEE]
            # Speed in evaluations/second
            speed: 7
            # Nuisance parameters (do not change)
            nuisance: [calib, TT, TEEE]"""


def is_installed(**kwargs):
    kwargs["name"] = __name__.split(".")[-1]
    return is_installed_common(**kwargs)


def install(**kwargs):
    kwargs["name"] = __name__.split(".")[-1]
    return install_common(**kwargs)
