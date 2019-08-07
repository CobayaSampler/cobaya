from cobaya.likelihoods._planck_clik_prototype import _planck_clik_prototype
from cobaya.likelihoods._planck_clik_prototype import install as install_common
from cobaya.likelihoods._planck_clik_prototype import is_installed as is_installed_common


class planck_2015_lensing(_planck_clik_prototype):

    defaults = r"""
        # Planck 2015 release: lensing T+P-based likelihood
        # See https://wiki.cosmos.esa.int/planckpla2015/index.php/CMB_spectrum_%26_Likelihood_Code

        likelihood:
          planck_2015_lensing:
            path:
            clik_file: plc_2.0/lensing/smica_g30_ftl_full_pp.clik_lensing
            product_id: "1900"
            # Aliases for automatic covariance matrix
            renames: [lensing]
            # Speed in evaluations/second
            speed: 800
            # Nuisance parameters (do not change)
            nuisance: [calib]"""


def is_installed(**kwargs):
    kwargs["name"] = __name__.split(".")[-1]
    return is_installed_common(**kwargs)


def install(**kwargs):
    kwargs["name"] = __name__.split(".")[-1]
    return install_common(**kwargs)
