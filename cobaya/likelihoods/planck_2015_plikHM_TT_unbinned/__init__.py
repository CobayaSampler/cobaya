from cobaya.likelihoods._planck_clik_prototype import _planck_clik_prototype
from cobaya.likelihoods._planck_clik_prototype import install as install_common
from cobaya.likelihoods._planck_clik_prototype import is_installed as is_installed_common


class planck_2015_plikHM_TT_unbinned(_planck_clik_prototype):

    defaults = """
        # Planck 2015 release: high-ell, unbinned CMB temperature only likelihood
        # See https://wiki.cosmos.esa.int/planckpla2015/index.php/CMB_spectrum_%26_Likelihood_Code

        likelihood:
          planck_2015_plikHM_TT_unbinned:
            path:
            clik_file: plik_unbinned/plik_dx11dr2_HM_v18_TT_bin1.clik
            product_id: "1903"
            # Aliases for automatic covariance matrix
            renames: [plikHM_TT]
            # Speed in evaluations/second
            speed: 8
            # Nuisance parameters (do not change)
            nuisance: [calib, TT]"""


def is_installed(**kwargs):
    kwargs["name"] = __name__.split(".")[-1]
    return is_installed_common(**kwargs)


def install(**kwargs):
    kwargs["name"] = __name__.split(".")[-1]
    return install_common(**kwargs)
