from cobaya.likelihoods._planck_clik_prototype import _planck_clik_prototype
from cobaya.likelihoods._planck_clik_prototype import install as install_common
from cobaya.likelihoods._planck_clik_prototype import is_installed as is_installed_common


class planck_2015_plikHM_TTTEEE_unbinned(_planck_clik_prototype):

    defaults = r"""
        # Planck 2015 release: high-ell, unbinned CMB temperature+polarization likelihood
        # See https://wiki.cosmos.esa.int/planckpla2015/index.php/CMB_spectrum_%26_Likelihood_Code
        # NB: There is a typo in the wiki: the sigma of galf_TE_A_143_217 should be 0.18, not 0.018

        likelihood:
          planck_2015_plikHM_TTTEEE_unbinned:
            path:
            clik_file: plik_unbinned/plik_dx11dr2_HM_v18_TTTEEE_bin1.clik
            product_id: "1903"
            # Aliases for automatic covariance matrix
            renames: [plikHM_TTTEEE]
            # Speed in evaluations/second
            speed: 1.7
            # Nuisance parameters (do not change)
            nuisance: [calib, TT, TEEE]"""


def is_installed(**kwargs):
    kwargs["name"] = __name__.split(".")[-1]
    return is_installed_common(**kwargs)


def install(**kwargs):
    kwargs["name"] = __name__.split(".")[-1]
    return install_common(**kwargs)
