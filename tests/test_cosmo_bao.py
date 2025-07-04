from copy import deepcopy

from cobaya.component import get_component_class

from .common_cosmo import body_of_test
from .test_cosmo_planck_2015 import derived_lowTEB_highTTTEEE, params_lowTEB_highTTTEEE


# Tests both the bao.generic class, and class renaming for multiple instances
def test_generic_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr12_consensus_bao"
    like_rename = "my_bao"
    chi2_generic = deepcopy(chi2_sdss_dr12_consensus_bao)
    chi2_generic[like_rename] = chi2_generic.pop(like)
    likelihood_defaults = get_component_class(like).get_defaults()
    likelihood_defaults.pop("path")
    likelihood_defaults["class"] = "bao.generic"
    info_likelihood = {like_rename: likelihood_defaults}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_generic,
        skip_not_installed=skip_not_installed,
    )


# Test generic bao class with different kind of observables
def test_generic_mixed_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr12_consensus_bao"
    like_rename = "bao_mixed_observables"
    chi2_generic = deepcopy(chi2_sdss_dr12_consensus_bao)
    chi2_generic.pop(like)
    chi2_generic[like_rename] = 5.0
    likelihood_defaults = get_component_class(like).get_defaults()
    likelihood_defaults.pop("path")
    likelihood_defaults["class"] = "bao.generic"
    likelihood_defaults["measurements_file"] = (
        "bao_data/test_bao_mixed_observables_mean.txt"
    )
    likelihood_defaults["cov_file"] = "bao_data/test_bao_mixed_observables_cov.txt"
    likelihood_defaults["rs_fid"] = 1.0
    info_likelihood = {like_rename: likelihood_defaults}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_generic,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_baoplus_lrg_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_baoplus_lrg"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr16_baoplus_lrg,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_baoplus_lrg_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_baoplus_lrg"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr16_baoplus_lrg)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_bao_lrg_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_lrg_bao_dmdh"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr16_bao_lrg,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_bao_lrg_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_lrg_bao_dmdh"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr16_bao_lrg)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr12_consensus_bao_lrg_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr12_lrg_bao_dmdh"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr12_bao_lrg,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr12_consensus_bao_lrg_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr12_lrg_bao_dmdh"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr12_bao_lrg)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_baoplus_elg_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_baoplus_elg"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr16_baoplus_elg,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_baoplus_elg_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_baoplus_elg"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr16_baoplus_elg)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_bao_elg_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_bao_elg"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr16_bao_elg,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_bao_elg_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_bao_elg"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr16_bao_elg)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_baoplus_qso_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_baoplus_qso"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr16_baoplus_qso,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_baoplus_qso_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_baoplus_qso"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr16_baoplus_qso)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_bao_qso_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_qso_bao_dmdh"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr16_bao_qso,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_bao_qso_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_qso_bao_dmdh"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr16_bao_qso)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_bao_lyauto_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_baoplus_lyauto"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr16_baoplus_lyauto,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_bao_lyauto_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_baoplus_lyauto"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr16_baoplus_lyauto)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_bao_lyxqso_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_baoplus_lyxqso"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr16_baoplus_lyxqso,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr16_consensus_bao_lyxqso_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr16_baoplus_lyxqso"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr16_baoplus_lyxqso)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr12_consensus_bao_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr12_consensus_bao"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr12_consensus_bao,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr12_consensus_bao_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr12_consensus_bao"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr12_consensus_bao)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr12_consensus_full_shape_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr12_consensus_full_shape"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    # Check sigma8(z=0): it used to fail bc it was computed internally by the theory code
    # for different redshifts
    derived = {"sigma8": derived_lowTEB_highTTTEEE["sigma8"]}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr12_consensus_full_shape,
        best_fit_derived=derived,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr12_consensus_full_shape_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr12_consensus_full_shape"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr12_consensus_full_shape)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    # Check sigma8(z=0): it used to fail bc it was computed internally by the theory code
    # for different redshifts
    derived = {"sigma8": derived_lowTEB_highTTTEEE["sigma8"]}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        best_fit_derived=derived,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr12_consensus_final_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr12_consensus_final"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr12_consensus_final,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr12_consensus_final_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr12_consensus_final"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr12_consensus_final)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sixdf_2011_bao_camb(packages_path, skip_not_installed):
    like = "bao.sixdf_2011_bao"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sixdf_2011_bao,
        skip_not_installed=skip_not_installed,
    )


def test_sixdf_2011_bao_classy(packages_path, skip_not_installed):
    like = "bao.sixdf_2011_bao"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sixdf_2011_bao)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr7_mgs_camb(packages_path, skip_not_installed):
    like = "bao.sdss_dr7_mgs"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_sdss_dr7_mgs,
        skip_not_installed=skip_not_installed,
    )


def test_sdss_dr7_mgs_classy(packages_path, skip_not_installed):
    like = "bao.sdss_dr7_mgs"
    info_likelihood = {like: {}}
    info_theory = {"classy": None}
    chi2 = deepcopy(chi2_sdss_dr7_mgs)
    chi2["tolerance"] += chi2.pop("classy_extra_tolerance", 0)
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2,
        skip_not_installed=skip_not_installed,
    )


def test_DESI_y1_camb(packages_path, skip_not_installed):
    like = "bao.desi_2024_bao_all"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_DESI_y1_bao,
        skip_not_installed=skip_not_installed,
    )


def test_DESI_dr2_camb(packages_path, skip_not_installed):
    like = "bao.desi_dr2"
    info_likelihood = {like: {}}
    info_theory = {"camb": None}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_DESI_dr2_bao,
        skip_not_installed=skip_not_installed,
    )
    info_likelihood = {"bao.desi_dr2.desi_bao_elg2": {}}
    body_of_test(
        packages_path,
        best_fit,
        info_likelihood,
        info_theory,
        chi2_DESI_dr2_elg2,
        skip_not_installed=skip_not_installed,
    )


# BEST FIT AND REFERENCE VALUES ##########################################################

best_fit = deepcopy(params_lowTEB_highTTTEEE)

chi2_sdss_dr16_baoplus_elg = {"bao.sdss_dr16_baoplus_elg": 3.24, "tolerance": 0.06}
chi2_sdss_dr16_bao_elg = {"bao.sdss_dr16_bao_elg": 0.37, "tolerance": 0.06}
chi2_sdss_dr16_baoplus_lrg = {"bao.sdss_dr16_baoplus_lrg": 5.96, "tolerance": 0.04}
chi2_sdss_dr16_bao_lrg = {"bao.sdss_dr16_lrg_bao_dmdh": 3.29, "tolerance": 0.04}
chi2_sdss_dr12_bao_lrg = {"bao.sdss_dr12_lrg_bao_dmdh": 2.95, "tolerance": 0.04}
chi2_sdss_dr16_baoplus_qso = {
    "bao.sdss_dr16_baoplus_qso": 8.78,
    "tolerance": 0.04,
    "classy_extra_tolerance": 0.21,
}
chi2_sdss_dr16_bao_qso = {
    "bao.sdss_dr16_qso_bao_dmdh": 0.54,
    "tolerance": 0.04,
    "classy_extra_tolerance": 0.21,
}
chi2_sdss_dr16_baoplus_lyauto = {"bao.sdss_dr16_baoplus_lyauto": 1.74, "tolerance": 0.04}
chi2_sdss_dr16_baoplus_lyxqso = {"bao.sdss_dr16_baoplus_lyxqso": 3.24, "tolerance": 0.04}
chi2_sdss_dr12_consensus_bao = {"bao.sdss_dr12_consensus_bao": 5.687, "tolerance": 0.04}
chi2_sdss_dr12_consensus_full_shape = {
    "bao.sdss_dr12_consensus_full_shape": 8.154,
    "tolerance": 0.02,
    "classy_extra_tolerance": 0.075,
}
chi2_sdss_dr12_consensus_final = {
    "bao.sdss_dr12_consensus_final": 8.051,
    "tolerance": 0.03,
    "classy_extra_tolerance": 0.03,
}
chi2_sixdf_2011_bao = {"bao.sixdf_2011_bao": 0.088, "tolerance": 0.02}
chi2_sdss_dr7_mgs = {"bao.sdss_dr7_mgs": 0.92689, "tolerance": 0.02}
chi2_DESI_y1_bao = {"bao.desi_2024_bao_all": 21.37, "tolerance": 0.02}
chi2_DESI_dr2_bao = {"bao.desi_dr2": 30.51, "tolerance": 0.02}
chi2_DESI_dr2_elg2 = {"bao.desi_dr2.desi_bao_elg2": 2.24, "tolerance": 0.01}
