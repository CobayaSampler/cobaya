from __future__ import division, absolute_import, print_function
import os


def process_modules_path(modules):
    if not modules:
        if os.path.exists(os.path.join(os.getcwd(), '..', 'modules')):
            modules = os.path.join('..', 'modules')
    assert modules, "I need a modules folder!"
    return modules if os.path.isabs(modules) else os.path.join(os.getcwd(), modules)


if __name__ == "__main__":
    from cobaya.log import logger_setup

    logger_setup()
    from cobaya.conventions import _likelihood, _theory, _sampler
    import sys

    info_install = {
        _sampler: {"polychord": None},
        _theory: {"camb": None, "classy": None},
        _likelihood: {
            "planck_2015_lowl": None,
            "planck_2015_plikHM_TT": None,
            "planck_2015_lowTEB": None,
            "planck_2015_plikHM_TTTEEE": None,
            "planck_2015_lensing": None,
            "planck_2015_lensing_cmblikes": None,
            "bicep_keck_2015": None,
            "sn_pantheon": None,
            "sn_jla": None,
            "sn_jla_lite": None,
            "sdss_dr12_consensus_bao": None,
            "sdss_dr12_consensus_full_shape": None,
            "sdss_dr12_consensus_final": None,
            "des_y1_joint": None}}
    path = sys.argv[1]

    if not os.path.exists(path):
        os.makedirs(path)
    from cobaya.install import install

    install(info_install, path=path, no_progress_bars=True)
