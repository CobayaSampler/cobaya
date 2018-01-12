if __name__ == "__main__":
    from cobaya.log import logger_setup
    logger_setup()
    from cobaya.conventions import _likelihood, _theory
    import sys
    info_install = {
        _theory: {"camb": None, "classy": None}}
    if sys.version_info.major <3:
        info_install[_likelihood] = {"planck_2015_lowl": None,
                      "planck_2015_plikHM_TT": None,
                      "planck_2015_lowTEB": None,
                      "planck_2015_plikHM_TTTEEE": None,
                      "bicep_keck_2015": None}
    path = sys.argv[1]
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    from cobaya.install import install
    install(info_install, path=path, no_progress_bars=True)
