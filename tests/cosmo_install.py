if __name__ == "__main__":
    from cobaya.conventions import input_likelihood, input_theory
    info_install = {
    input_theory: {"camb": None, "classy": None},
    input_likelihood: {"planck_2015_lowl": None,
                       "planck_2015_plikHM_TT": None,
                       "planck_2015_lowTEB": None,
                       "planck_2015_plikHM_TTTEEE": None}}
    import sys
    path = sys.argv[1]
    import os
    if not os.path.exists(path):
        os.path.mkdirs(path)
    from cobaya.install import install
    install(info_install, path=path)
