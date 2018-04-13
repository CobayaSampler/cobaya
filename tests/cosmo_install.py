if __name__ == "__main__":
    from cobaya.log import logger_setup
    logger_setup()
    from cobaya.conventions import _likelihood, _theory
    import sys
    info_install = {
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
            "sdss_dr12_consensus_final": None}}
    # Ignore Planck clik in Python 3 or GCC > 5
    from subprocess import Popen, PIPE
    process = Popen(["gcc", "-v"], stdout=PIPE, stderr=PIPE)
    out, err = process.communicate()
    prefix = "gcc version"
    try:
        version = [line for line in err.split("\n") if line.startswith(prefix)]
        version = version[0][len(prefix):].split()[0]
        gcc_major = int(version.split(".")[0])
    except:
        gcc_major = 0
    if sys.version_info.major >= 3 or gcc_major > 5:
        popliks = [lik for lik in info_install[_likelihood]
                   if lik.startswith("planck_2015") and not lik.endswith("cmblikes")]
        for lik in popliks:
            info_install[_likelihood].pop(lik)
    path = sys.argv[1]
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    from cobaya.install import install
    install(info_install, path=path, no_progress_bars=True)
