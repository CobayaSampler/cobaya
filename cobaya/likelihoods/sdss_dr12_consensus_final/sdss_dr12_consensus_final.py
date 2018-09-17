import os
import logging

from cobaya.likelihoods._bao_prototype import _bao_prototype
from cobaya.install import download_github_release


class sdss_dr12_consensus_final(_bao_prototype):
    pass


# Installation routines ##################################################################

# name of the data and covmats repo/folder
sdss_data_name = "bao_data"
sdss_data_version = "v1.1"


def get_path(path):
    return os.path.realpath(os.path.join(path, "data", sdss_data_name))


def is_installed(**kwargs):
    return os.path.exists(os.path.realpath(
        os.path.join(kwargs["path"], "data", sdss_data_name)))


def install(path=None, force=False, code=False, data=True, no_progress_bars=False):
    if not data:
        return True
    log = logging.getLogger(__name__.split(".")[-1])
    log.info("Downloading BAO data...")
    return download_github_release(os.path.join(path, "data"), sdss_data_name,
                                   sdss_data_version, no_progress_bars=no_progress_bars)
