# Global
from __future__ import division
import os

# Local
from cobaya.likelihood import Likelihood

# Logger
import logging
log = logging.getLogger(__name__)


class bicep_keck_2015(Likelihood):
    pass

# Installation routines ##################################################################

# path to be shared by all Planck likelihoods
common_path = "bicep_keck"


def get_path(path):
    return os.path.realpath(os.path.join(path, "../data", common_path))


def is_installed(**kwargs):
    if kwargs["data"]:
        if not os.path.exists(os.path.join(get_path(kwargs["path"]), "BK14_cosmomc")):
            return False
    return True


def install(path=None, name=None, force=False, code=False, data=True):
    # Create common folders: all planck likelihoods share install folder for code and data
    full_path = get_path(path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    if not data:
        return True
    log.info("Downloading likelihood data...")
    try:
        from wget import download, bar_thermometer
        wget_kwargs = {"out": full_path, "bar": bar_thermometer}
        filename = download(r"http://bicepkeck.org/BK14_datarelease/BK14_cosmomc.tgz",
                            **wget_kwargs)
        print ""  # force newline after wget
    except:
        print ""  # force newline after wget
        log.error("Error downloading!")
        return False
    import tarfile
    extension = os.path.splitext(filename)[-1][1:]
    if extension == "tgz":
        extension = "gz"
    tar = tarfile.open(filename, "r:"+extension)
    try:
        tar.extractall(full_path)
        tar.close()
        os.remove(filename)
        log.info("Likelihood data downloaded and uncompressed correctly.")
        return True
    except:
        log.error("Error decompressing downloaded file! Corrupt file?)")
        return False
