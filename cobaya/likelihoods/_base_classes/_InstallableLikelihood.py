r"""
.. module:: _InstallableLikelihood

:Synopsis: Prototype class adding class methods for simple installation of likelihood data.
:Author: Jesus Torrado and Antony Lewis

"""

# Global
import os
import logging

# Local
from cobaya.likelihood import Likelihood
from cobaya.likelihoods._base_classes._planck_clik_prototype import last_version_supp_data_and_covmats


class _InstallableLikelihood(Likelihood):
    install_options = {"github_repository": "CobayaSampler/planck_supp_data_and_covmats",
                       "github_release": last_version_supp_data_and_covmats}

    @classmethod
    def get_install_options(cls):
        return cls.install_options

    @classmethod
    def get_path(cls, path):
        opts = cls.get_install_options()
        repo = opts.get("github_repository", None)
        if repo:
            data_path = repo.split('/')[-1]
        else:
            data_path = opts.get("data_path", cls.__name__)
        return os.path.realpath(os.path.join(path, "data", data_path))

    @classmethod
    def is_installed(cls, **kwargs):
        if kwargs["data"]:
            path = cls.get_path(kwargs["path"])
            return cls.get_install_options() and os.path.exists(path) and len(os.listdir(path)) > 0
        return True

    @classmethod
    def install(cls, path=None, force=False, code=False, data=True, no_progress_bars=False):
        if not data:
            return True
        log = logging.getLogger(cls.get_module_name())
        opts = cls.get_install_options()
        repo = opts.get("github_repository", None)
        if repo:
            from cobaya.install import download_github_release
            log.info("Downloading %s data..." % repo)
            return download_github_release(
                os.path.join(path, "data"), repo, opts.get("github_release", "master"),
                no_progress_bars=no_progress_bars, logger=log)
        else:
            full_path = cls.get_path(path)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            if not data:
                return True
            filename = opts["download_url"]
            log.info("Downloading likelihood data file: %s...", filename)
            from cobaya.install import download_file
            if not download_file(filename, full_path, decompress=True, logger=log,
                                 no_progress_bars=no_progress_bars):
                return False
            log.info("Likelihood data downloaded and uncompressed correctly.")
            return True
