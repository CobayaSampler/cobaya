r"""
.. module:: InstallableLikelihood

:Synopsis: Prototype class with class methods for simple installation of likelihood data.
:Author: Jesus Torrado and Antony Lewis

"""

# Global
import os
from packaging import version

# Local
from cobaya.likelihood import Likelihood
from cobaya.typing import InfoDict
from cobaya.log import get_logger
from cobaya.install import _version_filename
from cobaya.component import ComponentNotInstalledError
from cobaya.tools import VersionCheckError, resolve_packages_path


class InstallableLikelihood(Likelihood):
    """
    Prototype class for likelihood with installation and installation-check methods.
    """

    install_options: InfoDict = {}

    def __init__(self, *args, **kwargs):
        # Ensure check for install and version errors
        # (e.g. may inherit from a class that inherits from this one, and not have them)
        if self.install_options:
            name = self.get_qualified_class_name()
            logger = get_logger(name)
            packages_path = kwargs.get("packages_path") or resolve_packages_path()
            old = False
            try:
                installed = self.is_installed(path=packages_path)
            except Exception as excpt:  # catches VersionCheckError and unexpected ones
                installed = False
                old = isinstance(excpt, VersionCheckError)
                logger.error(f"{type(excpt).__name__}: {excpt}")
            if not installed:
                not_or_old = ("is not up to date" if old
                              else "has not been correctly installed")
                raise ComponentNotInstalledError(
                    logger, (f"The data for this likelihood {not_or_old}. To install it, "
                             f"run `cobaya-install {name}{' --upgrade' if old else ''}`"))
        super().__init__(*args, **kwargs)

    @classmethod
    def get_install_options(cls):
        """
        Returns class variables containing e.g. a download url, a version tag...
        """
        return cls.install_options

    @classmethod
    def get_path(cls, path):
        """
        Returns the (real) path where the package will be installed,
        given a global installation ``path``.

        Repeated recursive calls on the same global path produce the same output,
        i.e. ``cls.get_path(cls.get_path(path)) = cls.get_path(path)``.
        """
        opts = cls.get_install_options()
        repo = opts.get("directory", opts.get("github_repository", None))
        if repo:
            data_path = repo.split('/')[-1]
        else:
            data_path = opts.get("data_path", cls.__name__)
        install_path = os.path.realpath(os.path.join(path, "data", data_path))
        # Idempotent: return if input == proposed output
        if path.rstrip(os.sep).endswith(data_path.rstrip(os.sep)):
            return path
        return install_path

    @classmethod
    def is_installed(cls, **kwargs):
        """
        Performs an installation check and returns ``True`` if successful, ``False`` if
        not, or raises :class:`tools.VersionCheckError` if there is an obsolete
        installation.
        """
        if kwargs.get("data", True):
            path = kwargs["path"]
            path = cls.get_path(path)  # ensure full install path passed
            opts = cls.get_install_options()
            if not opts:
                return True
            elif not (os.path.exists(path) and len(os.listdir(path)) > 0):
                log = get_logger(cls.get_qualified_class_name())
                log.error("The given installation path does not exist: '%s'", path)
                return False
            elif opts.get("github_release"):
                try:
                    with open(os.path.join(path, _version_filename), "r") as f:
                        installed_version = version.parse(f.readlines()[0])
                except FileNotFoundError:  # old install: no version file
                    raise VersionCheckError("Could not read current version.")
                min_version = version.parse(opts.get("github_release"))
                if installed_version < min_version:
                    raise VersionCheckError(
                        f"Installed version ({installed_version}) "
                        f"older than minimum required one ({min_version}).")
        return True

    @classmethod
    def install(cls, path=None, data=True, no_progress_bars=False, **_kwargs):
        """
        Installs the necessary data packages for this likelihood.
        """
        if not data:
            return True
        log = get_logger(cls.get_qualified_class_name())
        opts = cls.get_install_options()
        if not opts:
            log.info("No install options. Nothing to do.")
            return True
        repo = opts.get("github_repository", None)
        if repo:
            from cobaya.install import download_github_release
            log.info("Downloading %s data..." % repo)
            return download_github_release(
                os.path.join(path, "data"), repo, opts.get("github_release", "master"),
                asset=opts.get("asset", None), directory=opts.get("directory", None),
                no_progress_bars=no_progress_bars, logger=log)
        else:
            full_path = cls.get_path(path)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            if not data:
                return True
            url = opts["download_url"]
            log.info("Downloading likelihood data file: %s...", url)
            from cobaya.install import download_file
            if not download_file(url, full_path, decompress=True, logger=log,
                                 no_progress_bars=no_progress_bars):
                return False
            log.info("Likelihood data downloaded and uncompressed correctly.")
            return True
