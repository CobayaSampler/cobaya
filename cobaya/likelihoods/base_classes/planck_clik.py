r"""
.. module:: planck_clik

:Synopsis: Definition of the clik-based likelihoods using clipy
:Author: Jesus Torrado (initially based on MontePython's version
         by Julien Lesgourgues and Benjamin Audren)
         Updated to use clipy by Antony Lewis and Jesus Torrado

"""

import os

import numpy as np

from cobaya.component import ComponentNotInstalledError, load_external_module
from cobaya.input import get_default_info
from cobaya.install import download_file, download_github_release, pip_install
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError, get_logger
from cobaya.tools import VersionCheckError, are_different_params_lists

pla_url_prefix = r"https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.COSMOLOGY_OID="

# Clipy installation
clipy_repo_name = "benabed/clipy"
clipy_repo_min_version = "0.15"

last_version_supp_data_and_covmats = "v2.1"


class PlanckClik(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "CMB"

    path: str
    clik_file: str

    def initialize(self):
        # Disable JAX to avoid dependency issues
        os.environ["CLIPY_NOJAX"] = "1"
        msg_to_install = "run `cobaya-install planck_2018_highl_plik.TTTEEE`"
        try:
            install_path = (lambda p: self.get_code_path(p) if p else None)(
                self.packages_path
            )
            # Load clipy instead of clik
            clipy = load_clipy(
                path=self.path,
                install_path=install_path,
                logger=self.log,
                not_installed_level="debug",
            )
        except ComponentNotInstalledError as excpt:
            raise ComponentNotInstalledError(
                self.log,
                f"Could not find clipy: {excpt}. To install it, {msg_to_install}",
            ) from excpt
        except VersionCheckError as excpt:
            raise VersionCheckError(
                self.log,
                f"{excpt}. To install a new version, {msg_to_install} with `--upgrade`.",
            ) from excpt
        # Loading the likelihood data
        data_path = get_data_path(self.__class__.get_qualified_class_name())
        if not os.path.isabs(self.clik_file):
            self.path_data = getattr(
                self,
                "path_data",
                os.path.join(self.path or self.packages_path, "data", data_path),
            )
            self.clik_file = os.path.join(self.path_data, self.clik_file)
        # Prepare clipy commands
        if isinstance(self.commands, str):
            self.commands = [self.commands]
        try:
            self.clik_likelihood = clipy.clik(self.clik_file, crop=self.commands or [])
        except clipy.clik_emul_error as excpt:
            # Is it that the file was not found?
            if not os.path.exists(self.clik_file):
                raise ComponentNotInstalledError(
                    self.log,
                    "The .clik file was not found where specified in the 'clik_file' "
                    "field of the settings of this likelihood. Install this likelihood "
                    f"with 'cobaya-install {self.get_qualified_class_name()}'. If this "
                    "error persists, maybe the 'path' given is not correct? The full path"
                    " where the .clik file was searched for is '{self.clik_file}'",
                ) from excpt
            # Else: unknown clipy error
            raise LoggedError(
                self.log,
                f"An unexpected managed error occurred in clipy: {excpt}",
            ) from excpt
        except Exception as excpt:
            if self.commands:  # check if bad command
                raise LoggedError(
                    self.log,
                    f"An unmanaged error occurred in clipy: {excpt}. This may have been "
                    "caused by a worngly-formatted 'command'. Please check your command "
                    "syntax, or disable and try again to check that clipy is working as "
                    f"expected. The list of commands passed were: {self.commands}",
                ) from excpt
            else:  # unknown clippy error
                raise LoggedError(
                    self.log,
                    f"An unmanaged error occurred in clipy: {excpt}. Please report it as "
                    "a GitHub issue in the Cobaya repo.",
                ) from excpt
        lmaxs = self.clik_likelihood.lmax
        cls_sorted = ["tt", "ee", "bb", "te", "tb", "eb"]
        if len(lmaxs) > 6:  # lensing likelihood!
            cls_sorted = ["pp"] + cls_sorted
        self.requested_cls_lmax = {
            cl: lmax for cl, lmax in zip(cls_sorted, lmaxs) if lmax != -1
        }
        self.expected_params = list(self.clik_likelihood.extra_parameter_names)
        # Placeholder for vector passed to clipy
        self.vector = np.zeros(
            sum(list(self.requested_cls_lmax.values()))
            + len(self.requested_cls_lmax)  # account for ell=0
            + len(self.expected_params)
        )

    def initialize_with_params(self):
        # Check that the parameters are the right ones
        differences = are_different_params_lists(
            self.input_params, self.expected_params, name_A="given", name_B="expected"
        )
        if differences:
            raise LoggedError(
                self.log,
                "Configuration error in parameters: %r. "
                "If this has happened without you fiddling with the defaults, "
                "please open an issue in GitHub.",
                differences,
            )

    def get_requirements(self):
        # State requisites to the theory code
        return {"Cl": self.requested_cls_lmax}

    def logp(self, **params_values):
        # Get Cl's from the theory code
        cl = self.provider.get_Cl(units="FIRASmuK2")
        return self.log_likelihood(cl, **params_values)

    def log_likelihood(self, cl, **params_values):
        # Fill with Cl's
        self.vector[: -len(self.expected_params)] = np.concatenate(
            [
                cl[spec][: 1 + lmax] if spec not in ["tb", "eb"] else np.zeros(1 + lmax)
                for spec, lmax in self.requested_cls_lmax.items()
            ]
        )
        # check for nan's: may produce issues in clipy
        # dot product is apparently the fastest way in threading-enabled numpy
        if np.isnan(np.dot(self.vector, self.vector)):
            return -np.inf
        # Fill with likelihood parameters
        self.vector[-len(self.expected_params) :] = [
            params_values[p] for p in self.expected_params
        ]
        loglike = self.clik_likelihood(self.vector)
        # "zero" of clipy, and sometimes nan's returned
        if loglike <= -1e30 or np.isnan(loglike):
            loglike = -np.inf
        return loglike

    @classmethod
    def get_code_path(cls, path):
        return os.path.realpath(os.path.join(path, "code", common_path))

    @classmethod
    def is_installed(cls, reload=False, **kwargs):
        code_path = common_path
        data_path = get_data_path(cls.get_qualified_class_name())
        result = True
        if kwargs.get("code", True):
            # Check if clipy is installed in the packages path specifically
            # Don't fall back to global installation during installation check
            packages_clipy_path = os.path.realpath(
                os.path.join(kwargs["path"], "code", code_path)
            )
            result &= is_installed_clipy(packages_clipy_path, reload=reload)
        if kwargs.get("data", True):
            # NB: will never raise VersionCheckerror, since version number is in the path
            _, filename = get_product_id_and_clik_file(cls.get_qualified_class_name())
            result &= os.path.exists(
                os.path.realpath(
                    os.path.join(kwargs["path"], "data", data_path, filename)
                )
            )
            # Check for additional data and covmats -- can raise VersionCheckerror
            from cobaya.likelihoods.planck_2018_lensing import native

            result &= native.is_installed(**kwargs)
        return result

    @classmethod
    def install(
        cls,
        path=None,
        force=False,
        code=True,
        data=True,
        no_progress_bars=False,
        **_kwargs,
    ):
        name = cls.get_qualified_class_name()
        logger = get_logger(name)
        path_names = {"code": common_path, "data": get_data_path(name)}
        global _clipy_install_failed
        if _clipy_install_failed:
            logger.info("Previous clipy install failed, skipping")
            return False
        # Create common folders: all planck likelihoods share install
        # folder for code and data
        paths = {}
        for s in ("code", "data"):
            if eval(s):
                paths[s] = os.path.realpath(os.path.join(path, s, path_names[s]))
                if not os.path.exists(paths[s]):
                    os.makedirs(paths[s])
        success = True
        # Install clipy to packages path (don't rely on global installation)
        if code and (not is_installed_clipy(paths["code"], logger=logger) or force):
            logger.info("Installing clipy.")
            success *= install_clipy(
                paths["code"], no_progress_bars=no_progress_bars, logger=logger
            )
            if not success:
                logger.warning("clipy installation failed!")
                _clipy_install_failed = True
        if data:
            # 2nd test, in case the code wasn't there but the data is:
            if force or not cls.is_installed(path=path, code=False, data=True):
                logger.info("Downloading the likelihood data.")
                product_id, _ = get_product_id_and_clik_file(name)
                # Download and decompress the particular likelihood
                url = pla_url_prefix + product_id
                # Helper for the progress bars: some known product download sizes
                # (no actual effect if missing or wrong!)
                size = {
                    "1900": 314153370,
                    "1903": 4509715660,
                    "151902": 60293120,
                    "151905": 5476083302,
                    "151903": 8160437862,
                }.get(product_id)
                if not download_file(
                    url,
                    paths["data"],
                    size=size,
                    logger=logger,
                    no_progress_bars=no_progress_bars,
                ):
                    logger.error("Not possible to download this likelihood.")
                    success = False
                # Additional data and covmats, stored in same repo as the
                # 2018 python lensing likelihood
                from cobaya.likelihoods.planck_2018_lensing import native

                if not native.is_installed(data=True, path=path):
                    success *= native.install(
                        path=path,
                        force=force,
                        code=code,
                        data=data,
                        no_progress_bars=no_progress_bars,
                    )
        return success


# Installation routines ##################################################################

# path to be shared by all Planck likelihoods
common_path = "planck"

# Don't try again to install clipy if it failed for a previous likelihood
_clipy_install_failed = False


def get_data_path(name):
    return common_path + "_%s" % get_release(name)


def get_release(name):
    return next(re for re in ["2015", "2018"] if re in name)


def get_clipy_import_path(path):
    """
    Starting from the installation folder, returns the subdirectory from which the
    ``clipy`` module must be imported.

    Raises ``FileNotFoundError`` if no clipy install was found.
    """
    clipy_path = os.path.join(path, "clipy")
    if not os.path.exists(clipy_path):
        raise FileNotFoundError(f"clipy installation not found at {clipy_path}")
    # Check if it has the proper structure
    init_file = os.path.join(clipy_path, "clipy", "__init__.py")
    if not os.path.exists(init_file):
        raise FileNotFoundError(f"clipy package structure not found at {clipy_path}")
    return clipy_path


def load_clipy(
    path=None,
    install_path=None,
    logger=None,
    not_installed_level="error",
    reload=False,
    default_global=False,
):
    """
    Load clipy module and check that it's the correct one.

    Returns
    -------
    clipy: module
        The imported clipy module

    Raises
    ------
    ComponentNotInstalledError
        If clipy has not been installed
    VersionCheckError
        If clipy is found, but with a smaller version number.
    """
    if logger is None:
        logger = get_logger("clipy")
    clipy = load_external_module(
        module_name="clipy",
        path=path,
        install_path=install_path,
        get_import_path=get_clipy_import_path if install_path else None,
        min_version=clipy_repo_min_version,
        logger=logger,
        not_installed_level=not_installed_level,
        reload=reload,
        default_global=default_global,
    )
    # Check that it has the expected clipy interface
    if not hasattr(clipy, "clik"):
        raise ComponentNotInstalledError(
            logger,
            "Loaded wrong clipy: you may have pip-installed 'clipy' instead of "
            "'clipy-like'.",
        )
    return clipy


def is_installed_clipy(path=None, reload=False, logger=None):
    """
    Check if clipy is installed and working in the specified path. It should not raise any
    exception.

    Parameters
    ----------
    path: str, optional
        Path where the clipy installation is tested, to which ``clipy`` will be appended
        before testing. If not defined, a test for a global python installation will be
        performed instead (``default_global=True`` will be passed to the module loader).
    reload: bool
        Whether to attemp to reload the ``clipy`` module before checking.
    logger: logging.Logger, optional
        Initialized logger. If note passed, one named ``clipy`` will be created.

    Returns
    -------
    bool
       ``True`` if the installation was successful, and ``False`` otherwise.
    """
    if logger is None:
        logger = get_logger("clipy")
    try:
        # If path is specified, don't fall back to global import
        default_global = path is None
        load_clipy(
            path=os.path.join(path, "clipy"),
            logger=logger,
            not_installed_level="debug",
            reload=reload,
            default_global=default_global,
        )
        return True
    except (ComponentNotInstalledError, VersionCheckError):
        return False


def install_clipy(path, logger=None, no_progress_bars=False):
    """
    Install clipy from GitHub repository to the specified path.

    Parameters
    ----------
    path: str
        Path where clipy will be downloaded into, to which ``clipy`` will be appended.

    logger: logging.Logger, optional
        Initialized logger. If note passed, one named ``clipy`` will be created.

    no_progress_bars: bool
        Whether to show download/install progress bars.

    Returns
    -------
    bool
       ``True`` if the installation was successful, and ``False`` otherwise.
    """
    if logger is None:
        logger = get_logger("clipy")
    # Install pre-requisites
    logger.info("Installing pre-requisites...")
    for req in ("astropy",):
        exit_status = pip_install(req)
        if exit_status:
            logger.error("Failed installing '%s'.", req)
            return False
    logger.info("Installing clipy from GitHub repository...")
    success = download_github_release(
        path,
        clipy_repo_name,
        "clipy_" + clipy_repo_min_version,  # TODO: check if "clipy_" still in release
        no_progress_bars=no_progress_bars,
        logger=logger,
    )
    if not success:
        logger.error("Could not download clipy from GitHub.")
        return False
    logger.info("clipy installation finished!")
    return True


def get_product_id_and_clik_file(name):
    """Gets the PLA product info from the defaults file."""
    defaults = get_default_info(name, "likelihood")
    return defaults.get("product_id"), defaults.get("clik_file")


class Planck2018Clik(PlanckClik):
    bibtex_file = "planck2018.bibtex"
