r"""
.. module:: planck_clik

:Synopsis: Definition of the clik-based likelihoods using clipy
:Author: Jesus Torrado (initially based on MontePython's version
         by Julien Lesgourgues and Benjamin Audren)
         Updated to use clipy by Antony Lewis

"""

import os
import re
import sys

import numpy as np
from packaging import version

from cobaya.component import ComponentNotInstalledError, load_external_module
from cobaya.input import get_default_info
from cobaya.install import download_file, pip_install
from cobaya.component import load_external_module
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError, get_logger
from cobaya.tools import (
    VersionCheckError,
    are_different_params_lists,
    create_banner,
)

_deprecation_msg_2015 = create_banner("""
The likelihoods from the Planck 2015 data release have been superseded
by the 2018 ones, and will eventually be deprecated.
""")

pla_url_prefix = r"https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.COSMOLOGY_OID="
clipy_url = "https://github.com/benabed/clipy/archive/refs/heads/main.zip"

last_version_supp_data_and_covmats = "v2.1"
min_version_clipy = "0.11"


class PlanckClik(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "CMB"

    path: str
    clik_file: str

    def initialize(self):
        if "2015" in self.get_name():
            for line in _deprecation_msg_2015.split("\n"):
                self.log.warning(line)
        try:
            install_path = (lambda p: self.get_code_path(p) if p else None)(
                self.packages_path
            )
            # Load clipy instead of clik
            clik = load_clipy(
                path=self.path,
                install_path=install_path,
                logger=self.log,
                not_installed_level="debug"
            )
        except ComponentNotInstalledError as excpt:
            raise ComponentNotInstalledError(
                self.log,
                (
                    f"Could not find clipy: {excpt}. To install it, "
                    f"run `cobaya-install planck_2018_highl_plik.TTTEEE`"
                ),
            )
        # Loading the likelihood data
        data_path = get_data_path(self.__class__.get_qualified_class_name())
        if not os.path.isabs(self.clik_file):
            self.path_data = getattr(
                self,
                "path_data",
                os.path.join(self.path or self.packages_path, "data", data_path),
            )
            self.clik_file = os.path.join(self.path_data, self.clik_file)
        # clipy handles both lensing and non-lensing likelihoods with single constructor
        try:
            # Disable JAX to avoid dependency issues
            os.environ["CLIPY_NOJAX"] = "1"
            self.clik = clik.clik(self.clik_file)
        except Exception as e:
            # Is it that the file was not found?
            if not os.path.exists(self.clik_file):
                raise ComponentNotInstalledError(
                    self.log,
                    "The .clik file was not found where specified in the "
                    "'clik_file' field of the settings of this likelihood. "
                    "Maybe the 'path' given is not correct? The full path where"
                    " the .clik file was searched for is '%s'",
                    self.clik_file,
                )
            # Else: unknown clipy error
            self.log.error(
                "An unexpected error occurred in clipy (possibly related to "
                "multiple simultaneous initialization, or simultaneous "
                "initialization of incompatible likelihoods; e.g. polarised "
                "vs non-polarised 'lite' likelihoods. See error info below:"
            )
            raise
        self.l_maxs = self.clik.get_lmax()
        # calculate requirements here so class can also be separately instantiated
        requested_cls = ["tt", "ee", "bb", "te", "tb", "eb"]
        # clipy automatically handles lensing detection, but we need to check the lmax values
        has_cl = [lmax != -1 for lmax in self.l_maxs]
        # Check if this is a lensing likelihood by examining the structure
        if len(self.l_maxs) > 6 and self.l_maxs[0] != -1:
            # First element is pp for lensing likelihoods
            self.lensing = True
            requested_cls = ["pp"] + requested_cls
        else:
            self.lensing = False
            # For non-lensing, use get_has_cl if available
            if hasattr(self.clik, 'get_has_cl'):
                has_cl = self.clik.get_has_cl()
        self.requested_cls = [cl for cl, i in zip(requested_cls, has_cl) if int(i)]
        self.l_maxs_cls = [lmax for lmax, i in zip(self.l_maxs, has_cl) if int(i)]
        self.expected_params = list(self.clik.extra_parameter_names)
        # Placeholder for vector passed to clipy
        length = len(self.l_maxs) if self.lensing else len(has_cl)
        self.vector = np.zeros(np.sum(self.l_maxs) + length + len(self.expected_params))

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
        return {"Cl": dict(zip(self.requested_cls, self.l_maxs_cls))}

    def logp(self, **params_values):
        # get Cl's from the theory code
        cl = self.provider.get_Cl(units="FIRASmuK2")
        return self.log_likelihood(cl, **params_values)

    def log_likelihood(self, cl, **params_values):
        # fill with Cl's
        self.vector[: -len(self.expected_params)] = np.concatenate(
            [
                (
                    cl[spectrum][: 1 + lmax]
                    if spectrum not in ["tb", "eb"]
                    else np.zeros(1 + lmax)
                )
                for spectrum, lmax in zip(self.requested_cls, self.l_maxs_cls)
            ]
        )
        # check for nan's: may produce issues in clipy
        # dot product is apparently the fastest way in threading-enabled numpy
        if np.isnan(np.dot(self.vector, self.vector)):
            return -np.inf
        # fill with likelihood parameters
        self.vector[-len(self.expected_params) :] = [
            params_values[p] for p in self.expected_params
        ]
        # clipy returns a scalar, not an array like clik
        loglike = self.clik(self.vector)
        # Convert to Python float
        loglike = float(loglike)
        # "zero" of clipy, and sometimes nan's returned
        if np.allclose(loglike, -1e30) or np.isnan(loglike):
            loglike = -np.inf
        return loglike

    def close(self):
        del self.clik  # Clean up clipy object

    @classmethod
    def get_code_path(cls, path):
        return os.path.realpath(os.path.join(path, "code", common_path))

    @classmethod
    def is_compatible(cls):
        import platform

        return platform.system() != "Windows"

    @classmethod
    def is_installed(cls, reload=False, **kwargs):
        code_path = common_path
        data_path = get_data_path(cls.get_qualified_class_name())
        result = True
        if kwargs.get("code", True):
            # Check if clipy is installed
            result &= is_installed_clipy(
                os.path.realpath(os.path.join(kwargs["path"], "code", code_path)),
                reload=reload,
            )
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
        log = get_logger(name)
        path_names = {"code": common_path, "data": get_data_path(name)}
        import platform

        if platform.system() == "Windows":
            log.error("Not compatible with Windows.")
            return False
        global _clipy_install_failed
        if _clipy_install_failed:
            log.info("Previous clipy install failed, skipping")
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
        # Install clipy
        if code and (not is_installed_clipy(paths["code"]) or force):
            log.info("Installing clipy.")
            success *= install_clipy(paths["code"], no_progress_bars=no_progress_bars)
            if not success:
                log.warning("clipy installation failed!")
                _clipy_install_failed = True
        if data:
            # 2nd test, in case the code wasn't there but the data is:
            if force or not cls.is_installed(path=path, code=False, data=True):
                log.info("Downloading the likelihood data.")
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
                from cobaya.install import download_file
                if not download_file(
                    url,
                    paths["data"],
                    size=size,
                    logger=log,
                    no_progress_bars=no_progress_bars,
                ):
                    log.error("Not possible to download this likelihood.")
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


def load_clipy(path=None, install_path=None, logger=None, not_installed_level="error"):
    """
    Load clipy module and check that it's the correct one.
    """
    try:
        clipy = load_external_module(
            module_name="clipy", path=path, install_path=install_path,
            logger=logger, not_installed_level=not_installed_level
        )
        # Check that it has the expected clipy interface
        if not hasattr(clipy, "clik"):
            raise ComponentNotInstalledError(
                logger, "Loaded wrong clipy: missing clik class"
            )
        # Check version if possible
        if hasattr(clipy, "__version__"):
            installed_version = version.parse(clipy.__version__)
            if installed_version < version.parse(min_version_clipy):
                raise VersionCheckError(
                    f"Installed version of clipy ({installed_version}) "
                    f"older than minimum required one ({min_version_clipy})."
                )
        return clipy
    except ImportError:
        raise ComponentNotInstalledError(
            logger, "clipy not installed. Install with: cobaya-install planck_2018_highl_plik.TTTEEE"
        )


def is_installed_clipy(path=None, reload=False):
    """
    Check if clipy is installed and working.
    """
    try:
        if path:
            # Check if clipy is installed in the specified path
            import sys
            clipy_path = os.path.join(path, "clipy")
            if os.path.exists(clipy_path) and clipy_path not in sys.path:
                sys.path.insert(0, clipy_path)
        load_clipy(logger=get_logger("clipy"), not_installed_level="debug")
        return True
    except (ComponentNotInstalledError, VersionCheckError):
        return False


def install_clipy(path, no_progress_bars=False):
    """
    Install clipy from GitHub repository to the specified path.
    """
    log = get_logger("clipy")
    log.info("Installing clipy from GitHub repository...")

    # Install pre-requisites
    log.info("Installing pre-requisites...")
    for req in ("numpy", "astropy"):
        exit_status = pip_install(req)
        if exit_status:
            log.error("Failed installing '%s'.", req)
            return False

    # Download clipy from GitHub
    log.info("Downloading clipy...")
    if not download_file(clipy_url, path, no_progress_bars=no_progress_bars, logger=log):
        log.error("Not possible to download clipy.")
        return False

    # Extract and move clipy to the correct location
    import zipfile
    import shutil
    zip_file = os.path.join(path, "main.zip")
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(path)
        # Move clipy-main to clipy
        clipy_main_path = os.path.join(path, "clipy-main")
        clipy_path = os.path.join(path, "clipy")
        if os.path.exists(clipy_main_path):
            if os.path.exists(clipy_path):
                shutil.rmtree(clipy_path)
            shutil.move(clipy_main_path, clipy_path)
        # Clean up zip file
        os.remove(zip_file)

    # Verify installation
    if not is_installed_clipy(path):
        log.error("clipy installation verification failed.")
        return False

    log.info("clipy installation finished!")
    return True


def get_product_id_and_clik_file(name):
    """Gets the PLA product info from the defaults file."""
    defaults = get_default_info(name, "likelihood")
    return defaults.get("product_id"), defaults.get("clik_file")


class Planck2015Clik(PlanckClik):
    bibtex_file = "planck2015.bibtex"


class Planck2018Clik(PlanckClik):
    bibtex_file = "planck2018.bibtex"
