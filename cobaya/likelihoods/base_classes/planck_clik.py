r"""
.. module:: planck_clik

:Synopsis: Definition of the clik-based likelihoods
:Author: Jesus Torrado (initially based on MontePython's version
         by Julien Lesgourgues and Benjamin Audren)

"""
# Global
import os
import sys
import numpy as np
from packaging import version

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError, get_logger
from cobaya.input import get_default_info
from cobaya.install import pip_install, download_file
from cobaya.component import ComponentNotInstalledError, load_external_module
from cobaya.tools import are_different_params_lists, create_banner, VersionCheckError

_deprecation_msg_2015 = create_banner("""
The likelihoods from the Planck 2015 data release have been superseded
by the 2018 ones, and will eventually be deprecated.
""")

pla_url_prefix = r"https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.COSMOLOGY_OID="

last_version_supp_data_and_covmats = "v2.01"
last_version_clik = "3.1"


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
            install_path = (
                lambda p: self.get_code_path(p) if p else None)(self.packages_path)
            # min_version here is checked inside get_clik_import_path, since it is
            # displayed in the folder name and cannot be retrieved from the module.
            clik = load_clik(
                "clik", path=self.path, install_path=install_path,
                get_import_path=get_clik_import_path, logger=self.log,
                not_installed_level="debug")
        except VersionCheckError as excpt:
            raise VersionCheckError(
                str(excpt) + " Upgrade with `cobaya-install planck_2018_lowl.TT "
                "--upgrade`.")
        except ComponentNotInstalledError as excpt:
            raise ComponentNotInstalledError(
                self.log, (f"Could not find clik: {excpt}. "
                           "To install it, run `cobaya-install planck_2018_lowl.TT`"))
        # Loading the likelihood data
        data_path = get_data_path(self.__class__.get_qualified_class_name())
        if not os.path.isabs(self.clik_file):
            self.path_data = getattr(self, "path_data", os.path.join(
                self.path or self.packages_path, "data", data_path))
            self.clik_file = os.path.join(self.path_data, self.clik_file)
        # Differences in the wrapper for lensing and non-lensing likes
        self.lensing = clik.try_lensing(self.clik_file)
        try:
            self.clik = clik.clik_lensing(self.clik_file) if self.lensing \
                else clik.clik(self.clik_file)
        except clik.lkl.CError:
            # Is it that the file was not found?
            if not os.path.exists(self.clik_file):
                raise ComponentNotInstalledError(
                    self.log, "The .clik file was not found where specified in the "
                              "'clik_file' field of the settings of this likelihood. "
                              "Maybe the 'path' given is not correct? The full path where"
                              " the .clik file was searched for is '%s'", self.clik_file)
            # Else: unknown clik error
            self.log.error("An unexpected error occurred in clik (possibly related to "
                           "multiple simultaneous initialization, or simultaneous "
                           "initialization of incompatible likelihoods; e.g. polarised "
                           "vs non-polarised 'lite' likelihoods. See error info below:")
            raise
        self.l_maxs = self.clik.get_lmax()
        # calculate requirements here so class can also be separately instantiated
        requested_cls = ["tt", "ee", "bb", "te", "tb", "eb"]
        if self.lensing:
            has_cl = [lmax != -1 for lmax in self.l_maxs]
            requested_cls = ["pp"] + requested_cls
        else:
            has_cl = self.clik.get_has_cl()
        self.requested_cls = [cl for cl, i in zip(requested_cls, has_cl) if int(i)]
        self.l_maxs_cls = [lmax for lmax, i in zip(self.l_maxs, has_cl) if int(i)]
        self.expected_params = list(self.clik.extra_parameter_names)
        # Placeholder for vector passed to clik
        length = (len(self.l_maxs) if self.lensing else len(self.clik.get_has_cl()))
        self.vector = np.zeros(np.sum(self.l_maxs) + length + len(self.expected_params))

    def initialize_with_params(self):
        # Check that the parameters are the right ones
        differences = are_different_params_lists(
            self.input_params, self.expected_params, name_A="given", name_B="expected")
        if differences:
            raise LoggedError(
                self.log, "Configuration error in parameters: %r. "
                          "If this has happened without you fiddling with the defaults, "
                          "please open an issue in GitHub.", differences)

    def get_requirements(self):
        # State requisites to the theory code
        return {'Cl': dict(zip(self.requested_cls, self.l_maxs_cls))}

    def logp(self, **params_values):
        # get Cl's from the theory code
        cl = self.provider.get_Cl(units="FIRASmuK2")
        return self.log_likelihood(cl, **params_values)

    def log_likelihood(self, cl, **params_values):
        # fill with Cl's
        self.vector[:-len(self.expected_params)] = np.concatenate(
            [(cl[spectrum][:1 + lmax] if spectrum not in ["tb", "eb"]
              else np.zeros(1 + lmax))
             for spectrum, lmax in zip(self.requested_cls, self.l_maxs_cls)])
        # check for nan's: mey produce a segfault in clik
        # dot product is apparently the fastest way in threading-enabled numpy
        if np.isnan(np.dot(self.vector, self.vector)):
            return -np.inf
        # fill with likelihood parameters
        self.vector[-len(self.expected_params):] = (
            [params_values[p] for p in self.expected_params])
        loglike = self.clik(self.vector)[0]
        # "zero" of clik, and sometimes nan's returned
        if np.allclose(loglike, -1e30) or np.isnan(loglike):
            loglike = -np.inf
        return loglike

    def close(self):
        del self.clik  # MANDATORY: forces deallocation of the Cython class
        # Actually, it does not work for low-l likelihoods, which is quite dangerous!

    @classmethod
    def get_code_path(cls, path):
        return os.path.realpath(os.path.join(path, "code", common_path))

    @classmethod
    def is_compatible(cls):
        import platform
        if platform.system() == "Windows":
            return False
        return True

    @classmethod
    def is_installed(cls, reload=False, **kwargs):
        code_path = common_path
        data_path = get_data_path(cls.get_qualified_class_name())
        result = True
        if kwargs.get("code", True):
            result &= bool(is_installed_clik(
                os.path.realpath(os.path.join(kwargs["path"], "code", code_path)),
                reload=reload))
        if kwargs.get("data", True):
            # NB: will never raise VersionCheckerror, since version number is in the path
            _, filename = get_product_id_and_clik_file(cls.get_qualified_class_name())
            result &= os.path.exists(os.path.realpath(
                os.path.join(kwargs["path"], "data", data_path, filename)))
            # Check for additional data and covmats -- can raise VersionCheckerror
            from cobaya.likelihoods.planck_2018_lensing import native
            result &= native.is_installed(**kwargs)
        return result

    @classmethod
    def install(cls, path=None, force=False, code=True, data=True,
                no_progress_bars=False):
        name = cls.get_qualified_class_name()
        log = get_logger(name)
        path_names = {"code": common_path, "data": get_data_path(name)}
        import platform
        if platform.system() == "Windows":
            log.error("Not compatible with Windows.")
            return False
        global _clik_install_failed
        if _clik_install_failed:
            log.info("Previous clik install failed, skipping")
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
        # Install clik
        if code and (not is_installed_clik(paths["code"]) or force):
            log.info("Installing the clik code.")
            success *= install_clik(paths["code"], no_progress_bars=no_progress_bars)
            if not success:
                log.warning("clik code installation failed! "
                            "Try configuring+compiling by hand at " + paths["code"])
                _clik_install_failed = True
        if data:
            # 2nd test, in case the code wasn't there but the data is:
            if force or not cls.is_installed(path=path, code=False, data=True):
                log.info("Downloading the likelihood data.")
                product_id, _ = get_product_id_and_clik_file(name)
                # Download and decompress the particular likelihood
                url = pla_url_prefix + product_id
                # Helper for the progress bars: some known product download sizes
                # (no actual effect if missing or wrong!)
                size = {"1900": 314153370, "1903": 4509715660, "151902": 60293120,
                        "151905": 5476083302, "151903": 8160437862}.get(product_id)
                if not download_file(url, paths["data"], size=size, decompress=True,
                                     logger=log, no_progress_bars=no_progress_bars):
                    log.error("Not possible to download this likelihood.")
                    success = False
                # Additional data and covmats, stored in same repo as the
                # 2018 python lensing likelihood
                from cobaya.likelihoods.planck_2018_lensing import native
                if not native.is_installed(data=True, path=path):
                    success *= native.install(path=path, force=force, code=code,
                                              data=data,
                                              no_progress_bars=no_progress_bars)
        return success


# Installation routines ##################################################################

# path to be shared by all Planck likelihoods
common_path = "planck"

# To see full clik build output even if installs OK (e.g. to check warnings)
_clik_verbose = any((s in os.getenv('TRAVIS_COMMIT_MESSAGE', ''))
                    for s in ["clik", "planck"])
# Don't try again to install clik if it failed for a previous likelihood
_clik_install_failed = False


def get_data_path(name):
    return common_path + "_%s" % get_release(name)


def get_release(name):
    return next(re for re in ["2015", "2018"] if re in name)


def get_clik_source_folder(starting_path):
    """
    Starting from the installation folder, returns the subdirectory from which the
    compilation must be run.

    In practice, crawls inside the install folder ``packages/code/planck``
    until >1 subfolders.

    Raises ``FileNotFoundError`` if no clik install was found.
    """
    source_dir = starting_path
    while True:
        folders = [f for f in os.listdir(source_dir)
                   if os.path.isdir(os.path.join(source_dir, f))]
        if len(folders) > 1:
            break
        elif len(folders) == 0:
            raise FileNotFoundError(
                f"Could not find a clik installation under {starting_path}")
        source_dir = os.path.join(source_dir, folders[0])
    return source_dir


def get_clik_import_path(path, min_version=last_version_clik):
    """
    Starting from the installation folder, returns the subdirectory from which the
    ``clik`` module must be imported.

    Raises ``FileNotFoundError`` if no clik install was found, or
    :class:`tools.VersionCheckError` if the installed version is too old.
    """
    clik_src_path = get_clik_source_folder(path)
    installed_version = version.parse(clik_src_path.rstrip(os.sep).split("-")[1])
    if installed_version < version.parse(min_version):
        raise VersionCheckError(
            f"Installed version of the Plack likelihood code 'clik' ({installed_version})"
            f" older than minimum required one ({last_version_clik}).")
    elif installed_version > version.parse(last_version_clik):
        raise ValueError("This should not happen: min version needs update.")
    return os.path.join(clik_src_path, 'lib/python/site-packages')


def load_clik(*args, **kwargs):
    """
    Just a wrapper around :func:`component.load_external_module`, that checks that we are
    not being fooled by the wrong `clik <https://pypi.org/project/click/>`_.
    """
    clik = load_external_module(*args, **kwargs)
    if not hasattr(clik, "try_lensing"):
        raise ComponentNotInstalledError(
            kwargs.get("logger"), "Loaded wrong clik: `https://pypi.org/project/click/`")
    return clik


def is_installed_clik(path, reload=False):
    # min_version here is checked inside get_clik_import_path, since it is displayed
    # in the folder name and cannot be retrieved from the module.
    try:
        return bool(load_clik(
            "clik", path=path, get_import_path=get_clik_import_path,
            reload=reload, logger=get_logger("clik"), not_installed_level="debug"))
    except ComponentNotInstalledError:
        return False


def execute(command):
    from subprocess import Popen, PIPE, STDOUT
    if _clik_verbose:
        process = Popen(command, stdout=PIPE, stderr=STDOUT)
        out = []
        assert process.stdout
        while True:
            nextline = process.stdout.readline()
            if nextline == b"" and process.poll() is not None:
                break
            sys.stdout.buffer.write(nextline)
            out.append(nextline)
            sys.stdout.flush()
        _, err = process.communicate()
        return b"finished successfully" in out[-1]
    else:
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        OK = b"finished successfully" in stdout.split(b"\n")[-2]
        if not OK:
            print(stdout.decode('utf-8'))
            print(stderr.decode('utf-8'))
        return OK


def install_clik(path, no_progress_bars=False):
    log = get_logger("clik")
    log.info("Installing pre-requisites...")
    for req in ("cython", "astropy"):
        exit_status = pip_install(req)
        if exit_status:
            raise LoggedError(log, "Failed installing '%s'.", req)
    log.info("Downloading...")
    click_url = pla_url_prefix + '152000'
    if not download_file(click_url, path, size=2369782, decompress=True,
                         no_progress_bars=no_progress_bars, logger=log):
        log.error("Not possible to download clik.")
        return False
    source_dir = get_clik_source_folder(path)
    log.info('Installing from directory %s' % source_dir)
    cwd = os.getcwd()
    try:
        os.chdir(source_dir)
        log.info("Configuring... (and maybe installing dependencies...)")
        flags = ["--install_all_deps",
                 "--extra_lib=m"]  # missing for some reason in some systems, but harmless
        if not execute([sys.executable, "waf", "configure"] + flags):
            log.error("Configuration failed!")
            return False
        log.info("Compiling...")
        if not execute([sys.executable, "waf", "install"]):
            log.error("Compilation failed!")
            return False
    finally:
        os.chdir(cwd)
    log.info("Finished!")
    return True


def get_product_id_and_clik_file(name):
    """Gets the PLA product info from the defaults file."""
    defaults = get_default_info(name, "likelihood")
    return defaults.get("product_id"), defaults.get("clik_file")


class Planck2015Clik(PlanckClik):
    bibtex_file = 'planck2015.bibtex'


class Planck2018Clik(PlanckClik):
    bibtex_file = 'planck2018.bibtex'
