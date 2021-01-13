r"""
.. module:: _planck_clik_prototype

:Synopsis: Definition of the clik-based likelihoods
:Author: Jesus Torrado
         (initially based on MontePython's version by Julien Lesgourgues and Benjamin Audren)

"""
# Global
import os
import sys
import numpy as np
import logging

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from cobaya.conventions import _packages_path, kinds
from cobaya.input import get_default_info
from cobaya.install import pip_install, download_file, NotInstalledError
from cobaya.tools import are_different_params_lists, create_banner, load_module

_deprecation_msg_2015 = create_banner("""
The likelihoods from the Planck 2015 data release have been superseded
by the 2018 ones, and will eventually be deprecated.
""")

pla_url_prefix = r"https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.COSMOLOGY_OID="

last_version_supp_data_and_covmats = "v2.01"


class _planck_clik_prototype(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "CMB"

    path: str

    def initialize(self):
        if "2015" in self.get_name():
            for line in _deprecation_msg_2015.split("\n"):
                self.log.warning(line)
        code_path = common_path
        data_path = get_data_path(self.__class__.get_qualified_class_name())
        # Allow global import if no direct path specification
        allow_global = not self.path
        if self.path:
            self.path_clik = self.path
        elif self.packages_path:
            self.path_clik = self.get_code_path(self.packages_path)
        else:
            raise LoggedError(
                self.log, "No path given to the Planck likelihood. Set the "
                          "likelihood property 'path' or the common property "
                          "'%s'.", _packages_path)
        clik = is_installed_clik(path=self.path_clik, allow_global=allow_global)
        if not clik:
            raise NotInstalledError(
                self.log, "Could not find the 'clik' Planck likelihood code. "
                          "Check error message above.")
        # Loading the likelihood data
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
                raise NotInstalledError(
                    self.log, "The .clik file was not found where specified in the "
                              "'clik_file' field of the settings of this likelihood. "
                              "Maybe the 'path' given is not correct? The full path where"
                              " the .clik file was searched for is '%s'", self.clik_file)
            # Else: unknown clik error
            self.log.error("An unexpected error occurred in clik (possibly related to "
                           "multiple simultaneous initialization, or simultaneous "
                           "initialization of incompatible likelihoods (e.g. polarised "
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
        self.log.info("Initialized!")

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
        # fill with likelihood parameters
        self.vector[-len(self.expected_params):] = (
            [params_values[p] for p in self.expected_params])
        loglike = self.clik(self.vector)[0]
        # "zero" of clik
        if np.allclose(loglike, -1e30):
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
    def is_installed(cls, **kwargs):
        code_path = common_path
        data_path = get_data_path(cls.get_qualified_class_name())
        result = True
        if kwargs.get("code", True):
            result &= bool(is_installed_clik(os.path.realpath(
                os.path.join(kwargs["path"], "code", code_path))))
        if kwargs.get("data", True):
            _, filename = get_product_id_and_clik_file(cls.get_qualified_class_name())
            result &= os.path.exists(os.path.realpath(
                os.path.join(kwargs["path"], "data", data_path, filename)))
            # Check for additional data and covmats
            from cobaya.likelihoods.planck_2018_lensing import native
            result &= native.is_installed(**kwargs)
        return result

    @classmethod
    def install(cls, path=None, force=False, code=True, data=True,
                no_progress_bars=False):
        name = cls.get_qualified_class_name()
        log = logging.getLogger(name)
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
                if not download_file(url, paths["data"], decompress=True,
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
_clik_verbose = any(
    [(s in os.getenv('TRAVIS_COMMIT_MESSAGE', '')) for s in ["clik", "planck"]])
# Don't try again to install clik if it failed for a previous likelihood
try:
    _clik_install_failed
except NameError:
    _clik_install_failed = False


def get_data_path(name):
    return common_path + "_%s" % get_release(name)


def get_release(name):
    return next(re for re in ["2015", "2018"] if re in name)


def get_clik_source_folder(starting_path):
    """Safe source install folder: crawl packages/code/planck until >1 subfolders."""
    source_dir = starting_path
    while True:
        folders = [f for f in os.listdir(source_dir)
                   if os.path.isdir(os.path.join(source_dir, f))]
        if len(folders) > 1:
            break
        elif len(folders) == 0:
            raise FileNotFoundError
        source_dir = os.path.join(source_dir, folders[0])
    return source_dir


def is_installed_clik(path, allow_global=False):
    log = logging.getLogger("clik")
    if path is not None and path.lower() == "global":
        path = None
    clik_path = None
    if path and path.lower() != "global":
        try:
            clik_path = os.path.join(
                get_clik_source_folder(path), 'lib/python/site-packages')
        except FileNotFoundError:
            log.error("The given folder does not exist: '%s'", clik_path or path)
            return False
    if path and not allow_global:
        log.info("Importing *local* clik from %s ", path)
    elif not path:
        log.info("Importing *global* clik.")
    else:
        log.info("Importing *auto-installed* clik (but defaulting to *global*).")
    try:
        return load_module("clik", path=clik_path)
    except ImportError:
        if path is not None and path.lower() != "global":
            log.error("Couldn't find the clik python interface at '%s'. "
                      "Are you sure it has been installed and compiled there?", path)
        else:
            log.error("Could not import global clik installation. "
                      "Specify a Cobaya or clik installation path, "
                      "or install the clik Python interface globally.")
    except Exception as excpt:
        log.error("Error when trying to import clik from %s [%s]. Error message: [%s].",
                  path, clik_path, str(excpt))
        return False


def execute(command):
    from subprocess import Popen, PIPE, STDOUT
    if _clik_verbose:
        process = Popen(command, stdout=PIPE, stderr=STDOUT)
        out = []
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
        out, err = process.communicate()
        OK = b"finished successfully" in out.split(b"\n")[-2]
        if not OK:
            print(out.decode('utf-8'))
            print(err.decode('utf-8'))
        return OK


def install_clik(path, no_progress_bars=False):
    log = logging.getLogger("clik")
    log.info("Installing pre-requisites...")
    for req in ("cython", "astropy"):
        exit_status = pip_install(req)
        if exit_status:
            raise LoggedError(log, "Failed installing '%s'.", req)
    log.info("Downloading...")
    click_url = pla_url_prefix + '151912'
    if not download_file(click_url, path, decompress=True,
                         no_progress_bars=no_progress_bars, logger=log):
        log.error("Not possible to download clik.")
        return False
    source_dir = get_clik_source_folder(path)
    log.info('Installing from directory %s' % source_dir)
    # The following code patches a problem with the download source of cfitsio.
    # Left here in case the FTP server breaks again.
    if True:  # should be fixed: maybe a ping to the FTP server???
        log.info("Patching origin of cfitsio")
        cfitsio_filename = os.path.join(source_dir, "waf_tools", "cfitsio.py")
        with open(cfitsio_filename, "r") as cfitsio_file:
            lines = cfitsio_file.readlines()
            i_offending = next(i for i, l in enumerate(lines) if ".tar.gz" in l)
            lines[i_offending] = lines[i_offending].replace(
                "ftp://heasarc.gsfc.nasa.gov/software/fitsio/c/cfitsio3280.tar.gz",
                "https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3280.tar.gz")
        with open(cfitsio_filename, "w") as cfitsio_file:
            cfitsio_file.write("".join(lines))
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
    defaults = get_default_info(name, kinds.likelihood)
    return defaults.get("product_id"), defaults.get("clik_file")
