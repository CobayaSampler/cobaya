r"""
.. module:: _planck_clik_prototype

:Synopsis: Definition of the clik-based likelihoods
:Author: Jesus Torrado
         (initially based on MontePython's version by Julien Lesgourgues and Benjamin Audren)

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Global
import os
import sys
import numpy as np
import logging
import textwrap
import six

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from cobaya.conventions import _path_install, _likelihood
from cobaya.input import get_default_info, HasDefaults
from cobaya.install import pip_install, download_file
from cobaya.tools import are_different_params_lists, create_banner

_deprecation_msg_2015 = create_banner("""
The likelihoods from the Planck 2015 data release have been superseeded
by the 2018 ones, and will eventually be deprecated.
""")

pla_url_prefix = r"https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.COSMOLOGY_OID="

last_version_supp_data_and_covmats = "v2.01"

# py2 compatibility:
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = OSError


class _planck_clik_prototype(Likelihood, HasDefaults):

    def initialize(self):
        if "2015" in self.name:
            for line in _deprecation_msg_2015.split("\n"):
                self.log.warning(line)
        code_path = common_path
        data_path = get_data_path(self.__class__.get_module_name())
        if self.path:
            has_clik = False
        else:
            try:
                import clik
                has_clik = True
            except ImportError:
                has_clik = False
        if not has_clik:
            if not self.path:
                if self.path_install:
                    self.path_clik = os.path.join(self.path_install, "code", code_path)
                else:
                    raise LoggedError(
                        self.log, "No path given to the Planck likelihood. Set the "
                                  "likelihood property 'path' or the common property "
                                  "'%s'.", _path_install)
            else:
                self.path_clik = self.path
            self.log.info("Importing clik from %s", self.path_clik)
            # test and import clik
            is_installed_clik(self.path_clik, log_and_fail=True, import_it=False)
            import clik
        # Loading the likelihood data
        if not os.path.isabs(self.clik_file):
            self.path_data = getattr(self, "path_data", os.path.join(
                self.path or self.path_install, "data", data_path))
            self.clik_file = os.path.join(self.path_data, self.clik_file)
        # Differences in the wrapper for lensing and non-lensing likes
        self.lensing = clik.try_lensing(self.clik_file)
        try:
            self.clik = (
                clik.clik_lensing(self.clik_file) if self.lensing else clik.clik(self.clik_file))
        except clik.lkl.CError:
            # Is it that the file was not found?
            if not os.path.exists(self.clik_file):
                raise LoggedError(
                    self.log, "The .clik file was not found where specified in the "
                              "'clik_file' field of the settings of this likelihood. Maybe the "
                              "'path' given is not correct? The full path where the .clik file was "
                              "searched for is '%s'", self.clik_file)
            # Else: unknown clik error
            self.log.error("An unexpected error occurred in clik (possibly related to "
                           "multiple simultaneous initialization, or simultaneous "
                           "initialization of incompatible likelihoods (e.g. polarised "
                           "vs non-polarised 'lite' likelihoods. See error info below:")
            raise
        # Check that the parameters are the right ones
        self.expected_params = list(self.clik.extra_parameter_names)
        differences = are_different_params_lists(
            self.input_params, self.expected_params, name_A="given", name_B="expected")
        if differences:
            raise LoggedError(
                self.log, "Configuration error in parameters: %r. "
                          "If this has happened without you fiddling with the defaults, "
                          "please open an issue in GitHub.", differences)
        # Placeholder for vector passed to clik
        self.l_maxs = self.clik.get_lmax()
        length = (len(self.l_maxs) if self.lensing else len(self.clik.get_has_cl()))
        self.vector = np.zeros(np.sum(self.l_maxs) + length + len(self.expected_params))

    def add_theory(self):
        # State requisites to the theory code
        requested_cls = ["tt", "ee", "bb", "te", "tb", "eb"]
        if self.lensing:
            has_cl = [lmax != -1 for lmax in self.l_maxs]
            requested_cls = ["pp"] + requested_cls
        else:
            has_cl = self.clik.get_has_cl()
        self.requested_cls = [cl for cl, i in zip(requested_cls, has_cl) if int(i)]
        self.l_maxs_cls = [lmax for lmax, i in zip(self.l_maxs, has_cl) if int(i)]
        self.theory.needs(Cl=dict(zip(self.requested_cls, self.l_maxs_cls)))

    def logp(self, **params_values):
        # get Cl's from the theory code
        cl = self.theory.get_Cl()
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
    def is_installed(cls, **kwargs):
        code_path = common_path
        data_path = get_data_path(cls.get_module_name())
        result = True
        if kwargs["code"]:
            result &= is_installed_clik(os.path.realpath(
                os.path.join(kwargs["path"], "code", code_path)))
        if kwargs["data"]:
            _, filename = get_product_id_and_clik_file(cls.get_module_name())
            result &= os.path.exists(os.path.realpath(
                os.path.join(kwargs["path"], "data", data_path, filename)))
            # Check for additional data and covmats
            from cobaya.likelihoods.planck_2018_lensing.native import native
            result &= native.is_installed(**kwargs)
        return result

    @classmethod
    def install(cls, path=None, force=False, code=True, data=True, no_progress_bars=False):
        name = cls.get_module_name()
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
        # Create common folders: all planck likelihoods share install folder for code and data
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
                # url = get_default_info(name, _likelihood)[_likelihood][name].get("url", url)
                if not download_file(url, paths["data"], decompress=True,
                                     logger=log, no_progress_bars=no_progress_bars):
                    log.error("Not possible to download this likelihood.")
                    success = False
                # Additional data and covmats, stored in same repo as the 2018 python lensing likelihood
                from cobaya.likelihoods.planck_2018_lensing.native import native
                if not native.is_installed(data=True, path=path):
                    success *= native.install(path=path, force=force, code=code, data=data,
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
    """Safe source install folder: crawl modules/code/planck until >1 subfolders."""
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


def is_installed_clik(path, log_and_fail=False, import_it=True):
    log = logging.getLogger("clik")
    clik_path = None
    try:
        clik_path = os.path.join(get_clik_source_folder(path), 'lib/python/site-packages')
    except FileNotFoundError:
        if log_and_fail:
            raise LoggedError(log, "The given folder does not exist: '%s'", clik_path or path)
        return False
    sys.path.insert(0, clik_path)
    try:
        if import_it:
            import clik
        return True
    except:
        print('Failed to import clik')
        if log_and_fail:
            raise LoggedError(log, "Error importing click from: '%s'", clik_path)
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
            if six.PY3:
                sys.stdout.buffer.write(nextline)
            else:
                sys.stdout.write(nextline)
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
        if not execute([sys.executable, "waf", "configure", "--install_all_deps"]):
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
    defaults = get_default_info(name, _likelihood)[_likelihood][name]
    return defaults.get("product_id"), defaults.get("clik_file")
