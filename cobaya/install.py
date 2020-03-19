"""
.. module:: install

:Synopsis: Tools and script to install the modules requested in the given input.
:Author: Jesus Torrado

"""
# Global
import os
import sys
import subprocess
import traceback
import logging
import shutil
from pkg_resources import parse_version

# Local
from cobaya.log import logger_setup, LoggedError
from cobaya.tools import create_banner, warn_deprecation, get_class, load_config_file, \
    write_modules_path_in_config_file
from cobaya.input import get_used_modules, get_kind
from cobaya.conventions import _module_path, _code, _data, _external, _force
from cobaya.conventions import _modules_path_arg, _path_install, _modules_path_env
from cobaya.conventions import _yaml_extensions
from cobaya.mpi import set_mpi_disabled
from cobaya.tools import resolve_modules_path

log = logging.getLogger(__name__.split(".")[-1])


# noinspection PyUnresolvedReferences
def install(*infos, **kwargs):
    if not log.root.handlers:
        logger_setup()
    path = kwargs.get("path")
    if not path:
        path = resolve_modules_path(infos)
    if not path:
        raise LoggedError(
            log, "No 'path' argument given, and none could be found in input infos, the "
                 "%r env variable or the config file. "
                 "Maybe specify one via a command line argument '-%s [...]'?",
            _modules_path_env, _modules_path_arg[0])
    abspath = os.path.abspath(path)
    log.info("Installing modules at '%s'\n", abspath)
    kwargs_install = {"force": kwargs.get("force", False),
                      "no_progress_bars": kwargs.get("no_progress_bars")}
    for what in (_code, _data):
        kwargs_install[what] = kwargs.get(what, True)
        spath = os.path.join(abspath, what)
        if kwargs_install[what] and not os.path.exists(spath):
            try:
                os.makedirs(spath)
            except OSError:
                raise LoggedError(
                    log, "Could not create the desired installation folder '%s'", spath)
    failed_modules = []
    skip_list = os.environ.get("COBAYA_TEST_SKIP", "").replace(",", " ").lower().split()
    for kind, modules in get_used_modules(*infos).items():
        for module in modules:
            print(create_banner(kind + ":" + module, symbol="=", length=80))
            if any(s in module.lower() for s in skip_list):
                log.info("Skipping %s for test skip list %s" % (module, skip_list))
                continue
            info = (next(info for info in infos if module in
                         info.get(kind, {}))[kind][module]) or {}
            if isinstance(info, str) or _external in info:
                log.warning("Module '%s' is a custom function. "
                            "Nothing to do.\n", module)
                continue
            try:
                imported_class = \
                    get_class(module, kind, module_path=info.pop(_module_path, None))
            except ImportError as e:
                log.error("Module '%s' not recognized. [%s]\n" % (module, e))
                failed_modules += ["%s:%s" % (kind, module)]
                continue
            else:
                if any(s in imported_class.__name__.lower() for s in skip_list):
                    log.info(
                        "Skipping %s for test skip list %s" % (imported_class.__name__,
                                                               skip_list))
                    continue
            is_installed = getattr(imported_class, "is_installed", None)
            if is_installed is None:
                log.info("Built-in module %s: nothing to do.\n" % imported_class.__name__)
                continue
            if is_installed(path=abspath, **kwargs_install):
                log.info("External module already installed.\n")
                if kwargs.get("just_check", False):
                    continue
                if kwargs_install["force"]:
                    log.info("Forcing re-installation, as requested.")
                else:
                    log.info("Doing nothing.\n")
                    continue
            else:
                if kwargs.get("just_check", False):
                    log.info("NOT INSTALLED!\n")
                    continue
            try:
                install_this = getattr(imported_class, "install", None)
                success = install_this(path=abspath, **kwargs_install)
            except KeyboardInterrupt:
                raise
            except:
                traceback.print_exception(*sys.exc_info(), file=sys.stdout)
                log.error("An unknown error occurred. Delete the modules folder and try "
                          "again. Notify the developers if this error persists.")
                success = False
            if success:
                log.info("Successfully installed!\n")
            else:
                log.error("Installation failed! Look at the error messages above. "
                          "Solve them and try again, or, if you are unable to solve, "
                          "install this module manually.")
                failed_modules += ["%s:%s" % (kind, module)]
                continue
            # test installation
            if not is_installed(path=abspath, **kwargs_install):
                log.error("Installation apparently worked, "
                          "but the subsequent installation test failed! "
                          "Look at the error messages above. "
                          "Solve them and try again, or, if you are unable to solve, "
                          "install this module manually.")
                failed_modules += ["%s:%s" % (kind, module)]
    if failed_modules:
        bullet = "\n - "
        raise LoggedError(
            log, "The installation (or installation test) of some module(s) has failed: "
                 "%s\nCheck output of the installer of each module above "
                 "for precise error info.\n",
            bullet + bullet.join(failed_modules))
    # Set the installation path in the global config file
    if not kwargs.get("no_set_global", False) and not kwargs.get("just_check", False):
        write_modules_path_in_config_file(abspath)
        log.info("The installation path has been written in the global config file.")


def download_file(filename, path, no_progress_bars=False, decompress=False, logger=None):
    logger = logger or logging.getLogger(__name__)
    try:
        from wget import download, bar_thermometer
        wget_kwargs = {"out": path, "bar":
            (bar_thermometer if not no_progress_bars else None)}
        filename = os.path.normpath(download(filename, **wget_kwargs))
        print("")
        logger.info('Downloaded filename %s' % filename)
    except Exception as excpt:
        logger.error(
            "Error downloading file '%s' to folder '%s': %s", filename, path, str(excpt))
        return False
    logger.debug('Got: %s' % filename)
    if not decompress:
        return True
    extension = os.path.splitext(filename)[-1][1:]
    try:
        if extension == "zip":
            from zipfile import ZipFile
            with ZipFile(filename, 'r') as zipObj:
                zipObj.extractall(path)
        else:
            import tarfile
            if extension == "tgz":
                extension = "gz"
            with tarfile.open(filename, "r:" + extension) as tar:
                tar.extractall(path)
        logger.debug('Decompressed: %s' % filename)
        os.remove(filename)
        return True
    except Exception as e:
        logger.error("Error decompressing downloaded file! Corrupt file? [%s]" % e)
        return False


def download_github_release(directory, repo_name, release_name, repo_rename=None,
                            no_progress_bars=False, logger=None):
    logger = logger or logging.getLogger(__name__)
    if "/" in repo_name:
        github_user = repo_name[:repo_name.find("/")]
        repo_name = repo_name[repo_name.find("/") + 1:]
    else:
        github_user = "CobayaSampler"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = (r"https://github.com/" + github_user + "/" + repo_name +
                "/archive/" + release_name + ".tar.gz")
    if not download_file(filename, directory, decompress=True,
                         no_progress_bars=no_progress_bars, logger=logger):
        return False
    # Remove version number from directory name
    w_version = next(d for d in os.listdir(directory)
                     if (d.startswith(repo_name) and len(d) != len(repo_name)))
    repo_rename = repo_rename or repo_name
    repo_path = os.path.join(directory, repo_rename)
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    os.rename(os.path.join(directory, w_version), repo_path)
    logger.info("%s %s downloaded and decompressed correctly.", repo_name, release_name)
    return True


def pip_install(packages, upgrade=False):
    """
    Takes package name or list of them.

    Uses ``--user`` flag if does not appear to have write permission to python path

    Returns exit status.
    """
    if hasattr(packages, "split"):
        packages = [packages]
    cmd = [sys.executable, '-m', 'pip', 'install']
    if not os.access(os.path.dirname(sys.executable), os.W_OK):
        cmd += ['--user']
    if upgrade:
        cmd += ['--upgrade']
    res = subprocess.call(cmd + packages)
    if res:
        log.error("pip: error installing packages '%s'", packages)
    return res


def check_gcc_version(min_version="6.4", error_returns=None):
    try:
        version = subprocess.check_output(
            "gcc -dumpversion", shell=True, stderr=subprocess.STDOUT).decode().strip()
    except:
        return error_returns
    # Change in gcc >= 7: -dumpversion only dumps major version
    if "." not in version:
        version = subprocess.check_output(
            "gcc -dumpfullversion", shell=True, stderr=subprocess.STDOUT).decode().strip()
    return parse_version(str(min_version)) <= parse_version(version)


# Command-line script ####################################################################

def install_script():
    set_mpi_disabled(True)
    warn_deprecation()
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Cobaya's installation tool for external modules.")
    parser.add_argument("files_or_modules", action="store", nargs="+",
                        metavar="input_file.yaml|module_name",
                        help="One or more input files or module names "
                             "(or simply 'cosmo' for a basic collection of "
                             "cosmological modules)")
    parser.add_argument("-" + _modules_path_arg[0], "--" + _modules_path_arg,
                        action="store", nargs=1, required=False,
                        metavar="/modules/path", default=[None],
                        help="Desired path where to install external modules.")
    parser.add_argument("-" + _force[0], "--" + _force, action="store_true",
                        default=False,
                        help="Force re-installation of apparently installed modules.")
    parser.add_argument("--no-progress-bars", action="store_true", default=False,
                        help="No progress bars shown. Shorter logs (used in Travis).")
    parser.add_argument("--just-check", action="store_true", default=False,
                        help="Just check whether modules are installed.")
    parser.add_argument("--no-set-global", action="store_true", default=False,
                        help="Do not store the installation path for later runs.")
    group_just = parser.add_mutually_exclusive_group(required=False)
    group_just.add_argument("-C", "--just-code", action="store_false", default=True,
                            help="Install code of the modules.", dest=_data)
    group_just.add_argument("-D", "--just-data", action="store_false", default=True,
                            help="Install data of the modules.", dest=_code)
    arguments = parser.parse_args()
    # Configure the logger ASAP
    logger_setup()
    logger = logging.getLogger(__name__.split(".")[-1])
    # Gather requests
    infos = []
    for f in arguments.files_or_modules:
        if f.lower() == "cosmo":
            logger.info("Installing basic cosmological modules.")
            from cobaya.cosmo_input import install_basic
            infos += [install_basic]
        elif f.lower() == "cosmo-tests":
            logger.info("Installing *tested* cosmological modules.")
            from cobaya.cosmo_input import install_tests
            infos += [install_tests]
        elif os.path.splitext(f)[1].lower() in _yaml_extensions:
            from cobaya.input import load_input
            infos += [load_input(f)]
        else:
            try:
                kind = get_kind(f)
                infos += [{kind: {f: None}}]
            except Exception as e:
                logger.warning("Could not identify module %r. Skipping.", f)
    if not infos:
        logger.info("Nothing to install.")
        return
    # Launch installer
    install(*infos, path=getattr(arguments, _modules_path_arg)[0],
            **{arg: getattr(arguments, arg)
               for arg in ["force", _code, _data, "no_progress_bars", "just_check",
                           "no_set_global"]})


if __name__ == '__main__':
    install_script()
