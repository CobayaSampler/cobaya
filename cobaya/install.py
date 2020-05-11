"""
.. module:: install

:Synopsis: Tools and script to install the external code and data packages needed by
           the Cobaya components to be used.
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
from cobaya.tools import create_banner, warn_deprecation, get_class, \
    write_packages_path_in_config_file
from cobaya.input import get_used_components, get_kind
from cobaya.conventions import _component_path, _code, _data, _external, _force, \
    _packages_path, _packages_path_arg, _packages_path_env, _yaml_extensions, \
    _install_skip_env, _packages_path_arg_posix
from cobaya.mpi import set_mpi_disabled
from cobaya.tools import resolve_packages_path

log = logging.getLogger(__name__.split(".")[-1])


# noinspection PyUnresolvedReferences
def install(*infos, **kwargs):
    if not log.root.handlers:
        logger_setup()
    path = kwargs.get("path")
    if not path:
        path = resolve_packages_path(infos)
    if not path:
        raise LoggedError(
            log, "No 'path' argument given, and none could be found in input infos "
                 "(as %r), the %r env variable or the config file. "
                 "Maybe specify one via a command line argument '-%s [...]'?",
            _packages_path, _packages_path_env, _packages_path_arg[0])
    abspath = os.path.abspath(path)
    log.info("Installing external packages at '%s'", abspath)
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
    failed_components = []
    skip_keywords = set(kwargs.get("skip", []) or [])
    skip_keywords_env = set(
        os.environ.get(_install_skip_env, "").replace(",", " ").lower().split())
    skip_keywords = skip_keywords.union(skip_keywords_env)
    for kind, components in get_used_components(*infos).items():
        for component in components:
            print()
            print(create_banner(kind + ":" + component, symbol="=", length=80))
            print()
            if _skip_helper(component.lower(), skip_keywords, skip_keywords_env, log):
                continue
            info = (next(info for info in infos if component in
                         info.get(kind, {}))[kind][component]) or {}
            if isinstance(info, str) or _external in info:
                log.warning("Component '%s' is a custom function. "
                            "Nothing to do.", component)
                continue
            try:
                imported_class = get_class(component, kind,
                                           component_path=info.pop(_component_path, None))
            except ImportError as e:
                log.error("Component '%s' not recognized. [%s]." % (component, e))
                failed_components += ["%s:%s" % (kind, component)]
                continue
            else:
                if _skip_helper(imported_class.__name__.lower(), skip_keywords,
                                skip_keywords_env, log):
                    continue
            is_installed = getattr(imported_class, "is_installed", None)
            if is_installed is None:
                log.info("%s.%s is a fully built-in component: nothing to do.",
                         kind, imported_class.__name__)
                continue
            if is_installed(path=abspath, **kwargs_install):
                log.info("External component already installed.")
                if kwargs.get("just_check", False):
                    continue
                if kwargs_install["force"]:
                    log.info("Forcing re-installation, as requested.")
                else:
                    log.info("Doing nothing.")
                    continue
            else:
                if kwargs.get("just_check", False):
                    log.info("NOT INSTALLED!")
                    continue
            try:
                install_this = getattr(imported_class, "install", None)
                success = install_this(path=abspath, **kwargs_install)
            except KeyboardInterrupt:
                raise
            except:
                traceback.print_exception(*sys.exc_info(), file=sys.stdout)
                log.error("An unknown error occurred. Delete the external packages "
                          "folder %r and try again. "
                          "Please, notify the developers if this error persists.",
                          abspath)
                success = False
            if success:
                log.info("Successfully installed!")
            else:
                log.error("Installation failed! Look at the error messages above. "
                          "Solve them and try again, or, if you are unable to solve, "
                          "install the packages required by this component manually.")
                failed_components += ["%s:%s" % (kind, component)]
                continue
            # test installation
            if not is_installed(path=abspath, **kwargs_install):
                log.error("Installation apparently worked, "
                          "but the subsequent installation test failed! "
                          "Look at the error messages above. "
                          "Solve them and try again, or, if you are unable to solve, "
                          "install the packages required by this component manually.")
                failed_components += ["%s:%s" % (kind, component)]
    if failed_components:
        bullet = "\n - "
        raise LoggedError(
            log, "The installation (or installation test) of some component(s) has "
                 "failed: %s\nCheck output of the installer of each component above "
                 "for precise error info.\n",
            bullet + bullet.join(failed_components))
    # Set the installation path in the global config file
    if not kwargs.get("no_set_global", False) and not kwargs.get("just_check", False):
        write_packages_path_in_config_file(abspath)
        log.info("The installation path has been written in the global config file.")


def _skip_helper(name, skip_keywords, skip_keywords_env, logger):
    try:
        this_skip_keyword = next(s for s in skip_keywords
                                 if s.lower() in name.lower())
        env_msg = (" in env var %r" % _install_skip_env
                   if this_skip_keyword in skip_keywords_env else "")
        logger.info("Skipping %r as per skip keyword %r" + env_msg,
                    name, this_skip_keyword)
        return True
    except StopIteration:
        return False


def download_file(filename, path, no_progress_bars=False, decompress=False, logger=None):
    logger = logger or logging.getLogger(__name__)
    try:
        from wget import download, bar_thermometer
        wget_kwargs = {"out": path, "bar":
            (bar_thermometer if not no_progress_bars else None)}
        filename = os.path.normpath(download(filename, **wget_kwargs))
        print("")
        logger.info('Downloaded filename %s' % filename)
    except Exception as e:
        logger.error(
            "Error downloading file '%s' to folder '%s': %s", filename, path, e)
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
        description="Cobaya's installation tool for external packages.")
    parser.add_argument("files_or_components", action="store", nargs="+",
                        metavar="input_file.yaml|component_name",
                        help="One or more input files or component names "
                             "(or simply 'cosmo' to install all the requisites for basic"
                             " cosmological runs)")
    parser.add_argument("-" + _packages_path_arg[0], "--" + _packages_path_arg_posix,
                        action="store", nargs=1, required=False,
                        metavar="/packages/path", default=[None],
                        help="Desired path where to install external packages. "
                             "Optional if one has been set globally or as an env variable"
                             " (run with '--show_%s' to check)." %
                             _packages_path_arg_posix)
    # MARKED FOR DEPRECATION IN v3.0
    modules = "modules"
    parser.add_argument("-" + modules[0], "--" + modules,
                        action="store", nargs=1, required=False,
                        metavar="/packages/path", default=[None],
                        help="To be deprecated! "
                             "Alias for %s, which should be used instead." %
                             _packages_path_arg_posix)
    # END OF DEPRECATION BLOCK -- CONTINUES BELOW!
    output_show_packages_path = resolve_packages_path()
    if output_show_packages_path and os.environ.get(_packages_path_env):
        output_show_packages_path += " (from env variable %r)" % _packages_path_env
    elif output_show_packages_path:
        output_show_packages_path += " (from config file)"
    else:
        output_show_packages_path = "(Not currently set.)"
    parser.add_argument("--show-" + _packages_path_arg_posix, action="version",
                        version=output_show_packages_path,
                        help="Prints default external packages installation folder "
                             "and exits.")
    parser.add_argument("-" + _force[0], "--" + _force, action="store_true",
                        default=False,
                        help="Force re-installation of apparently installed packages.")
    parser.add_argument("--skip", action="store", nargs="*",
                        metavar="keyword",
                        help="Keywords of components that will be skipped during "
                             "installation.")
    parser.add_argument("--no-progress-bars", action="store_true", default=False,
                        help="No progress bars shown. Shorter logs (used in Travis).")
    parser.add_argument("--just-check", action="store_true", default=False,
                        help="Just check whether components are installed.")
    parser.add_argument("--no-set-global", action="store_true", default=False,
                        help="Do not store the installation path for later runs.")
    group_just = parser.add_mutually_exclusive_group(required=False)
    group_just.add_argument("-C", "--just-code", action="store_false", default=True,
                            help="Install code of the components.", dest=_data)
    group_just.add_argument("-D", "--just-data", action="store_false", default=True,
                            help="Install data of the components.", dest=_code)
    arguments = parser.parse_args()
    # Configure the logger ASAP
    logger_setup()
    logger = logging.getLogger(__name__.split(".")[-1])
    # Gather requests
    infos = []
    for f in arguments.files_or_components:
        if f.lower() == "cosmo":
            logger.info("Installing basic cosmological packages.")
            from cobaya.cosmo_input import install_basic
            infos += [install_basic]
        elif f.lower() == "cosmo-tests":
            logger.info("Installing *tested* cosmological packages.")
            from cobaya.cosmo_input import install_tests
            infos += [install_tests]
        elif os.path.splitext(f)[1].lower() in _yaml_extensions:
            from cobaya.input import load_input
            infos += [load_input(f)]
        else:
            try:
                kind = get_kind(f)
                infos += [{kind: {f: None}}]
            except Exception:
                logger.warning("Could not identify component %r. Skipping.", f)
    if not infos:
        logger.info("Nothing to install.")
        return
    # MARKED FOR DEPRECATION IN v3.0
    if getattr(arguments, modules) != [None]:
        logger.warning("*DEPRECATION*: -m/--modules will be deprecated in favor of "
                       "-%s/--%s in the next version. Please, use that one instead.",
                       _packages_path_arg[0], _packages_path_arg_posix)
        # BEHAVIOUR TO BE REPLACED BY ERROR:
        if getattr(arguments, _packages_path_arg) == [None]:
            setattr(arguments, _packages_path_arg, getattr(arguments, modules))
    # END OF DEPRECATION BLOCK
    # Launch installer
    install(*infos, path=getattr(arguments, _packages_path_arg)[0],
            **{arg: getattr(arguments, arg)
               for arg in ["force", _code, _data, "no_progress_bars", "just_check",
                           "no_set_global", "skip"]})


if __name__ == '__main__':
    install_script()
