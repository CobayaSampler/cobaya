"""
.. module:: install

:Synopsis: Tools and script to install the modules requested in the given input.
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# Global
import os
import sys
import traceback
import logging
from importlib import import_module
import shutil
from six import string_types

# Local
from cobaya.log import logger_setup, HandledException
from cobaya.tools import get_folder, make_header, warn_deprecation
from cobaya.input import get_modules
from cobaya.conventions import _package, _code, _data, _likelihood, _external
from cobaya.conventions import _modules_path_arg, _path_install

log = logging.getLogger(__name__.split(".")[-1])


def install(*infos, **kwargs):
    if not log.root.handlers:
        logger_setup()
    path = kwargs.get("path", ".")
    if not path:
        # See if we can get one (and only one) from infos
        paths = set([p for p in [info.get(_path_install) for info in infos] if p])
        if len(paths) == 1:
            path = paths[0]
        else:
            print("logging?")
            log.error("No 'path' argument given and could not extract one (and only one) "
                      "from the infos.")
            raise HandledException
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
                log.error("Could not create the desired installation folder '%s'", spath)
                raise HandledException
    failed_modules = []
    for kind, modules in get_modules(*infos).items():
        for module in modules:
            print(make_header(kind, module))
            module_folder = get_folder(module, kind, sep=".", absolute=False)
            try:
                imported_module = import_module(module_folder, package=_package)
            except ImportError:
                if kind == _likelihood:
                    info = (next(info for info in infos
                                 if module in info.get(_likelihood, {}))
                            [_likelihood][module]) or {}
                    if isinstance(info, string_types) or _external in info:
                        log.warning("Module '%s' is a custom likelihood. "
                                    "Nothing to do.\n", module)
                        flag = False
                    else:
                        log.error("Module '%s' not recognized.\n" % module)
                        failed_modules += ["%s:%s" % (kind, module)]
                continue
            is_installed = getattr(imported_module, "is_installed", None)
            if is_installed is None:
                log.info("Built-in module: nothing to do.\n")
                continue
            if is_installed(path=abspath, **kwargs_install):
                log.info("External module already installed.\n")
                if kwargs_install["force"]:
                    log.info("Forcing re-installation, as requested.")
                else:
                    log.info("Doing nothing.\n")
                    continue
            try:
                success = imported_module.install(path=abspath, **kwargs_install)
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
        log.error("The installation (or installation test) of some module(s) has failed: "
                  "%s\nCheck output of the installer of each module above "
                  "for precise error info.\n",
                  bullet + bullet.join(failed_modules))
        raise HandledException


def download_github_release(directory, repo_name, release_name, repo_rename=None,
                            no_progress_bars=False):
    if "/" in repo_name:
        github_user = repo_name[:repo_name.find("/")]
        repo_name = repo_name[repo_name.find("/")+1:]
    else:
        github_user = "CobayaSampler"
    if not os.path.exists(directory):
        os.makedirs(directory)
    try:
        from wget import download, bar_thermometer
        wget_kwargs = {"out": directory,
                       "bar": (bar_thermometer if not no_progress_bars else None)}
        filename = download(
            r"https://github.com/" + github_user + "/" + repo_name +
            "/archive/" + release_name + ".tar.gz", **wget_kwargs)
        print("")  # force newline after wget
    except:
        print("")  # force newline after wget
        log.error("Error downloading!")
        return False
    import tarfile
    extension = os.path.splitext(filename)[-1][1:]
    try:
        if extension == "tgz":
            extension = "gz"
        tar = tarfile.open(filename, "r:" + extension)
        tar.extractall(directory)
        tar.close()
        os.remove(filename)
    except:
        log.error("Error decompressing downloaded file! Corrupt file?)")
        return False
    # Remove version number from directory name
    w_version = next(d for d in os.listdir(directory)
                     if (d.startswith(repo_name) and len(d) != len(repo_name)))
    repo_rename = repo_rename or repo_name
    repo_path = os.path.join(directory, repo_rename)
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    os.rename(os.path.join(directory, w_version), repo_path)
    log.info("%s %s downloaded and uncompressed correctly.", repo_name, release_name)
    return True


def pip_install(packages):
    """
    Takes package name or list of them.

    Retries installation with ``--user`` flag if it fails without it.

    Returns exit status.
    """
    if hasattr(packages, "split"):
        packages = [packages]
    try:
        import pip._internal as pip  # pip version >= 10
    except ImportError:
        import pip  # pip version < 10
    for package in packages:
        for user_flag in [[], ["--user"]]:
            exit_status = pip.main(["install", package] + user_flag)
            if not exit_status:
                # It worked! Let's not try again
                break
        if exit_status:
            log.error("pip: error installing package '%s'", package)
            break
    return exit_status


# Command-line script ####################################################################

def install_script():
    from cobaya.mpi import am_single_or_primary_process
    if not am_single_or_primary_process():
        warn_deprecation()
        # Configure the logger ASAP
        logger_setup()
        log = logging.getLogger(__name__.split(".")[-1])
        # Parse arguments
        import argparse
        parser = argparse.ArgumentParser(
            description="Cobaya's installation tool for external modules.")
        parser.add_argument("files", action="store", nargs="+", metavar="input_file.yaml",
                            help="One or more input files "
                                 "(or 'cosmo' for a basic collection of cosmological modules)")
        parser.add_argument("-" + _modules_path_arg[0], "--" + _modules_path_arg,
                            action="store", nargs=1, required=True,
                            metavar="/install/path",
                            help="Desired path where to install external modules.")
        parser.add_argument("-f", "--force", action="store_true", default=False,
                            help="Force re-installation of apparently installed modules.")
        parser.add_argument("--no-progress-bars", action="store_true", default=False,
                            help="No progress bars shown. Shorter logs (used in Travis).")
        group_just = parser.add_mutually_exclusive_group(required=False)
        group_just.add_argument("-C", "--just-code", action="store_false", default=True,
                                help="Install code of the modules.", dest=_data)
        group_just.add_argument("-D", "--just-data", action="store_false", default=True,
                                help="Install data of the modules.", dest=_code)
        arguments = parser.parse_args()
        if arguments.files == ["cosmo"]:
            log.info("Installing cosmological modules (input files will be ignored")
            from cobaya.cosmo_input import install_basic
            infos = [install_basic]
        else:
            from cobaya.input import load_input
            infos = [load_input(f) for f in arguments.files]
        # Launch installer
        install(*infos, path=getattr(arguments, _modules_path_arg)[0],
                **{arg: getattr(arguments, arg)
                   for arg in ["force", _code, _data, "no_progress_bars"]})
