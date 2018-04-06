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

# Local
from cobaya.log import logger_setup, HandledException
from cobaya.tools import get_folder, make_header
from cobaya.input import get_modules
from cobaya.conventions import package, _code, _data

log = logging.getLogger(__name__.split(".")[-1])


def install(*infos, **kwargs):
    path = kwargs.get("path", ".")
    abspath = os.path.abspath(path)
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
            imported_module = import_module(module_folder, package=package)
            is_installed = getattr(imported_module, "is_installed", None)
            if is_installed is None:
                print("Not and external module: nothing to do.\n")
                continue
            if is_installed(path=abspath, **kwargs_install):
                print("External module already installed.")
                if kwargs_install["force"]:
                    print("Forcing re-installation, as requested.")
                else:
                    print("Doing nothing.\n")
                    continue
            try:
                success = imported_module.install(path=abspath, **kwargs_install)
            except:
                traceback.print_exception(*sys.exc_info(), file=sys.stdout)
                log.error("An unknown error occurred. Delete the modules folder and try "
                          "again. Notify the developers if this error persists.")
                success = False
            if success:
                print("Successfully installed!\n")
            else:
                log.error("Installation failed! Look at the error messages above. "
                          "Solve them and try again, or, if you are unable to solve, "
                          "install this module manually.")
                failed_modules += ["%s:%s"%(kind, module)]
                continue
            # test installation
            if not is_installed(path=abspath, **kwargs_install):
                log.error("Installation apparently worked, "
                          "but the subsequent installation test failed! "
                          "Look at the error messages above. "
                          "Solve them and try again, or, if you are unable to solve, "
                          "install this module manually.")
                failed_modules += ["%s:%s"%(kind, module)]
    if failed_modules:
        log.error("The instalation (or installation test) of some module(s) has failed: "
                  "%r. Check output above.", failed_modules)
        raise HandledException


def user_flag_if_needed():
    """
    Adds --user flag to pip, if needed:
    when not in Travis, Docker, Anaconda or a virtual env.
    """
    if (    "TRAVIS" not in os.environ and  # Travis
            "CONTAINED" not in os.environ and  # Docker, Shifter, Singularity
            not any([(s in sys.version) for s in ["conda", "Continuum"]]) and  # Anaconda
            not hasattr(sys, 'real_prefix') and  # Virtual environment (virtualenv)
            getattr(sys, 'base_prefix', sys.prefix) == sys.prefix):  # Idem (pyvenv)
        return ["--user"]
    return []


def download_github_release(directory, repo_name, release_name, repo_rename=None,
                            github_user="CobayaSampler", no_progress_bars=False):
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
        tar = tarfile.open(filename, "r:"+extension)
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


# Command-line script ####################################################################

def install_script():
    from cobaya.mpi import get_mpi_rank
    if not get_mpi_rank():
        # Configure the logger ASAP
        logger_setup()
        log = logging.getLogger(__name__.split(".")[-1])
        # Parse arguments
        import argparse
        parser = argparse.ArgumentParser(
            description="Cobaya's installation tool for external modules.")
        parser.add_argument("files", action="store", nargs="+", metavar="input_file.yaml",
                            help="One or more input files.")
        parser.add_argument("-p", "--path", action="store", nargs=1, required=True,
                            metavar="/install/path",
                            help="Desired path where to install external modules.")
        parser.add_argument("-f", "--force", action="store_true", default=False,
                            help="Force re-installation of apparently installed modules.")
        parser.add_argument("--no-progress-bars", action="store_true", default=False,
                            help="No progress bars shown. Shorter logs (used in Travis).")
        group_just = parser.add_mutually_exclusive_group(required=False)
        group_just.add_argument("-c", "--just-code", action="store_false", default=True,
                                help="Install code of the modules.", dest=_data)
        group_just.add_argument("-d", "--just-data", action="store_false", default=True,
                                help="Install data of the modules.", dest=_code)
        arguments = parser.parse_args()
        from cobaya.input import load_input
        try:
            infos = [load_input(f) for f in arguments.files]
        except HandledException:
            log.error("Maybe you meant to pass an installation path? "
                      "In that case, use '--path=/path/to/modules'.")
            raise HandledException
        # Launch installer
        install(*infos, path=arguments.path[0],
                **{arg: getattr(arguments, arg)
                   for arg in ["force", _code, _data, "no_progress_bars"]})
