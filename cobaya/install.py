"""
.. module:: install

:Synopsis: Tools and script to install the modules requested in the given input.
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import os
import sys
import logging
from cobaya.log import logger_setup, HandledException
from importlib import import_module

# Local
from cobaya.tools import get_folder, make_header
from cobaya.input import get_modules
from cobaya.conventions import package


def install(*infos, **kwargs):
    log = logging.getLogger(__name__)
    path = kwargs.get("path", ".")
    abspath = os.path.abspath(path)
    if not os.path.exists(abspath):
        log.error("The given path, %s, must exist, but it doesn't."%abspath)
        raise HandledException
    kwargs_install = {"force": kwargs.get("force", False)}
    for s in ("code", "data"):
        kwargs_install[s] = kwargs.get(s, True)
        spath = os.path.join(abspath, s)
        if kwargs_install[s] and not os.path.exists(spath):
            os.makedirs(spath)
    for kind, modules in get_modules(*infos).iteritems():
        for module in modules:
            print make_header(kind, module)
            installpath = os.path.join(abspath, module)
            module_folder = get_folder(module, kind, sep=".", absolute=False)
            imported_module = import_module(module_folder, package=package)
            is_installed = getattr(imported_module, "is_installed", None)
            if is_installed is None:
                print "Not and external module: nothing to do.\n"
                continue
            if is_installed(path=installpath, **kwargs_install):
                print "External module already installed or not requested."
                if kwargs_install["force"]:
                    print "Forcing re-installation, as requested."
                else:
                    print "Doing nothing.\n"
                    continue
            success = imported_module.install(path=installpath, **kwargs_install)
            if success:
                print "Successfully installed!\n"
            else:
                log.error("Installation failed! Look at the error messages above. "
                          "Solve them and try again, or, if you are unable to solve, "
                          "install this module manually.")
                continue
            # test installation
            if not is_installed(path=installpath, **kwargs_install):
                log.error("Installation apparently worked, "
                          "but the subsequent installation test failed! "
                          "Look at the error messages above. "
                          "Solve them and try again, or, if you are unable to solve, "
                          "install this module manually.")


# Add --user flag to pip, if needed: when not in Travis, Docker, Anaconda or a virtual env
def user_flag_if_needed():
    if (    "TRAVIS" not in os.environ and  # Travis
            "CONTAINED" not in os.environ and  # Docker, Shifter, Singularity
            not any([(s in sys.version) for s in ["conda", "Continuum"]]) and  # Anaconda
            not hasattr(sys, 'real_prefix') and  # Virtual environment (virtualenv)
            getattr(sys, 'base_prefix', sys.prefix) == sys.prefix):  # Idem (pyvenv)
        return ["--user"]
    return []


# Command-line script
def install_script():
    from cobaya.mpi import get_mpi_rank
    if not get_mpi_rank():
        # Configure the logger ASAP
        logger_setup()
        log = logging.getLogger(__name__)
        # Parse arguments
        import argparse
        parser = argparse.ArgumentParser(
            description="Cobaya's installation tool for external modules.")
        parser.add_argument("files", action="store", nargs="+", metavar="input_file.yaml",
                            help="One or more input files.")
        parser.add_argument("-p", "--path",
                            action="store", nargs=1, default=".", metavar=("/some/path"),
                            help="Desired path where to install external modules.")
        parser.add_argument("-f", "--force", action="store_true", default=False,
                            help="Force re-installation of apparently installed modules.")
        parser.add_argument("-c", "--just-code", action="store_false", default=True,
                            help="Install code of the modules.", dest="data")
        parser.add_argument("-d", "--just-data", action="store_false", default=True,
                            help="Install data of the modules.", dest="code")
        arguments = parser.parse_args()
        if not arguments.code and not arguments.data:
            log.error("You are not asking either for code or data. "
                      "Use: one of '--just-code' or '--just-data', not both.")
            raise HandledException
        from cobaya.input import load_input
        try:
            infos = [load_input(f) for f in arguments.files]
        except HandledException:
            log.error("Maybe you meant to pass an installation path? "
                      "In that case, use '--path=/path/to/modules'.")
            raise HandledException
        # Launch installer
        install(*infos, path=arguments.path[0],
                **{arg: getattr(arguments, arg) for arg in ["force", "code", "data"]})
