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
    force = kwargs.get("force", False)
    for kind, modules in get_modules(*infos).iteritems():
        # Create folder for kind
        kindpath = os.path.join(abspath, get_folder("", kind, sep="/", absolute=False))
        if not os.path.exists(kindpath):
            os.makedirs(kindpath)
        for module in modules:
            installpath = os.path.join(kindpath, module)
            print make_header(kind, module)
            module_folder = get_folder(module, kind, sep=".", absolute=False)
            imported_module = import_module(module_folder, package=package)
            is_installed = getattr(imported_module, "is_installed", None)
            if is_installed == None:
                print "Not and external module: nothing to do.\n"
                continue
            if is_installed(path=installpath):
                print "External module appears to be installed."
                if force:
                    print "Forcing re-installation, as requested."
                else:
                    print "Doing nothing.\n"
                    continue
            success = imported_module.install(path=installpath, force=force)
            if success:
                print "Successfully installed!\n"
            else:
                log.error("Installation failed! Look at the error messages above. "
                          "Solve them and try again, or, if you are unable to solve, "
                          "install this module manually.")
                continue
            # test installation
            if not is_installed(path=installpath):
                log.error("Installation apparently worked, but the subsequent installation"
                          " test failed! Look at the error messages above. "
                          "Solve them and try again, or, if you are unable to solve, "
                          "install this module manually.")

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
        arguments = parser.parse_args()
        from cobaya.input import load_input
        infos = [load_input(f) for f in arguments.files]
        # Launch installer
        install(*infos, path=arguments.path[0], force=arguments.force)
