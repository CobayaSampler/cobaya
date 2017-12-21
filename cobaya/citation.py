"""
.. module:: citation

:Synopsis: Tools and script to get the references to cite
:Author: Jesus Torrado

Inspired by a similar characteristic of
`CosmoSIS <https://bitbucket.org/joezuntz/cosmosis/wiki/Home>`_.

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Global
import os

# Local
from cobaya.tools import get_folder, make_header
from cobaya.input import get_modules

def get_citation_info(module, kind):
    folder = get_folder(module, kind, absolute=True)
    filename = os.path.join(folder, module+".bibtex")
    try:
        with open(filename, "r") as f:
            lines = "".join(f.readlines())
    except IOError:
        if not os.path.isdir(folder):
            lines = "[Module '%s.%s' not known.]"%(kind, module)
        else:
            lines = "[no citation information found]"
    return lines+"\n"

def citation(*infos):
    print(make_header("This framework", ""))
    print("[Paper in preparation]\n")
    for kind, modules in get_modules(*infos).items():
        for module in modules:
            print(make_header(kind, module))
            print(get_citation_info(module, kind))

# Command-line script
def citation_script():
    from cobaya.mpi import get_mpi_rank
    if not get_mpi_rank():
        # Configure the logger ASAP
        from cobaya.log import logger_setup
        logger_setup()
        # Parse arguments and launch
        import argparse
        parser = argparse.ArgumentParser(description="Cobaya's citation tool.")
        parser.add_argument("files", action="store", nargs="+", metavar="input_file.yaml",
                            help="One or more input files.")
        from cobaya.input import load_input
        infos = [load_input(f) for f in parser.parse_args().files]
        citation(*infos)
