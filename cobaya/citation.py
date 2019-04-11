"""
.. module:: citation

:Synopsis: Tools and script to get the references to cite
:Author: Jesus Torrado

Inspired by a similar characteristic of
`CosmoSIS <https://bitbucket.org/joezuntz/cosmosis/wiki/Home>`_.

"""
# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

# Global
import os
from collections import OrderedDict as odict

# Local
from cobaya.tools import get_folder, make_header, warn_deprecation
from cobaya.input import get_modules


def get_citation_info(module, kind):
    folder = get_folder(module, kind, absolute=True)
    filename = os.path.join(folder, module + ".bibtex")
    try:
        with open(filename, "r") as f:
            lines = "".join(f.readlines())
    except IOError:
        if not os.path.isdir(folder):
            lines = "[Module '%s.%s' not known.]" % (kind, module)
        else:
            lines = "[no citation information found]"
    return lines + "\n"


def citation(*infos):
    blocks_text = odict([["Cobaya", "[Paper in preparation]"]])
    for kind, modules in get_modules(*infos).items():
        for module in modules:
            blocks_text["%s:%s"%(kind, module)] = get_citation_info(module, kind)
    return blocks_text

def prettyprint_citation(blocks_text):
    txt = ""
    for block, text in blocks_text.items():
        if not txt.endswith("\n\n"):
            txt += "\n\n"
        txt += make_header(*block.split(":")) + "\n" + text
    return txt.lstrip().rstrip() + "\n"

# Command-line script
def citation_script():
    warn_deprecation()
    from cobaya.mpi import am_single_or_primary_process
    if am_single_or_primary_process():
        warn_deprecation()
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
        print(prettyprint_citation(citation(*infos)))
