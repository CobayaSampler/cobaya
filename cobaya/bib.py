"""
.. module:: bib

:Synopsis: Tools and script to get the bibliography to be cited for each module
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
from cobaya.conventions import _yaml_extensions, _kinds
from cobaya.tools import create_banner, warn_deprecation
from cobaya.input import load_input, get_used_modules, get_class

# Banner defaults
_default_symbol = "="
_default_length = 80


def get_bib_module(module, kind):
    cls = get_class(module, kind, None_if_not_found=True)
    if cls:
        filename = cls.get_bibtex_file()
        if filename:
            with open(filename, "r") as f:
                lines = "".join(f.readlines())
        else:
            lines = "[no bibliography information found]"
    else:
        lines = "[Module '%s.%s' not known.]" % (kind, module)
    return lines + "\n"


def get_bib_info(*infos):
    blocks_text = odict([["Cobaya", "[Paper in preparation]"]])
    for kind, modules in get_used_modules(*infos).items():
        for module in modules:
            blocks_text["%s:%s" % (kind, module)] = get_bib_module(module, kind)
    return blocks_text


def prettyprint_bib(blocks_text):
    txt = ""
    for block, text in blocks_text.items():
        if not txt.endswith("\n\n"):
            txt += "\n"
        txt += create_banner(block, symbol=_default_symbol, length=_default_length)
        txt += "\n" + text
    return txt.lstrip().rstrip() + "\n"


# Command-line script
def bib_script():
    from cobaya.mpi import am_single_or_primary_process
    if not am_single_or_primary_process():
        return
    warn_deprecation()
    # Parse arguments and launch
    import argparse
    parser = argparse.ArgumentParser(
        description="Prints bibliography to be cited for a module or input file.")
    parser.add_argument("modules_or_files", action="store", nargs="+",
                        metavar="module_name or input_file.yaml",
                        help="Module(s) or input file(s) whose bib info is requested.")
    kind_opt, kind_opt_ishort = "kind", 0
    parser.add_argument("-" + kind_opt[kind_opt_ishort], "--" + kind_opt, action="store",
                        nargs=1, default=None, metavar="module_kind",
                        help=("If module name given, "
                              "kind of module whose bib is requested: " +
                              ", ".join(['%s' % kind for kind in _kinds]) + ". " +
                              "Use only when module name is not unique (it would fail)."))
    arguments = parser.parse_args()
    # Case of files
    are_yaml = [
        (os.path.splitext(f)[1] in _yaml_extensions) for f in arguments.modules_or_files]
    if all(are_yaml):
        infos = [load_input(f) for f in arguments.modules_or_files]
        print(prettyprint_bib(get_bib_info(*infos)))
    elif not any(are_yaml):
        if arguments.kind:
            arguments.kind = arguments.kind[0].lower()
        for module in arguments.modules_or_files:
            try:
                print(create_banner(
                    module, symbol=_default_symbol, length=_default_length))
                print(get_bib_module(module, arguments.kind))
                return
            except:
                if not arguments.kind:
                    print("Specify its kind with '--%s [module_kind]'." % kind_opt +
                          "(NB: all requested modules must have the same kind, "
                          "or be requested separately).")
                print("")
    else:
        print("Give either a list of input yaml files, "
              "or of module names (not a mix of them).")
        return 1
    return
