"""
.. module:: doc

:Synopsis: Show defaults for modules
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import, print_function, division

# Global
from pprint import pformat

# Local
from cobaya.tools import warn_deprecation, get_class, get_modules
from cobaya.conventions import _sampler, _theory, _likelihood, subfolders
from cobaya.input import get_default_info


_kinds = [_sampler, _theory, _likelihood]
_indent = 2 * " "
import_odict = "from collections import OrderedDict\n\ninfo = "




# Command-line script ####################################################################

def doc_script():
    from cobaya.mpi import am_single_or_primary_process
    if not am_single_or_primary_process():
        return
    warn_deprecation()

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Prints defaults for Cobaya's internal modules.")
    parser.add_argument("kind", action="store", nargs="?", default="",
                        metavar="module_kind",
                        help=("The kind of module whose defaults are requested: "
                              ", ".join(['%s' % kind for kind in _kinds]) +
                              "Can also be a *unique* module name."))
    parser.add_argument("module", action="store", nargs="?", default="",
                        metavar="module_name",
                        help="Name of module whose defaults are requested.")
    parser.add_argument("-p", "--python", action="store_true", default=False,
                        help="Request Python instead of YAML.")
    parser.add_argument("-x", "--expand", action="store_true", default=False,
                        help="Expand YAML defaults.")
    arguments = parser.parse_args()
    # Remove plurals, for user-friendliness
    if arguments.kind in subfolders.values():
        arguments.kind = next(k for k in subfolders if arguments.kind == subfolders[k])
    # Nothing given, list all
    if not arguments.module and not arguments.kind:
        msg = "Available modules: (some may need external code/data)"
        print(msg + "\n" + "-" * len(msg))
        for kind in _kinds:
            print("%s:" % kind)
            print(_indent + ("\n" + _indent).join(get_modules(kind)))
    # Only kind given and it's actually a "kind": list all modules of that kind;
    # otherwise, check if it's a unique module name
    do_print = True
    if arguments.kind and not arguments.module:
        if arguments.kind.lower() in _kinds:
            print("%s:" % arguments.kind)
            print(_indent + ("\n" + _indent).join(get_modules(arguments.kind)))
            do_print = False
    if do_print:
        module, kind = ((arguments.module, arguments.kind) if arguments.module
                        else (arguments.kind, None))
        try:
            to_print = get_default_info(
                module, kind, return_yaml=not arguments.python, yaml_expand_defaults=arguments.expand, fail_if_not_found=True)
            if arguments.python:
                print(import_odict + pformat(to_print))
            else:
                print(to_print)
        except:
            pass
    return
