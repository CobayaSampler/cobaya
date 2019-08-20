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
from cobaya.tools import warn_deprecation, get_class, get_available_modules
from cobaya.conventions import _sampler, _theory, _likelihood, subfolders, _kinds
from cobaya.input import get_default_info


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
    parser.add_argument("module", action="store", nargs="?", default="",
                        metavar="module_name",
                        help="The module whose defaults are requested.")
    kind_opt, kind_opt_ishort = "kind", 0
    parser.add_argument("-" + kind_opt[kind_opt_ishort], "--" + kind_opt, action="store",
                        nargs=1, default=None, metavar="module_kind",
                        help=("Kind of module whose defaults are requested: " +
                              ", ".join(['%s' % kind for kind in _kinds]) + ". " +
                              "Use only when module name is not unique (it would fail)."))
    parser.add_argument("-p", "--python", action="store_true", default=False,
                        help="Request Python instead of YAML.")
    expand_flag, expand_flag_ishort = "expand", 1
    parser.add_argument("-" + expand_flag[expand_flag_ishort], "--" + expand_flag,
                        action="store_true", default=False, help="Expand YAML defaults.")
    arguments = parser.parse_args()
    # Remove plurals (= name of src subfolders), for user-friendliness
    if arguments.module.lower() in subfolders.values():
        arguments.module = next(k for k in subfolders if arguments.module == subfolders[k])
    # Kind given, list all
    if not arguments.module:
        msg = "Available modules: (some may need external code/data)"
        print(msg + "\n" + "-" * len(msg))
        for kind in _kinds:
            print("%s:" % kind)
            print(_indent + ("\n" + _indent).join(get_available_modules(kind)))
        return
    # Kind given: list all modules of that kind
    if arguments.module.lower() in _kinds:
        print("%s:" % arguments.module.lower())
        print(_indent +
              ("\n" + _indent).join(get_available_modules(arguments.module.lower())))
        return
    # Otherwise, check if it's a unique module name
    try:
        if arguments.kind:
            arguments.kind = arguments.kind[0].lower()
        to_print = get_default_info(
            arguments.module, arguments.kind, return_yaml=not arguments.python,
            yaml_expand_defaults=arguments.expand, fail_if_not_found=True)
        if arguments.python:
            print(import_odict + pformat(to_print))
        else:
            print(to_print)
            if "!defaults" in to_print:
                print("# This file contains defaults. "
                      "To populate them, use the flag --%s (or -%s)." % (
                          expand_flag, expand_flag[expand_flag_ishort]))
    except:
        if not arguments.kind:
            print("Specify its kind with '--%s [module_kind]'." % kind_opt)
        return 1
    return
