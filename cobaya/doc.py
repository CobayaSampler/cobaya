"""
.. module:: doc

:Synopsis: Show defaults for modules
:Author: Jesus Torrado

"""

# Global
from pprint import pformat

# Local
from cobaya.tools import warn_deprecation, get_available_internal_class_names
from cobaya.conventions import subfolders, kinds
from cobaya.input import get_default_info, get_kind
from cobaya.log import LoggedError

_indent = 2 * " "


# Command-line script ####################################################################

def doc_script():
    from cobaya.mpi import is_main_process
    if not is_main_process():
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
                              ", ".join(['%s' % kind for kind in kinds]) + ". " +
                              "Use only when module name is not unique (it would fail)."))
    parser.add_argument("-p", "--python", action="store_true", default=False,
                        help="Request Python instead of YAML.")
    expand_flag, expand_flag_ishort = "expand", 1
    parser.add_argument("-" + expand_flag[expand_flag_ishort], "--" + expand_flag,
                        action="store_true", default=False, help="Expand YAML defaults.")
    arguments = parser.parse_args()
    # Remove plurals (= name of src subfolders), for user-friendliness
    if arguments.module.lower() in subfolders.values():
        arguments.module = next(
            k for k in subfolders if arguments.module == subfolders[k])
    # Kind given, list all
    if not arguments.module:
        msg = "Available modules: (some may need external code/data)"
        print(msg + "\n" + "-" * len(msg))
        for kind in kinds:
            print("%s:" % kind)
            print(
                _indent + ("\n" + _indent).join(get_available_internal_class_names(kind)))
        return
    # Kind given: list all modules of that kind
    if arguments.module.lower() in kinds:
        print("%s:" % arguments.module.lower())
        print(_indent +
              ("\n" + _indent).join(
                  get_available_internal_class_names(arguments.module.lower())))
        return
    # Otherwise, check if it's a unique module name
    try:
        if arguments.kind:
            arguments.kind = arguments.kind[0].lower()
            if arguments.kind not in kinds:
                print("Kind %r not recognized. Try one of %r" % (
                arguments.kind, tuple(kinds)))
                raise ValueError
        else:
            arguments.kind = get_kind(arguments.module)
        to_print = get_default_info(
            arguments.module, arguments.kind, return_yaml=not arguments.python,
            yaml_expand_defaults=arguments.expand)
        if arguments.python:
            print(pformat({arguments.kind: {arguments.module: to_print}}))
        else:
            print(arguments.kind + ":\n" + _indent + arguments.module + ":\n" +
                  2 * _indent + ("\n" + 2 * _indent).join(to_print.split("\n")))
            if "!defaults" in to_print:
                print("# This file contains defaults. "
                      "To populate them, use the flag --%s (or -%s)." % (
                          expand_flag, expand_flag[expand_flag_ishort]))
    except Exception as e:
        if isinstance(Exception, LoggedError.__class__):
            pass
        else:
            if not arguments.kind:
                print("Specify its kind with '--%s [module_kind]'." % kind_opt)
        return 1
    return
