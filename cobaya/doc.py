"""
.. module:: doc

:Synopsis: Show defaults for modules
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import, print_function, division

# Global

# Local
from cobaya.tools import warn_deprecation, get_class, get_modules
from cobaya.conventions import _sampler, _theory, _likelihood, subfolders
from cobaya.yaml import yaml_dump

_kinds = [_sampler, _theory, _likelihood]
_indent = 2 * " "


def dump_defaults(module, kind=None):
    return yaml_dump(get_class(module, kind=kind).get_defaults())


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
    if arguments.kind and not arguments.module:
        if arguments.kind.lower() in _kinds:
            print("%s:" % arguments.kind)
            print(_indent + ("\n" + _indent).join(get_modules(arguments.kind)))
        else:
            print(dump_defaults(arguments.kind, kind=None))
    if arguments.kind and arguments.module:
        print(dump_defaults(arguments.module, kind=arguments.kind))
    return
