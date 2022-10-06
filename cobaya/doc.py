"""
.. module:: doc

:Synopsis: Show defaults options for components.
:Author: Jesus Torrado

"""

# Global
from pprint import pformat

# Local
from cobaya.tools import warn_deprecation, get_available_internal_class_names, \
    similar_internal_class_names
from cobaya.component import get_component_class, ComponentNotFoundError
from cobaya.conventions import subfolders, kinds
from cobaya.input import get_default_info
from cobaya.log import logger_setup, get_logger

_indent = 2 * " "


# Command-line script ####################################################################

def doc_script(args=None):
    """Command line script for the documentation."""
    warn_deprecation()
    logger_setup()
    logger = get_logger("doc")
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        prog="cobaya doc", description="Prints defaults for Cobaya's components.")
    parser.add_argument("component", action="store", nargs="?", default="",
                        metavar="component_name",
                        help=("The component whose defaults are requested. "
                              "Pass a component kind (sampler, theory, likelihood) to "
                              "list all available (internal) ones, pass nothing to list "
                              "all available (internal) components of all kinds."))
    parser.add_argument("-p", "--python", action="store_true", default=False,
                        help="Request Python instead of YAML.")
    expand_flag, expand_flag_ishort = "expand", 1
    parser.add_argument("-" + expand_flag[expand_flag_ishort], "--" + expand_flag,
                        action="store_true", default=False, help="Expand YAML defaults.")
    arguments = parser.parse_args(args)
    # Nothing passed: list all
    if not arguments.component:
        msg = "Available components: (some may need external code/data)"
        print(msg + "\n" + "-" * len(msg))
        for kind in kinds:
            print("%s:" % kind)
            print(_indent + ("\n" + _indent).join(
                get_available_internal_class_names(kind)))
        return
    # A kind passed (plural or singular): list all of that kind
    if arguments.component.lower() in subfolders.values():
        arguments.component = next(
            k for k in subfolders if arguments.component == subfolders[k])
    if arguments.component.lower() in kinds:
        print("%s:" % arguments.component.lower())
        print(_indent +
              ("\n" + _indent).join(
                  get_available_internal_class_names(arguments.component.lower())))
        return
    # Otherwise, try to identify the component
    try:
        cls = get_component_class(arguments.component, logger=logger)
    except ComponentNotFoundError:
        suggestions = similar_internal_class_names(arguments.component)
        logger.error(
            f"Could not identify component '{arguments.component}'. "
            f"Did you mean any of the following? {suggestions} (mind capitalization!)")
        return 1
    to_print = get_default_info(
        cls, return_yaml=not arguments.python, yaml_expand_defaults=arguments.expand)
    if arguments.python:
        print(pformat({cls.get_kind(): {arguments.component: to_print}}))
    else:
        print(cls.get_kind() + ":\n" + _indent + arguments.component + ":\n" +
              2 * _indent + ("\n" + 2 * _indent).join(to_print.split("\n")))
        if "!defaults" in to_print:
            print("# This file contains defaults. "
                  "To populate them, use the flag --%s (or -%s)." % (
                      expand_flag, expand_flag[expand_flag_ishort]))


if __name__ == '__main__':
    doc_script()
