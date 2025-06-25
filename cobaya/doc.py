"""
.. module:: doc

:Synopsis: Show defaults options for components.
:Author: Jesus Torrado and Antony Lewis

"""

import argparse
from inspect import cleandoc
from pprint import pformat

from cobaya.component import ComponentNotFoundError, get_component_class
from cobaya.conventions import kinds, subfolders
from cobaya.input import get_default_info
from cobaya.log import NoLogging, get_logger, logger_setup
from cobaya.tools import (
    get_available_internal_class_names,
    similar_internal_class_names,
    warn_deprecation,
)

_indent = 2 * " "


# Command-line script ####################################################################


def doc_script(args=None):
    """Command line script for the documentation."""
    warn_deprecation()
    logger_setup()
    logger = get_logger("doc")
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="cobaya doc", description="Prints defaults for Cobaya's components."
    )
    parser.add_argument(
        "component",
        action="store",
        nargs="?",
        default="",
        metavar="component_name",
        help=(
            "The component whose defaults are requested. "
            "Pass a component kind (sampler, theory, likelihood) to "
            "list all available (internal) ones, pass nothing to list "
            "all available (internal) components of all kinds. In case "
            "of ambiguity (two components of different kinds sharing a "
            "name), you can pass the kind as 'kind:name'."
        ),
    )
    parser.add_argument(
        "-p",
        "--python",
        action="store_true",
        default=False,
        help="Request Python instead of YAML.",
    )
    expand_flag, expand_flag_ishort = "expand", 1
    parser.add_argument(
        "-" + expand_flag[expand_flag_ishort],
        "--" + expand_flag,
        action="store_true",
        default=False,
        help="Expand YAML defaults.",
    )
    arguments = parser.parse_args(args)
    # Nothing passed: list all
    if not arguments.component:
        msg = "Available components: (some may need external code/data)"
        print(msg + "\n" + "-" * len(msg))
        for kind in kinds:
            print("%s:" % kind)
            print(
                _indent + ("\n" + _indent).join(get_available_internal_class_names(kind))
            )
        return 0
    # A kind passed (plural or singular): list all of that kind
    if arguments.component.lower() in subfolders.values():
        arguments.component = next(
            k for k, sub in subfolders.items() if arguments.component == sub
        )
    if arguments.component.lower() in kinds:
        print("%s:" % arguments.component.lower())
        print(
            _indent
            + ("\n" + _indent).join(
                get_available_internal_class_names(arguments.component.lower())
            )
        )
        return 0
    # Otherwise, try to identify the component
    kind = None
    if ":" in arguments.component:
        kind, arguments.component = arguments.component.split(":")

    try:
        with NoLogging("CRITICAL"):
            cls = get_component_class(arguments.component, kind=kind, logger=logger)
    except (ComponentNotFoundError, AttributeError):
        if matches := list(
            get_available_internal_class_names(kind=kind, stem=arguments.component)
        ):
            if not matches[0].startswith(arguments.component + "."):
                logger.error(
                    f"Could not identify component '{arguments.component}'. "
                    f"Options in this package are:"
                )

            for match in matches:
                description = cleandoc(get_component_class(match).get_desc() or "")
                print(
                    f"{match}:\n"
                    + (
                        (_indent + description.replace("\n", "\n" + _indent))
                        if description
                        else ""
                    )
                    + "\n"
                )
            return 0

        suggestions = similar_internal_class_names(arguments.component)
        logger.error(
            f"Could not identify component '{arguments.component}'. "
            f"Did you mean any of the following? {suggestions} (mind capitalization!)"
        )
        return 1
    to_print = get_default_info(
        cls, return_yaml=not arguments.python, yaml_expand_defaults=arguments.expand
    )
    if arguments.python:
        print(pformat({cls.get_kind(): {arguments.component: to_print}}))
    else:
        print(
            cls.get_kind()
            + ":\n"
            + _indent
            + arguments.component
            + ":\n"
            + 2 * _indent
            + ("\n" + 2 * _indent).join(to_print.split("\n"))
        )
        if "!defaults" in to_print:
            print(
                "# This file contains defaults. "
                "To populate them, use the flag --%s (or -%s)."
                % (expand_flag, expand_flag[expand_flag_ishort])
            )
    return 0


if __name__ == "__main__":
    doc_script()
