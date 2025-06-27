"""
.. module:: bib

:Synopsis: Tools and script to get the bibliography to be cited for each component
:Author: Jesus Torrado

Inspired by a similar characteristic of
`CosmoSIS <https://bitbucket.org/joezuntz/cosmosis/wiki/Home>`_.

"""

import argparse
import os
from inspect import cleandoc

from cobaya.component import ComponentNotFoundError, get_component_class
from cobaya.conventions import Extension, dump_sort_cosmetic
from cobaya.input import get_used_components, load_input
from cobaya.log import get_logger, logger_setup
from cobaya.tools import create_banner, similar_internal_class_names, warn_deprecation
from cobaya.typing import InfoDict, InputDict

# Banner defaults
_default_symbol = "="
_default_length = 80

# Cobaya's own bib info
cobaya_desc = cleandoc(r"""
The posterior has been explored/maximized/reweighted using Cobaya \cite{torrado:2020dgo}.
""")

cobaya_bib = r"""
@article{Torrado:2020dgo,
    author = "Torrado, Jesus and Lewis, Antony",
    title = "{Cobaya: Code for Bayesian Analysis of hierarchical physical models}",
    eprint = "2005.05290",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    reportNumber = "TTK-20-15",
    doi = "10.1088/1475-7516/2021/05/057",
    journal = "JCAP",
    volume = "05",
    pages = "057",
    year = "2021"
}
""".lstrip("\n")


def get_desc_component(component, kind, info=None):
    """Extract a short description of a component, if defined."""
    cls = get_component_class(component, kind)
    return cleandoc(cls.get_desc(info) or "") + "\n"


def get_bib_component(component, kind):
    """Extract the bibliographic sources of a component, if defined."""
    cls = get_component_class(component, kind)
    lines = (cls.get_bibtex() or "").lstrip("\n").rstrip(
        "\n"
    ) or "# [no bibliography information found]"
    return lines + "\n"


def get_bib_info(*infos, logger=None):
    """
    Gathers and returns the descriptions and bibliographic sources for the components
    mentioned in ``infos``.

    ``infos`` can be input dictionaries or single component names.
    """
    if not logger:
        logger_setup()
        logger = get_logger("bib")
    used_components, component_infos = get_used_components(*infos, return_infos=True)
    descs: InfoDict = {}
    bibs: InfoDict = {}
    used_components = get_used_components(*infos)
    for kind, components in used_components.items():
        if kind is None:
            continue  # we will deal with bare component names later, to avoid repetition
        descs[kind], bibs[kind] = {}, {}
        for component in components:
            try:
                descs[kind][component] = get_desc_component(
                    component, kind, component_infos[component]
                )
                bibs[kind][component] = get_bib_component(component, kind)
            except ComponentNotFoundError:
                sugg = similar_internal_class_names(component)
                logger.error(
                    f"Could not identify component '{component}'. "
                    f"Did you mean any of the following? {sugg} (mind capitalization!)"
                )
                continue
    # Deal with bare component names
    for component in used_components.get(None, []):
        kind = None
        if ":" in component:
            kind, component = component.split(":")
        try:
            cls = get_component_class(component, kind=kind, logger=logger)
        except ComponentNotFoundError:
            sugg = similar_internal_class_names(component)
            logger.error(
                f"Could not identify component '{component}'. "
                f"Did you mean any of the following? {sugg} (mind capitalization!)"
            )
            continue
        kind = cls.get_kind()
        if kind not in descs:
            descs[kind], bibs[kind] = {}, {}
        if kind in descs and component in descs[kind]:
            continue  # avoid repetition
        descs[kind][component] = get_desc_component(cls, kind)
        bibs[kind][component] = get_bib_component(cls, kind)
    descs["cobaya"] = {"cobaya": cobaya_desc}
    bibs["cobaya"] = {"cobaya": cobaya_bib}
    return descs, bibs


def pretty_repr_bib(descs, bibs):
    """
    Generates a pretty-print multi-line string from component descriptions and
    bibliographical sources.
    """
    # Sort them "optimally"
    sorted_kinds = [k for k in dump_sort_cosmetic if k in descs]
    sorted_kinds += [k for k in descs if k not in dump_sort_cosmetic]
    txt = ""
    txt += (
        create_banner("Descriptions", symbol=_default_symbol, length=_default_length)
        + "\n"
    )
    for kind in sorted_kinds:
        txt += kind + ":\n\n"
        for component, desc in descs[kind].items():
            txt += f" * [{component}] {desc}\n"
        txt += "\n"
    txt += (
        "\n"
        + create_banner("Bibtex", symbol=_default_symbol, length=_default_length)
        + "\n"
    )
    for kind in sorted_kinds:
        for component, bib in bibs[kind].items():
            txt += "\n### %s " % component + "########################" + "\n\n"
            txt += bib
    return txt.lstrip().rstrip() + "\n"


# Command-line script ####################################################################


def bib_script(args=None):
    """Command line script for the bibliography."""
    warn_deprecation()
    # Parse arguments and launch
    parser = argparse.ArgumentParser(
        prog="cobaya bib",
        description=(
            "Prints bibliography to be cited for one or more components or input files."
        ),
    )
    parser.add_argument(
        "files_or_components",
        action="store",
        nargs="+",
        metavar="input_file.yaml|component_name",
        help="Component(s) or input file(s) whose bib info is requested.",
    )
    arguments = parser.parse_args(args)
    # Configure the logger ASAP
    logger_setup()
    logger = get_logger("bib")
    # Gather requests
    infos: list[InputDict | str] = []
    for f in arguments.files_or_components:
        if os.path.splitext(f)[1].lower() in Extension.yamls:
            infos += [load_input(f)]
        else:  # a single component name, no kind specified
            infos += [f]
    if not infos:
        logger.info("Nothing to do. Pass input files or component names as arguments.")
        return
    print(pretty_repr_bib(*get_bib_info(*infos, logger=logger)))


if __name__ == "__main__":
    bib_script()
