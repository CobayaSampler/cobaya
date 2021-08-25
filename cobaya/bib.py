"""
.. module:: bib

:Synopsis: Tools and script to get the bibliography to be cited for each component
:Author: Jesus Torrado

Inspired by a similar characteristic of
`CosmoSIS <https://bitbucket.org/joezuntz/cosmosis/wiki/Home>`_.

"""

# Global
import os
from inspect import cleandoc

# Local
from cobaya.conventions import Extension, kinds, dump_sort_cosmetic
from cobaya.tools import create_banner, warn_deprecation, get_class
from cobaya.input import load_input, get_used_components
from cobaya.typing import InfoDict

# Banner defaults
_default_symbol = "="
_default_length = 80

# Cobaya's own bib info
cobaya_desc = cleandoc(r"""
The posterior has been explored/maximized/reweighted using Cobaya \cite{torrado:2020xyz}.
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
    cls = get_class(component, kind, None_if_not_found=True)
    if cls:
        lines = cleandoc(cls.get_desc(info) or "")
    else:
        lines = "[no description found]"
    return lines + "\n"


def get_bib_component(component, kind):
    cls = get_class(component, kind, None_if_not_found=True)
    if cls:
        lines = ((cls.get_bibtex() or "").lstrip("\n").rstrip("\n")
                 or "# [no bibliography information found]")
    else:
        lines = "# [Component '%s.%s' not known.]" % (kind, component)
    return lines + "\n"


def get_bib_info(*infos):
    used_components, component_infos = get_used_components(*infos, return_infos=True)
    descs: InfoDict = {}
    bibs: InfoDict = {}
    for kind, components in get_used_components(*infos).items():
        descs[kind], bibs[kind] = {}, {}
        for component in components:
            descs[kind][component] = get_desc_component(
                component, kind, component_infos[component])
            bibs[kind][component] = get_bib_component(component, kind)
    descs["cobaya"] = {"cobaya": cobaya_desc}
    bibs["cobaya"] = {"cobaya": cobaya_bib}
    return descs, bibs


def prettyprint_bib(descs, bibs):
    # Sort them "optimally"
    sorted_kinds = [k for k in dump_sort_cosmetic if k in descs]
    sorted_kinds += [k for k in descs if k not in dump_sort_cosmetic]
    txt = ""
    txt += create_banner(
        "Descriptions", symbol=_default_symbol, length=_default_length) + "\n"
    for kind in sorted_kinds:
        txt += kind + ":\n\n"
        for component, desc in descs[kind].items():
            txt += " * [%s] %s\n" % (component, desc)
        txt += "\n"
    txt += "\n" + create_banner(
        "Bibtex", symbol=_default_symbol, length=_default_length) + "\n"
    for kind in sorted_kinds:
        for component, bib in bibs[kind].items():
            txt += "\n### %s " % component + "########################" + "\n\n"
            txt += bib
    return txt.lstrip().rstrip() + "\n"


# Command-line script
def bib_script(args=None):
    warn_deprecation()
    # Parse arguments and launch
    import argparse
    parser = argparse.ArgumentParser(
        prog="cobaya bib",
        description="Prints bibliography to be cited for a component or input file.")
    parser.add_argument("components_or_files", action="store", nargs="+",
                        metavar="component_name or input_file.yaml",
                        help="Component(s) or input file(s) whose bib info is requested.")
    kind_opt, kind_opt_ishort = "kind", 0
    parser.add_argument("-" + kind_opt[kind_opt_ishort], "--" + kind_opt, action="store",
                        default=None, metavar="component_kind",
                        help=("If component name given, "
                              "kind of component whose bib is requested: " +
                              ", ".join(['%s' % kind for kind in kinds]) + ". " +
                              "Use only when component name is not unique "
                              "(it would fail)."))
    arguments = parser.parse_args(args)
    # Case of files
    are_yaml = [
        (os.path.splitext(f)[1] in Extension.yamls) for f in
        arguments.components_or_files]
    if all(are_yaml):
        infos = [load_input(f) for f in arguments.components_or_files]
        print(prettyprint_bib(*get_bib_info(*infos)))
    elif not any(are_yaml):
        if arguments.kind:
            arguments.kind = arguments.kind.lower()
        for component in arguments.components_or_files:
            try:
                print(create_banner(
                    component, symbol=_default_symbol, length=_default_length))
                print(get_bib_component(component, arguments.kind))
            except Exception:
                if not arguments.kind:
                    print("Specify its kind with '--%s [component_kind]'." % kind_opt +
                          "(NB: all requested components must have the same kind, "
                          "or be requested separately).")
                print("")
    else:
        print("Give either a list of input yaml files, "
              "or of component names (not a mix of them).")
        return 1
    return


if __name__ == '__main__':
    bib_script()
