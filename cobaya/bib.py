"""
.. module:: bib

:Synopsis: Tools and script to get the bibliography to be cited for each component
:Author: Jesus Torrado

Inspired by a similar characteristic of
`CosmoSIS <https://bitbucket.org/joezuntz/cosmosis/wiki/Home>`_.

"""

# Global
import os

# Local
from cobaya.conventions import _yaml_extensions, kinds
from cobaya.tools import create_banner, warn_deprecation
from cobaya.input import load_input, get_used_components, get_class

# Banner defaults
_default_symbol = "="
_default_length = 80

# Cobaya's own bib info
cobaya_bib = """
The posterior has been explored/maximised/reweighted using Cobaya \cite{torrado:2020xyz}.

@article{Torrado:2020xyz,
    author = "Torrado, Jesus and Lewis, Antony",
    title = "{Cobaya: Code for Bayesian Analysis of hierarchical physical models}",
    eprint = "2005.05290",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    reportNumber = "TTK-20-15",
    month = "5",
    year = "2020"
}
""".lstrip("\n")


def get_bib_component(component, kind):
    cls = get_class(component, kind, None_if_not_found=True)
    if cls:
        lines = cls.get_bibtex() or "[no bibliography information found]"
    else:
        lines = "[Component '%s.%s' not known.]" % (kind, component)
    return lines + "\n"


def get_bib_info(*infos):
    blocks_text = {"Cobaya": cobaya_bib}
    for kind, components in get_used_components(*infos).items():
        for component in components:
            blocks_text["%s:%s" % (kind, component)] = get_bib_component(component, kind)
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
    from cobaya.mpi import is_main_process
    if not is_main_process():
        return
    warn_deprecation()
    # Parse arguments and launch
    import argparse
    parser = argparse.ArgumentParser(
        description="Prints bibliography to be cited for a component or input file.")
    parser.add_argument("components_or_files", action="store", nargs="+",
                        metavar="component_name or input_file.yaml",
                        help="Component(s) or input file(s) whose bib info is requested.")
    kind_opt, kind_opt_ishort = "kind", 0
    parser.add_argument("-" + kind_opt[kind_opt_ishort], "--" + kind_opt, action="store",
                        nargs=1, default=None, metavar="component_kind",
                        help=("If component name given, "
                              "kind of component whose bib is requested: " +
                              ", ".join(['%s' % kind for kind in kinds]) + ". " +
                              "Use only when component name is not unique (it would fail)."))
    arguments = parser.parse_args()
    # Case of files
    are_yaml = [
        (os.path.splitext(f)[1] in _yaml_extensions) for f in arguments.components_or_files]
    if all(are_yaml):
        infos = [load_input(f) for f in arguments.components_or_files]
        print(prettyprint_bib(get_bib_info(*infos)))
    elif not any(are_yaml):
        if arguments.kind:
            arguments.kind = arguments.kind[0].lower()
        for component in arguments.components_or_files:
            try:
                print(create_banner(
                    component, symbol=_default_symbol, length=_default_length))
                print(get_bib_component(component, arguments.kind))
                return
            except:
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
