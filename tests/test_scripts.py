import os

from cobaya.bib import bib_script
from cobaya.doc import doc_script
from cobaya.test_tools import stdout_check


def test_doc():
    with stdout_check("mcmc"):
        doc_script([])  # just lists available components

    with stdout_check("evaluate:"):
        doc_script(["evaluate"])


yaml = """
likelihood:
 gaussian:
sampler:
 minimize:
"""


def test_bib(tmpdir):
    with stdout_check("Neal:2005"):
        bib_script(["gaussian", "mcmc"])

    f = os.path.join(tmpdir, "input.yaml")
    with open(f, "w", encoding="utf-8") as output:
        output.write(yaml)

    with stdout_check("Torrado:2020"):
        bib_script([f])
