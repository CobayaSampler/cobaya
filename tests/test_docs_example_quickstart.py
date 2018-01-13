"""
Automatic tests of the quickstart example in the documentation,
to make sure it remains up to date.
"""

import os

from cobaya.yaml import yaml_load_file
from cobaya.tools import is_equal_info

docs_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../docs")
docs_src_folder = os.path.join(docs_folder, "src_examples/quickstart/")
docs_img_folder = os.path.join(docs_folder, "img")


def test_consistency():
    info_yaml = yaml_load_file(os.path.join(docs_src_folder, "gaussian.yaml"))
    global_vars = {}  # globar vars used by the "interactive example"
    execfile(os.path.join(docs_src_folder, "create_info.py"), global_vars)
    assert is_equal_info(info_yaml, global_vars["info"]), (
        "Inconsistent info between yaml and insteractive.")
    # temporarily change workign directory to load the example yaml file
    cwd = os.getcwd()
    os.chdir(docs_src_folder)
    execfile(os.path.join(docs_src_folder, "load_info.py"), global_vars)
    os.chdir(cwd)
    assert is_equal_info(info_yaml, global_vars["info_from_yaml"]), (
        "Inconsistent info between interactive and *loaded* yaml.")
