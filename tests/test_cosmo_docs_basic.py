"""
Testing and automatic generation of basic cosmological examples in the docs.
"""

import os
import sys

from cobaya.yaml import yaml_dump, yaml_load
from cobaya.input import is_equal_info
from cobaya.cosmo_input import create_input
from cobaya.tools import sort_cosmetic
from .test_docs_example_quickstart import docs_folder

path = os.path.join(docs_folder, "src_examples", "cosmo_basic")
file_pre = "basic_"
preset_pre = "planck_2018_"


def test_cosmo_docs_basic():
    flag = True
    for theo in ["camb", "classy"]:
        info_new = create_input(preset=preset_pre + theo)
        info_yaml_new = yaml_dump(sort_cosmetic(info_new))
        file_path = os.path.join(path, file_pre + theo + ".yaml")
        with open(file_path) as docs_file:
            info_yaml_docs = "".join(docs_file.readlines())
        info_docs = yaml_load(info_yaml_docs)
        if not is_equal_info(info_new, info_docs, strict=True, print_not_log=True):
            with open(file_path, "w") as docs_file:
                docs_file.write(info_yaml_new)
            flag = False
            print("OLD:\n%s" % info_yaml_docs)
            print("----------------------------------------")
            print("NEW:\n%s" % info_yaml_new)
            sys.stdout.flush()
    assert flag, ("Differences in example input file. "
                  "Files have been re-generated; check out your git diff.")
