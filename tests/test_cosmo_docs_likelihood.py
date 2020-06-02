"""
Automatic tests of the cosmo_externa_likelihood example in the documentation,
to make sure it remains up to date.
"""

import os

from cobaya.conventions import _packages_path
from .common import process_packages_path
from .conftest import install_test_wrapper

tests_folder = os.path.dirname(os.path.realpath(__file__))
docs_folder = os.path.join(tests_folder, "..", "docs")
docs_src_folder = os.path.join(docs_folder, "src_examples", "cosmo_external_likelihood")
docs_img_folder = os.path.join(docs_folder, "img")



def test_cosmo_docs_likelihood_camb(packages_path, skip_not_installed):
    packages_path = process_packages_path(packages_path)
    # Since we are going to change dirs, make it absolute
    packages_path = os.path.abspath(packages_path)
    # Go to the folder containing the python code
    cwd = os.getcwd()
    try:
        os.chdir(docs_src_folder)
        lines = open(os.path.join(docs_src_folder, "1_fiducial_Cl.py")).readlines()
        for i, line in enumerate(lines):
            if line.startswith(_packages_path):
                lines[i] = "packages_path = '%s'" % packages_path.strip("\'\"")
        globals_example = {}
        install_test_wrapper(skip_not_installed, exec, "\n".join(lines), globals_example)
        exec(open(os.path.join(docs_src_folder, "2_function.py")).read(), globals_example)
        exec(open(os.path.join(docs_src_folder, "3_info_and_plots.py")).read(),
             globals_example)
    except Exception as excpt:
        raise excpt
    finally:
        # Back to the working directory of the tests, just in case
        os.chdir(cwd)
