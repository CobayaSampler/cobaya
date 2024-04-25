"""
Automatic tests of models.rst example.

For not it only checks that it does not fail.
"""

import os
from cobaya.tools import working_directory

tests_folder = os.path.dirname(os.path.realpath(__file__))
docs_folder = os.path.join(tests_folder, "..", "docs")
docs_src_folder = os.path.join(docs_folder, "src_examples", "advanced_and_models")


def test_docs_model():
    with working_directory(docs_src_folder):
        globals_example = {}
        exec(open(os.path.join(docs_src_folder, "model_create.py")).read(),
             globals_example)
