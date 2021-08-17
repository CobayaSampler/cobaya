"""
Automatic tests of models.rst example.

For not it only checks that it does not fail.
"""

import os

tests_folder = os.path.dirname(os.path.realpath(__file__))
docs_folder = os.path.join(tests_folder, "..", "docs")
docs_src_folder = os.path.join(docs_folder, "src_examples", "advanced_and_models")


def test_docs_model():
    cwd = os.getcwd()
    try:
        os.chdir(docs_src_folder)
        globals_example = {}
        exec(open(os.path.join(docs_src_folder, "model_create.py")).read(),
             globals_example)
    finally:
        # Back to the working directory of the tests, just in case
        os.chdir(cwd)
