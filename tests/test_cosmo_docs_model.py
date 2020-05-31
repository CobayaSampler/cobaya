"""
Automatic tests of the cosmo_model example in the documentation,
to make sure it remains up to date.
"""

import os
import numpy as np
from io import StringIO

from cobaya.conventions import _packages_path
from .common import process_packages_path, stdout_redirector
from .conftest import install_test_wrapper

tests_folder = os.path.dirname(os.path.realpath(__file__))
docs_folder = os.path.join(tests_folder, "..", "docs")
docs_src_folder = os.path.join(docs_folder, "src_examples", "cosmo_model")
docs_img_folder = os.path.join(docs_folder, "img")

# Number of possible different pixels
pixel_tolerance = 0.995


def test_cosmo_docs_model_classy(packages_path, skip_not_installed):
    packages_path = process_packages_path(packages_path)
    # Go to the folder containing the python code
    cwd = os.getcwd()
    try:
        os.chdir(docs_src_folder)
        globals_example = {}
        exec(open(os.path.join(docs_src_folder, "1.py")).read(), globals_example)
        globals_example["info"][_packages_path] = packages_path
        install_test_wrapper(
            skip_not_installed, exec,
            open(os.path.join(docs_src_folder, "2.py")).read(), globals_example)
        stream = StringIO()
        with stdout_redirector(stream):
            exec(open(os.path.join(docs_src_folder, "3.py")).read(), globals_example)
        # Comparing text output for this cell -- only derived parameter values
        out_filename = "3.out"
        derived_line_old, derived_line_new = map(
            lambda lines: next(line for line in lines[::-1] if line),
            [open(os.path.join(docs_src_folder, out_filename)).readlines(),
             stream.getvalue().split("\n")])
        derived_params_old, derived_params_new = map(
            lambda x: eval(x[x.find("{"):]), [derived_line_old, derived_line_new])
        oldvals = list(derived_params_old.values())
        newvals = [derived_params_new[v] for v in derived_params_old]
        assert np.allclose(oldvals, newvals), (
                "Wrong derived parameters line:\nBEFORE: %s\nNOW:    %s" %
                (derived_line_old, derived_line_new))
        # Not testing figures for now (not robust)
        # Instead, let's just check that no error is raised
        for filename in ["4.py", "5.py"]:
            try:
                exec(open(os.path.join(docs_src_folder, filename)).read(),
                     globals_example)
            except:
                assert False, "File %s failed." % filename
    #        if test_figs:
    #            # Compare plots
    #            pre = "cosmo_model_"
    #            for filename, imgname in zip(["4.py", "5.py"], ["cltt.png", "omegacdm.png"]):
    #                exec(open(os.path.join(docs_src_folder, filename)).read(), globals_example)
    #                old_img = imread(os.path.join(docs_img_folder, pre + imgname)).astype(float)
    #                new_img = imread(imgname).astype(float)
    #                npixels = (lambda x: x.shape[0] + x.shape[1])(old_img)
    #                assert np.count_nonzero(old_img == new_img) / (4 * npixels) >= pixel_tolerance, (
    #                        "Images '%s' are too different!" % imgname)
        # Passing the model to a sampler
        exec(open(os.path.join(docs_src_folder, "6.py")).read(), globals_example)
    finally:
        # Back to the working directory of the tests, just in case
        os.chdir(cwd)
