"""
Automatic tests of the quickstart example in the documentation,
to make sure it remains up to date.
"""

from __future__ import division
import os
from contextlib import contextmanager
import sys
import numpy as np
from imageio import imread

from cobaya.yaml import yaml_load_file
from cobaya.input import is_equal_info
from cobaya.conventions import _output_prefix
import six
import platform
from six import StringIO

docs_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "docs")
docs_src_folder = os.path.join(docs_folder, "src_examples", "quickstart")
docs_img_folder = os.path.join(docs_folder, "img")

# Number of possible different pixels
if not six.PY3:
    pixel_tolerance = 0.995
else:
    pixel_tolerance = 0.980


@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout


def test_example(tmpdir):
    # temporarily change working directory to be able to run the files "as is"
    cwd = os.getcwd()
    os.chdir(docs_src_folder)
    info_yaml = yaml_load_file("gaussian.yaml")
    info_yaml.pop(_output_prefix)
    globals_example = {}
    exec(open(os.path.join(docs_src_folder, "create_info.py")).read(), globals_example)
    try:
        assert is_equal_info(info_yaml, globals_example["info"]), (
            "Inconsistent info between yaml and interactive.")
        exec(open(os.path.join(docs_src_folder, "load_info.py")).read(), globals_example)
        globals_example["info_from_yaml"].pop(_output_prefix)
        assert is_equal_info(info_yaml, globals_example["info_from_yaml"]), (
            "Inconsistent info between interactive and *loaded* yaml.")
        # Run the chain -- constant seed so results are the same!
        globals_example["info"]["sampler"]["mcmc"] = (
                globals_example["info"]["sampler"]["mcmc"] or {})
        globals_example["info"]["sampler"]["mcmc"].update({"seed": 0})
        exec(open(os.path.join(docs_src_folder, "run.py")).read(), globals_example)
        # Analyze and plot -- capture print output
        stream = StringIO()
        with stdout_redirector(stream):
            exec(open(os.path.join(docs_src_folder, "analyze.py")).read(),
                 globals_example)
        # Comparing text output
        out_filename = "analyze_out.txt"
        contents = "".join(open(os.path.join(docs_src_folder, out_filename)).readlines())
        # The endswith guarantees that getdist messages and warnings are ignored
        if platform.system() != 'Windows':
            # randoms not reproducible on Windows in general
            assert stream.getvalue().replace("\n", "").replace(" ", "").endswith(
                contents.replace("\n", "").replace(" ", "")), (
                    "Text output does not coincide:\nwas\n%s\nand " % contents +
                    "now it's\n%sstream.getvalue()" % stream.getvalue())
        # Comparing plot
        # plot_filename = "example_quickstart_plot.png"
        # test_filename = tmpdir.join(plot_filename)
        # globals_example["gdplot"].export(str(test_filename))
        # print("Plot created at '%s'" % str(test_filename))
        # test_img = imread(str(test_filename)).astype(float)
        # docs_img = imread(os.path.join(docs_img_folder, plot_filename)).astype(float)
        # npixels = test_img.shape[0] * test_img.shape[1]
        # assert (np.count_nonzero(test_img == docs_img) / (4 * npixels) >=
        #         pixel_tolerance), (
        #     "Images are too different. Maybe GetDist conventions changed?")
    except:
        raise
    finally:
        # Back to the working directory of the tests, just in case, and restart the rng
        os.chdir(cwd)
