"""
Automatic tests of the quickstart example in the documentation,
to make sure it remains up to date.
"""

from flaky import flaky
from io import StringIO
import os

from cobaya.yaml import yaml_load_file
from cobaya.input import is_equal_info
from cobaya.conventions import _output_prefix
from cobaya.tools import KL_norm
from .common_sampler import KL_tolerance
from .common import stdout_redirector

docs_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "docs")
docs_src_folder = os.path.join(docs_folder, "src_examples", "quickstart")
docs_img_folder = os.path.join(docs_folder, "img")


@flaky(max_runs=3, min_passes=1)
def test_example(tmpdir):
    # temporarily change working directory to be able to run the files "as is"
    cwd = os.getcwd()
    try:
        os.chdir(docs_src_folder)
        info_yaml = yaml_load_file("gaussian.yaml")
        info_yaml.pop(_output_prefix)
        globals_example = {}
        exec(open(os.path.join(docs_src_folder, "create_info.py")).read(), globals_example)
        assert is_equal_info(info_yaml, globals_example["info"]), (
            "Inconsistent info between yaml and interactive.")
        exec(open(os.path.join(docs_src_folder, "load_info.py")).read(), globals_example)
        globals_example["info_from_yaml"].pop(_output_prefix)
        assert is_equal_info(info_yaml, globals_example["info_from_yaml"]), (
            "Inconsistent info between interactive and *loaded* yaml.")
        # Run the chain -- constant seed so results are the same!
        globals_example["info"]["sampler"]["mcmc"] = (
                globals_example["info"]["sampler"]["mcmc"] or {})
        exec(open(os.path.join(docs_src_folder, "run.py")).read(), globals_example)
        # Analyze and plot -- capture print output
        stream = StringIO()
        with stdout_redirector(stream):
            exec(open(os.path.join(docs_src_folder, "analyze.py")).read(),
                 globals_example)
        # Checking results
        mean, covmat = [globals_example["info"]["likelihood"]["gaussian_mixture"][x]
                        for x in ["means", "covs"]]
        assert (KL_norm(
            m1=mean, S1=covmat, m2=globals_example["mean"], S2=globals_example["covmat"])
                <= KL_tolerance), (
            "Sampling appears not to have worked too well. Run again?")
    finally:
        # Back to the working directory of the tests, just in case
        os.chdir(cwd)
