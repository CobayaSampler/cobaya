import pytest
import os

# Paths ##################################################################################

from cobaya.conventions import _modules_path_arg, _modules_path_env
from cobaya.tools import resolve_modules_path


def pytest_addoption(parser):
    parser.addoption("--" + _modules_path_arg, action="store", default=resolve_modules_path(),
                     help="Path to folder of automatic installation of modules")


@pytest.fixture
def modules(request):
    cmd_modules = request.config.getoption("--" + _modules_path_arg, None)
    if not cmd_modules:
        raise ValueError("Could not determine modules installation path. "
                         "Either define it in the env variable %r, or pass it as an "
                         "argument with `--%s`" % (_modules_path_env, _modules_path_arg))
    return cmd_modules


# Skip certain keywords ##################################################################

def pytest_collection_modifyitems(config, items):
    _env_var = "COBAYA_TEST_SKIP"
    skip_keywords = os.environ.get(_env_var, "").replace(",", " ").split()
    for k in skip_keywords:
        skip_mark = pytest.mark.skip(reason="'%s' skipped by envvar '%s'" % (k, _env_var))
        for item in items:
            if any([(k.lower() in x) for x in [item.name.lower(), item.keywords]]):
                item.add_marker(skip_mark)
