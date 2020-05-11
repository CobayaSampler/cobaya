import pytest
import os

# Paths ##################################################################################

from cobaya.conventions import _packages_path_arg, _packages_path_env, _test_skip_env
from cobaya.tools import resolve_packages_path


def pytest_addoption(parser):
    parser.addoption("--" + _packages_path_arg, action="store",
                     default=resolve_packages_path(),
                     help="Path to folder of automatic installation of packages")


@pytest.fixture
def packages_path(request):
    cmd_packages_path = request.config.getoption("--" + _packages_path_arg, None)
    if not cmd_packages_path:
        raise ValueError("Could not determine packages installation path. "
                         "Either define it in the env variable %r, or pass it as an "
                         "argument with `--%s`" %
                         (_packages_path_env, _packages_path_arg))
    return cmd_packages_path


# Skip certain keywords ##################################################################

def pytest_collection_modifyitems(config, items):
    skip_keywords = os.environ.get(_test_skip_env, "").replace(",", " ").split()
    for k in skip_keywords:
        skip_mark = pytest.mark.skip(
            reason="'%s' skipped by envvar '%s'" % (k, _test_skip_env))
        for item in items:
            if any([(k.lower() in x) for x in [item.name.lower(), item.keywords]]):
                item.add_marker(skip_mark)
