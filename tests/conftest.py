import pytest
import os
import inspect

# Paths ##################################################################################

from cobaya.conventions import _packages_path_env, _packages_path_arg_posix, \
    _test_skip_env
from cobaya.tools import resolve_packages_path


def pytest_addoption(parser):
    parser.addoption("--" + _packages_path_arg_posix, action="store",
                     default=resolve_packages_path(),
                     help="Path to folder of automatic installation of packages")
    parser.addoption("--skip-not-installed", action="store_true",
                     help="Skip tests for which dependencies of used components are not "
                     "installed.")


@pytest.fixture
def packages_path(request):
    cmd_packages_path = request.config.getoption("--" + _packages_path_arg_posix, None)
    if not cmd_packages_path:
        raise ValueError("Could not determine packages installation path. "
                         "Either define it in the env variable %r, or pass it as an "
                         "argument with `--%s`" %
                         (_packages_path_env, _packages_path_arg_posix))
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


# Skip not installed #####################################################################

from cobaya.install import NotInstalledError


@pytest.fixture
def skip_not_installed(request):
    return request.config.getoption("--skip-not-installed")


def install_test_wrapper(skip_not_installed, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except NotInstalledError:
        if skip_not_installed:
            pytest.xfail("Missing dependencies.")
        raise
