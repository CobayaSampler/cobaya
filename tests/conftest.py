import logging

import pytest
import os
from cobaya.component import ComponentNotInstalledError
from cobaya.log import LoggedError
from cobaya import mpi

# Paths ##################################################################################

from cobaya.conventions import packages_path_env, packages_path_arg_posix, \
    test_skip_env
from cobaya.tools import resolve_packages_path


def pytest_addoption(parser):
    parser.addoption("--" + packages_path_arg_posix, action="store",
                     default=resolve_packages_path(),
                     help="Path to folder of automatic installation of packages")
    parser.addoption("--skip-not-installed", action="store_true",
                     help="Skip tests for which dependencies of used components are not "
                          "installed.")


@pytest.fixture
def packages_path(request):
    cmd_packages_path = request.config.getoption("--" + packages_path_arg_posix, None)
    if not cmd_packages_path:
        raise ValueError("Could not determine packages installation path. "
                         "Either define it in the env variable %r, or pass it as an "
                         "argument with `--%s`" %
                         (packages_path_env, packages_path_arg_posix))
    return cmd_packages_path


# Skip certain keywords ##################################################################

def pytest_collection_modifyitems(config, items):
    skip_keywords = os.environ.get(test_skip_env, "").replace(",", " ").split()
    for k in skip_keywords:
        skip_mark = pytest.mark.skip(
            reason="'%s' skipped by envvar '%s'" % (k, test_skip_env))
        for item in items:
            if any((k.lower() in x) for x in [item.name.lower(), item.keywords]):
                item.add_marker(skip_mark)
    if not mpi.more_than_one_process():
        skip_not_mpi = pytest.mark.skip(reason="not MPI")
        for item in items:
            if "mpionly" in item.keywords:
                item.add_marker(skip_not_mpi)


# Skip not installed #####################################################################

@pytest.fixture
def skip_not_installed(request):
    return request.config.getoption("--skip-not-installed")


def install_test_wrapper(skip_not_installed, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except ComponentNotInstalledError:
        if skip_not_installed:
            pytest.xfail("Missing dependencies.")
        raise


if mpi.more_than_one_process():
    @pytest.fixture(scope="session", autouse=True)
    def mpi_handling(request):
        mpi.capture_manager = request.config.pluginmanager.getplugin("capturemanager")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" and rep.failed:
        rep.sections = [i for i in rep.sections if i[0] != "Captured log call"]
        if call.excinfo and isinstance(call.excinfo.value, mpi.OtherProcessError):
            # Only need very short message; stdout printed in raised-error process
            rep.longrepr = str(call.excinfo.value)
            rep.sections = []
        elif call.excinfo and isinstance(call.excinfo.value, LoggedError):
            # Don't show call stack, do show output (but log is already printed)
            if logging.root.getEffectiveLevel() > logging.DEBUG and rep.longrepr:
                rep.longrepr = str(rep.longrepr).split("\n")[-1]

    return rep


# use one shared tmpdir
if mpi.is_main_process():
    @pytest.fixture
    def tmpdir(tmpdir):
        return mpi.share_mpi(str(tmpdir))
else:
    @pytest.fixture
    def tmpdir():
        return mpi.share_mpi()


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "mpi: mark test explicitly supports mpi"
    )
    config.addinivalue_line(
        "markers", "mpionly: mark test to only run when run with more than one process"
    )
