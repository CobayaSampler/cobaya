import logging
import os

import pytest

import cobaya.typing
from cobaya import mpi

# Paths ##################################################################################
from cobaya.conventions import test_skip_env
from cobaya.log import LoggedError
from cobaya.test_tools import packages_path, skip_not_installed

cobaya.typing.enforce_type_checking = True


def pytest_addoption(parser):
    # These imports need to be in here!
    from cobaya.test_tools import (
        pytest_addoption_packages_path,
        pytest_addoption_skip_not_installed,
    )

    pytest_addoption_packages_path(parser)
    pytest_addoption_skip_not_installed(parser)
    parser.addoption(
        "--do-plots",
        action="store_true",
        help=(
            "Does some plots to check the results of some tests visually. "
            "Plots are stored in the (tmp?) folder of the test results."
        ),
    )


# Skip certain keywords ##################################################################

def pytest_collection_modifyitems(config, items):
    skip_keywords = os.environ.get(test_skip_env, "").replace(",", " ").split()
    for k in skip_keywords:
        skip_mark = pytest.mark.skip(reason=f"'{k}' skipped by envvar '{test_skip_env}'")
        for item in items:
            if any((k.lower() in x) for x in [item.name.lower(), item.keywords]):
                item.add_marker(skip_mark)
    if not mpi.more_than_one_process():
        skip_not_mpi = pytest.mark.skip(reason="not MPI")
        for item in items:
            if "mpionly" in item.keywords:
                item.add_marker(skip_not_mpi)


# Plots from tests #######################################################################

@pytest.fixture
def do_plots(request):
    return request.config.getoption("--do-plots")


# Other MPI-related ######################################################################

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
    config.addinivalue_line("markers", "mpi: mark test explicitly supports mpi")
    config.addinivalue_line(
        "markers", "mpionly: mark test to only run when run with more than one process"
    )
