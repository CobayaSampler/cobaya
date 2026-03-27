import os
import sys
from contextlib import contextmanager
from io import StringIO

import pytest

from cobaya.component import ComponentNotInstalledError
from cobaya.conventions import packages_path_arg_posix, packages_path_env
from cobaya.tools import resolve_packages_path


def is_ci_test():
    return os.environ.get("GITHUB_ACTIONS") == "true"


def process_packages_path(packages_path) -> str:
    packages_path = os.getenv("COBAYA_FORCE_PACKAGES_PATH", packages_path)
    if not packages_path:
        if os.path.exists(os.path.join(os.getcwd(), "..", "packages_path")):
            packages_path = os.path.join("..", "packages_path")
    assert packages_path, "I need a packages folder!"
    return (
        packages_path
        if os.path.isabs(packages_path)
        else os.path.join(os.getcwd(), packages_path)
    )


@contextmanager
def stdout_redirector(stream):
    stream.seek(0)
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout


@contextmanager
def stdout_check(*strs, match=True):
    stream = StringIO()
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
    output = stream.getvalue()
    for s in strs:
        assert match == (s in output), f"Output should contain '{s}'"


# pytest: deal with packages_path ########################################################

def pytest_addoption_packages_path(parser):
    parser.addoption(
        "--" + packages_path_arg_posix,
        action="store",
        default=resolve_packages_path(),
        help="Path to folder of automatic installation of packages",
    )


@pytest.fixture
def packages_path(request):
    cmd_packages_path = request.config.getoption("--" + packages_path_arg_posix, None)
    if not cmd_packages_path:
        raise ValueError(
            "Could not determine packages installation path. "
            f"Either define it in the env variable {packages_path_env!r}, or pass it as an "
            f"argument with `--{packages_path_arg_posix}`"
        )
    return cmd_packages_path


# pytest: skip not installed #############################################################

def pytest_addoption_skip_not_installed(parser):
    parser.addoption(
        "--skip-not-installed",
        action="store_true",
        help="Skip tests for which dependencies of used components are not installed.",
    )


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
