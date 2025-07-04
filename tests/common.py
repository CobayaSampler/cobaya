import os
import sys
from contextlib import contextmanager
from io import StringIO


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
        assert match == (s in output), "Output should contain '%s'" % s
