import sys
import os
from contextlib import contextmanager


def is_travis():
    return os.environ.get('TRAVIS') == 'true'


def process_packages_path(packages_path):
    packages_path = os.getenv('COBAYA_FORCE_PACKAGES_PATH', packages_path)
    if not packages_path:
        if os.path.exists(os.path.join(os.getcwd(), '..', 'packages_path')):
            packages_path = os.path.join('..', 'packages_path')
    assert packages_path, "I need a packages folder!"
    return (packages_path if os.path.isabs(packages_path)
            else os.path.join(os.getcwd(), packages_path))


@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
