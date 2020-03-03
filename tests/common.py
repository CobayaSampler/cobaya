import sys
import os
from contextlib import contextmanager


def is_travis():
    return os.environ.get('TRAVIS') == 'true'


def process_modules_path(modules):
    modules = os.getenv('COBAYA_FORCE_MODULES_PATH', modules)
    if not modules:
        if os.path.exists(os.path.join(os.getcwd(), '..', 'modules')):
            modules = os.path.join('..', 'modules')
    assert modules, "I need a modules folder!"
    return modules if os.path.isabs(modules) else os.path.join(os.getcwd(), modules)


@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
