from __future__ import division, absolute_import, print_function
import sys
import os
from contextlib import contextmanager


def process_modules_path(modules):
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
