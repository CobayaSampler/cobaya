import pytest
import os

# Paths ###################################################################################

def pytest_addoption(parser):
    parser.addoption("--modules", action="store", default=None,
        help="Path to folder of automatic installation of modules")
