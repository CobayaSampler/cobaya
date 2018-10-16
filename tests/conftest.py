import pytest
import os


# Paths ###################################################################################

def pytest_addoption(parser):
    parser.addoption("--modules", action="store", default=None,
                     help="Path to folder of automatic installation of modules")


@pytest.fixture
def modules(request):
    cmd_modules = request.config.getoption("--modules", None)
    # By default, check in cobaya/modules/tests
    cmd_modules = (cmd_modules or
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))
    return cmd_modules
