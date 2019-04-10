import pytest
import os


# Paths ##################################################################################

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


# Skip certain keywords ##################################################################

def pytest_collection_modifyitems(config, items):
    _env_var = "COBAYA_TEST_SKIP"
    skip_keywords = os.environ.get(_env_var, "").replace(","," ").split()
    for k in skip_keywords:
        skip_mark = pytest.mark.skip(reason="'%s' skipped by envvar '%s'"%(k, _env_var))
        for item in items:
            if any([(k.lower() in x) for x in [item.name.lower(), item.keywords]]):
                item.add_marker(skip_mark)
