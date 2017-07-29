import pytest
import os

# Paths ###################################################################################

def pytest_addoption(parser):
    parser.addoption("--root", action="store", default=None,
        help="Path to root installation folder")
    parser.addoption("--polychord", action="store", default=None,
        help="Path to PolyChord")
    parser.addoption("--camb", action="store", default=None,
        help="Path to CAMB")
    parser.addoption("--classy", action="store", default=None,
        help="Path to CLASS")
    parser.addoption("--planck", action="store", default=None,
        help="Path to the Planck likelihood")

@pytest.fixture
def polychord_path(request):
    return request.config.getoption("--polychord",
        default=os.path.join(request.config.getoption("--root"), "PolyChord"))
    
@pytest.fixture
def camb_path(request):
    camb_path =request.config.getoption("--camb")
    if not camb_path:
        camb_path = os.path.join(request.config.getoption("--root"), "CAMB")
    return camb_path

@pytest.fixture
def classy_path(request):
    classy_path = request.config.getoption("--classy")
    if not classy_path:
        classy_path = os.path.join(request.config.getoption("--root"), "CLASS")
    return classy_path

@pytest.fixture
def planck_path(request):
    planck_path = request.config.getoption("--planck")
    if not planck_path:
        planck_path = os.path.join(request.config.getoption("--root"),
                                   "likelihoods/planck_2015")
    return planck_path

