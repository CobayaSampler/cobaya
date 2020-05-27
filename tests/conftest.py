import pytest
import os
import inspect

# Paths ##################################################################################

from cobaya.conventions import _packages_path_env, _packages_path_arg_posix, \
    _test_skip_env
from cobaya.tools import resolve_packages_path


def pytest_addoption(parser):
    parser.addoption("--" + _packages_path_arg_posix, action="store",
                     default=resolve_packages_path(),
                     help="Path to folder of automatic installation of packages")
    parser.addoption("--skip-not-installed", action="store_true",
                     help="Skip tests for which dependencies of used components are not "
                     "installed.")


@pytest.fixture
def packages_path(request):
    cmd_packages_path = request.config.getoption("--" + _packages_path_arg_posix, None)
    if not cmd_packages_path:
        raise ValueError("Could not determine packages installation path. "
                         "Either define it in the env variable %r, or pass it as an "
                         "argument with `--%s`" %
                         (_packages_path_env, _packages_path_arg_posix))
    return cmd_packages_path


# Skip certain keywords ##################################################################

def pytest_collection_modifyitems(config, items):
    skip_keywords = os.environ.get(_test_skip_env, "").replace(",", " ").split()
    for k in skip_keywords:
        skip_mark = pytest.mark.skip(
            reason="'%s' skipped by envvar '%s'" % (k, _test_skip_env))
        for item in items:
            if any([(k.lower() in x) for x in [item.name.lower(), item.keywords]]):
                item.add_marker(skip_mark)


# Skip not installed #####################################################################

from cobaya.input import get_used_components, get_class
from cobaya.conventions import _component_path, _external


class NotInstalledError(Exception):
    pass


@pytest.fixture
def skip_not_installed(request):
    return request.config.getoption("--skip-not-installed")


def check_installed(info, packages_path=None, skip_not_installed=False):
    for kind, components in get_used_components(info).items():
        for component in components:
            this_info = info[kind][component] or {}
            if isinstance(this_info, str) or callable(this_info) \
               or inspect.isclass(this_info) or _external in this_info:
                # Custom function -- nothing to do
                continue
            try:
                imported_class = get_class(component, kind,
                                           component_path=info.pop(_component_path, None))
            except ImportError as e:
                raise ValueError("Component %s:%s not recognized [%s]." % (
                    kind, component, str(e)))
            is_installed = getattr(imported_class, "is_installed", None)
            if is_installed is None:
                # Built-in component
                continue
            install_path = packages_path
            get_path = getattr(imported_class, "get_path", None)
            if get_path and packages_path:
                install_path = get_path(install_path)
            if not is_installed(path=install_path, allow_global=True):
                if skip_not_installed:
                    pytest.xfail("Missing dependencies: %s:%s" % (kind, component))
                else:
                    raise NotInstalledError
