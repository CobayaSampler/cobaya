"""
.. module:: install

:Synopsis: Tools and script to install the external code and data packages needed by
           the Cobaya components to be used.
:Author: Jesus Torrado

"""

import importlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Mapping
from itertools import chain

import requests  # type: ignore
import tqdm  # type: ignore
from packaging import version

from cobaya.component import ComponentNotFoundError, get_component_class
from cobaya.conventions import (
    Extension,
    code_path,
    data_path,
    install_skip_env,
    packages_path_arg,
    packages_path_arg_posix,
    packages_path_config_file,
    packages_path_env,
    packages_path_input,
)
from cobaya.input import get_used_components
from cobaya.log import LoggedError, NoLogging, get_logger, logger_setup
from cobaya.mpi import set_mpi_disabled
from cobaya.tools import (
    VersionCheckError,
    create_banner,
    find_with_regexp,
    get_config_path,
    resolve_packages_path,
    similar_internal_class_names,
    warn_deprecation,
    write_packages_path_in_config_file,
)
from cobaya.typing import InfoDict, InputDict
from cobaya.yaml import yaml_load_file

_banner_symbol = "="
_banner_length = 80
_version_filename = "version.dat"


def do_package_install(
    component: str, package_install: InfoDict | str, full_code_path: str, logger
):
    # attempt to install package if package_install download info is present
    # similar to InstallableLikelihood install_options (could be refactored)
    component_root = component.split(".")[0]
    directory = package_install.get("directory")
    min_version = package_install.get("min_version")
    if package_install == "pip":
        package_install = {"pip": None}
    elif not isinstance(package_install, Mapping):
        raise LoggedError(
            logger,
            "Invalid package_install: must be 'pip' or a "
            "dictionary with download_url or github_repository",
        )
    package = package_install.get("pip")
    if package is None:
        package = component_root
    cwd = None
    if repo := package_install.get("github_repository"):
        logger.info("Downloading code from github (%s)", repo)
        directory = directory or repo.split("/")[-1]
        if not download_github_release(
            full_code_path,
            repo,
            package_install.get("github_release"),
            directory=directory,
            logger=logger,
        ):
            return False
        cwd = os.path.join(full_code_path, directory)
        package = "."
    elif url := package_install.get("download_url"):
        logger.info("Downloading code from %s", url)
        cwd = os.path.join(full_code_path, directory or component_root)
        if not download_file(url, cwd, logger=logger):
            return False
        if not (
            paths := find_with_regexp("pyproject\\.toml", cwd, True)
            or find_with_regexp("setup\\.py", cwd, True)
        ):
            raise LoggedError(
                logger, "No setup.py or pyproject.toml found in %s for %s", cwd, component
            )
        cwd = os.path.dirname(paths[0])
        package = "."
    elif "pip" not in package_install:
        raise LoggedError(
            logger,
            "Invalid package_install: must define pip, github_repository or download_url",
        )
    if min_version is not None and package != ".":
        package += f">={min_version}"
    logger.info("pip installing %s", package)
    return not pip_install(package, upgrade=True, logger=logger, cwd=cwd)


def install(*infos, **kwargs):
    """
    Installs the external packages required by the components mentioned in ``infos``.

    ``infos`` can be input dictionaries, single component names, or names of yaml files

    :param force: force re-installation of apparently installed packages (default:
       ``False``).
    :param test: just check whether components are installed  (default: ``False``).
    :param upgrade: force upgrade of obsolete components (default: ``False``).
    :param skip: keywords of components that will be skipped during installation.
    :param skip_global: skip installation of already-available Python modules (default:
       ``False``).
    :param debug: produce verbose debug output  (default: ``False``).
    :param path: optional path where to install the packages
                 (defaults to any packages_path entry given in the info dictionaries).
    :param code: set to ``False`` to skip code packages (default: ``True``).
    :param data: set to ``False`` to skip data packages (default: ``True``).
    :param no_progress_bars: no progress bars shown; use when output is saved into a text
       file (e.g. when running on a cluster) (default: ``False``).
    :param no_set_global: do not store the installation path for later runs (default:
       ``False``).

    :return: ``True`` if all components are installed without skips, ``False`` if
                  some components skipped, None (or exception raised) otherwise
    """
    debug = kwargs.get("debug", False)
    logger = kwargs.get("logger")
    if not logger:
        logger_setup(debug=debug)
        logger = get_logger("install")
    path = kwargs.get("path")
    infos = [
        yaml_load_file(info)
        if isinstance(info, str) and info.split(".")[-1] in Extension.yamls
        else info
        for info in infos
    ]
    infos_not_single_names = [info for info in infos if isinstance(info, Mapping)]
    if not (path := path or resolve_packages_path(*infos_not_single_names)):
        raise LoggedError(
            logger,
            (
                "No 'path' argument given, and none could be found in input infos "
                "(as %r), the %r env variable or the config file. "
                "Maybe specify one via a command line argument '-%s [...]'?"
            ),
            packages_path_input,
            packages_path_env,
            packages_path_arg[0],
        )
    # General install path for all dependencies
    general_abspath = os.path.abspath(path)
    logger.info("Installing external packages at '%s'", general_abspath)
    # Set the installation path in the global config file
    if not kwargs.get("no_set_global") and not kwargs.get("test"):
        write_packages_path_in_config_file(general_abspath)
        logger.info(
            "The installation path has been written into the global config file: %s",
            os.path.join(get_config_path(), packages_path_config_file),
        )
    kwargs_install = {
        "force": kwargs.get("force", False),
        "no_progress_bars": kwargs.get("no_progress_bars"),
        "show_error": False,
    }
    for what in (code_path, data_path):
        kwargs_install[what] = kwargs.get(what, True)
        spath = os.path.join(general_abspath, what)
        if kwargs_install[what] and not os.path.exists(spath):
            try:
                os.makedirs(spath)
            except OSError:
                raise LoggedError(
                    logger, f"Could not create the desired installation folder '{spath}'"
                )
    # To check e.g. for a version upgrade, it needs to reload the component class and
    # all relevant imported modules: the implementation of `is_installed` for each
    # class is expected to always reload external modules if passed `reload=True`
    # (should be False by default to avoid deleting objects unnecessarily).
    kwargs_is_installed = {"reload": True}
    unknown_components = []  # could not be identified
    failed_components = []  # general errors
    obsolete_components = []  # older or unknown version already installed
    incompatible_components = []  # not compatible with current OS
    skip_keywords_arg = set(kwargs.get("skip") or [])
    # NB: if passed with quotes as `--skip "a b"`, it's interpreted as a single key
    skip_keywords_arg = set(chain(*[word.split() for word in skip_keywords_arg]))
    skip_keywords_env = set(
        os.environ.get(install_skip_env, "").replace(",", " ").lower().split()
    )
    skip_keywords = skip_keywords_arg.union(skip_keywords_env)
    skipped = False
    # Combine all requested components and install them
    # NB: components mentioned by name may be repeated with those given in dict infos.
    #     That's OK, because the install check will skip them in the 2nd pass
    used_components, components_infos = get_used_components(*infos, return_infos=True)
    for kind, components in used_components.items():
        for component in components:
            name_w_kind = (kind + ":" if kind else "") + component
            print()
            print(
                create_banner(name_w_kind, symbol=_banner_symbol, length=_banner_length),
                end="",
            )
            print()
            if _skip_helper(component.lower(), skip_keywords, skip_keywords_env, logger):
                skipped = True
                continue
            info = components_infos[component]
            if isinstance(info, str) or "external" in info:
                logger.info(
                    f"Component '{name_w_kind}' is a custom function. Nothing to do."
                )
                continue
            class_name = (info or {}).get("class")
            if class_name:
                logger.info(f"Class to be installed for this component: {class_name}")
            python_path = info.pop("python_path", None)
            package_install = info.get("package_install") or {}
            # If a class_name AND pip install were specified, make sure the class name is
            # used. Otherwise the pip package would be inferred from the alias.
            if package_install == "pip":
                package_install = {"pip": None}
            if "pip" in package_install and package_install["pip"] is None:
                package_install["pip"] = class_name
            min_version = package_install.get("min_version")

            def _imported_class():
                return get_component_class(
                    component,
                    kind=kind,
                    component_path=python_path,
                    class_name=class_name,
                    logger=logger,
                    min_package_version=min_version,
                    allow_internal=not package_install,
                )

            try:
                try:
                    if package_install and (
                        kwargs_install["force"] or kwargs.get("upgrade")
                    ):
                        raise ComponentNotFoundError(logger)
                    imported_class = _imported_class()
                except (ComponentNotFoundError, VersionCheckError):
                    if package_install:
                        if do_package_install(
                            component,
                            package_install,
                            os.path.join(general_abspath, code_path),
                            logger,
                        ):
                            importlib.invalidate_caches()
                            imported_class = _imported_class()
                        else:
                            logger.error(
                                f"Package install failed for '{name_w_kind}'. Skipping."
                            )
                            unknown_components += [name_w_kind]
                            continue
                    else:
                        raise
                except Exception:
                    raise
            except ComponentNotFoundError:
                logger.error(
                    f"Component '{name_w_kind}' could not be identified. Skipping."
                )
                unknown_components += [name_w_kind]
                continue
            except Exception:
                traceback.print_exception(*sys.exc_info(), file=sys.stdout)
                logger.error(f"An error occurred when loading '{name_w_kind}'. Skipping.")
                failed_components += [name_w_kind]
                continue
            else:
                if _skip_helper(
                    imported_class.__name__.lower(),
                    skip_keywords,
                    skip_keywords_env,
                    logger,
                ):
                    skipped = True
                    continue
            # Update the name if the kind was unknown
            if not kind:
                name_w_kind = imported_class.get_kind() + ":" + component
            is_compatible = getattr(imported_class, "is_compatible", lambda: True)()
            if not is_compatible:
                logger.info(
                    f"Skipping '{name_w_kind}' because it is not compatible with your OS."
                )
                incompatible_components += [name_w_kind]
                continue
            logger.info("Checking if dependencies have already been installed...")
            is_installed = getattr(imported_class, "is_installed", None)
            if is_installed is None:
                logger.info(
                    f"Component '{name_w_kind}' is a fully built-in component: "
                    "nothing to do."
                )
                continue
            this_component_install_path = general_abspath
            get_path = getattr(imported_class, "get_path", None)
            if get_path:
                this_component_install_path = get_path(this_component_install_path)
            # Check previous installations and their versions
            has_been_installed = False
            is_old_version_msg = None
            with NoLogging(None if debug else logging.ERROR):
                try:
                    if kwargs.get("skip_global"):
                        has_been_installed = is_installed(
                            path="global", **kwargs_install, **kwargs_is_installed
                        )
                    if not has_been_installed:
                        has_been_installed = is_installed(
                            path=this_component_install_path,
                            **kwargs_install,
                            **kwargs_is_installed,
                        )
                except VersionCheckError as excpt:
                    is_old_version_msg = str(excpt)
            if has_been_installed:  # no VersionCheckError was raised
                logger.info("External dependencies for this component already installed.")
                if kwargs.get("test"):
                    continue
                if kwargs_install["force"] and not kwargs.get("skip_global"):
                    logger.info("Forcing re-installation, as requested.")
                else:
                    logger.info("Doing nothing.")
                    continue
            elif is_old_version_msg:
                logger.info(f"Version check failed: {is_old_version_msg}")
                if kwargs.get("test"):
                    continue
                if not kwargs.get("upgrade") and not kwargs.get("force"):
                    logger.info("Skipping because '--upgrade' not requested.")
                    obsolete_components += [name_w_kind]
                    continue
            else:
                logger.info("Check found no existing installation")
                if not debug:
                    logger.info(
                        "(If you expected this to be already installed, re-run "
                        "`cobaya-install` with --debug to get more verbose output.)"
                    )
                if kwargs.get("test"):
                    # We are only testing whether it was installed, so consider it failed
                    failed_components += [name_w_kind]
                    continue
            # Do the install
            logger.info("Installing...")
            install_args = dict(kwargs_install, show_error=True)
            try:
                install_this = getattr(imported_class, "install", None)
                success = install_this(path=general_abspath, **install_args)
            except Exception:
                traceback.print_exception(*sys.exc_info(), file=sys.stdout)
                logger.error(
                    "An unknown error occurred. Delete the external packages "
                    "folder %r and try again. "
                    "Please, notify the developers if this error persists.",
                    general_abspath,
                )
                success = False
            if success:
                logger.info("Successfully installed! Let's check it...")
            else:
                logger.error(
                    "Installation failed! Look at the error messages above. "
                    "Solve them and try again, or, if you are unable to solve them, "
                    "install the packages required by this component manually."
                )
                failed_components += [name_w_kind]
                continue
            # Test installation
            reloaded_class = get_component_class(
                component,
                kind=kind,
                component_path=info.pop("python_path", None),
                class_name=class_name,
                logger=logger,
            )
            reloaded_is_installed = getattr(reloaded_class, "is_installed", None)
            with NoLogging(None if debug else logging.ERROR):
                try:
                    successfully_installed = reloaded_is_installed(
                        path=this_component_install_path,
                        **kwargs_install,
                        **kwargs_is_installed,
                    )
                except Exception:
                    traceback.print_exception(*sys.exc_info(), file=sys.stdout)
                    successfully_installed = False
            if not successfully_installed:
                logger.error(
                    "Installation apparently worked, "
                    "but the subsequent installation test failed! "
                    "This does not always mean that there was an actual error, "
                    "and is sometimes fixed simply by running the installer "
                    "again. If not, look closely at the error messages above, "
                    "or re-run with --debug for more verbose output. "
                    "If you are unable to fix the issues above, "
                    "try installing the packages required by this "
                    "component manually."
                )
                failed_components += [name_w_kind]
            else:
                logger.info("Installation check successful.")
    print()
    print(
        create_banner(" * Summary * ", symbol=_banner_symbol, length=_banner_length),
        end="",
    )
    print()
    bullet = "\n - "
    if unknown_components:
        suggestions_dict = {
            name: similar_internal_class_names(name) for name in unknown_components
        }
        suggestions_msg = bullet + bullet.join(
            f"{name}: did you mean any of the following? {sugg} (mind capitalization!)"
            for name, sugg in suggestions_dict.items()
        )
        raise LoggedError(
            logger,
            (
                "The following components could not be identified and were skipped:"
                f"{suggestions_msg}"
            ),
        )
    if failed_components:
        raise LoggedError(
            logger,
            (
                "The installation (or installation test) of some component(s) has "
                "failed: %s\nCheck output of the installer of each component above "
                "for precise error info.\n"
            ),
            bullet + bullet.join(failed_components),
        )
    if obsolete_components:
        raise LoggedError(
            logger,
            (
                "The following packages are obsolete. Re-run with `--upgrade` option"
                " (not upgrading by default to preserve possible user changes): %s"
            ),
            bullet + bullet.join(obsolete_components),
        )
    if incompatible_components:
        logger.warning(
            "The following components were skipped because they are not compatible "
            "with your OS: %s",
            bullet + bullet.join(incompatible_components),
        )
    if not unknown_components and not failed_components and not obsolete_components:
        logger.info(
            "All requested components' dependencies correctly installed at "
            f"{general_abspath}"
        )
        return not skipped


def _skip_helper(name, skip_keywords, skip_keywords_env, logger):
    try:
        this_skip_keyword = next(s for s in skip_keywords if s.lower() in name.lower())
        env_msg = (
            " in env var %r" % install_skip_env
            if this_skip_keyword in skip_keywords_env
            else ""
        )
        logger.info(
            "Skipping %r as per skip keyword %r" + env_msg, name, this_skip_keyword
        )
        return True
    except StopIteration:
        return False


def download_file(url, path, *, size=None, no_progress_bars=False, logger=None):
    """
    Downloads (and optionally decompresses) a file into a given path.

    :param url: url from which to download the file.
    :param path: path to where the file should be downloaded.
    :param size: size in bytes of the file to download; can be used to show percentages
       when the url headers do not show the actual size.
    :param no_progress_bars: no progress bars shown; use when output is saved into a text
       file (e.g. when running on a cluster).
    :param logger: logger to use for reporting information; a new logger is created if not
       specified.
    :return: ``True`` if the download (and decompression, if requested) was successful,
       and ``False`` otherwise.
    """
    logger = logger or get_logger("install")
    with tempfile.TemporaryDirectory() as tmp_path:
        try:
            req = requests.get(url, allow_redirects=True, stream=True)
            # get hinted filename if available:
            try:
                filename = re.findall(
                    "filename=(.+)", req.headers["content-disposition"]
                )[0]
                filename = filename.strip("\"'")
            except KeyError:
                filename = os.path.basename(url)
            filename_tmp_path = os.path.normpath(os.path.join(tmp_path, filename))
            size = size or int(req.headers.get("content-length", 0))
            # Adapted from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
            if not no_progress_bars:
                bar = tqdm.tqdm(total=size, unit="iB", unit_scale=True, unit_divisor=1024)
            with open(filename_tmp_path, "wb") as f:
                for data in req.iter_content(chunk_size=32768):
                    chunk_size = f.write(data)
                    if not no_progress_bars:
                        bar.update(chunk_size)
            if not no_progress_bars:
                bar.close()
            if os.path.getsize(filename_tmp_path) < 1024:  # 1kb
                with open(filename_tmp_path) as f:
                    lines = f.readlines()
                    if lines[0].startswith("404") or "not found" in lines[0].lower():
                        raise ValueError("File not found (404)!")
            logger.info("Downloaded filename %s", filename)
        except Exception as excpt:
            logger.error("Error downloading %s' to folder '%s': %s", url, tmp_path, excpt)
            return False
        logger.debug("Got: %s", filename)
        extension = os.path.splitext(filename)[-1][1:]
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            if extension == "zip":
                from zipfile import ZipFile

                with ZipFile(filename_tmp_path, "r") as zipObj:
                    zipObj.extractall(path)
            else:
                import tarfile

                if extension == "tgz":
                    extension = "gz"
                kwargs = {}
                if sys.version_info >= (3, 12):
                    kwargs["filter"] = "data"
                with tarfile.open(filename_tmp_path, "r:" + extension) as tar:
                    tar.extractall(path, **kwargs)
            logger.debug("Decompressed: %s", filename)
            return True
        except Exception as excpt:
            logger.error("Error decompressing downloaded file! Corrupt file? [%s]", excpt)
            return False


def download_github_release(
    base_directory,
    repo_name,
    release_name=None,
    *,
    asset=None,
    directory=None,
    no_progress_bars=False,
    logger=None,
):
    """
    Downloads a release (i.e. a tagged commit) or a release asset from a GitHub repo.

    :param base_directory: directory into which the release will be downloaded; will be
       created if it does not exist.
    :param repo_name: repository name as ``user/repo``.
    :param release_name: name or tag of the release.
                         If not set, uses latest default branch.
    :param asset: download just an asset (attached file) from a release.
    :param directory: name of the directory that will contain the asset or release, if
       different ``repo_name``.
    :param no_progress_bars: no progress bars shown; use when output is saved into a text
       file (e.g. when running on a cluster).
    :param logger: logger to use for reporting information; a new logger is created if not
       specified.
    :return: ``True`` if the download was successful, and ``False`` otherwise.
    """
    logger = logger or get_logger("install")
    if "/" in repo_name:
        github_user = repo_name[: repo_name.find("/")]
        repo_name = repo_name[repo_name.find("/") + 1 :]
    else:
        github_user = "CobayaSampler"
    base_url = r"https://github.com/" + github_user + "/" + repo_name
    download_directory = base_directory
    if not release_name:
        data = requests.get(
            "https://api.github.com/repos/" + github_user + "/" + repo_name
        ).json()
        release_name = (data or {}).get("default_branch", "main")
    if asset:
        url = base_url + "/releases/download/" + release_name + "/" + asset
        # Asset would get decompressed in base directory
        download_directory = os.path.join(download_directory, directory or repo_name)
    else:
        url = base_url + "/archive/" + release_name + ".tar.gz"

    if not download_file(
        url, download_directory, no_progress_bars=no_progress_bars, logger=logger
    ):
        return False
    # In releases, not assets, remove version number from directory name
    # and rename if requested
    if not asset:
        w_version = next(
            d
            for d in os.listdir(base_directory)
            if (d.startswith(repo_name) and len(d) != len(repo_name))
        )
        actual_download_directory = os.path.join(base_directory, w_version)
        download_directory = os.path.join(base_directory, directory or repo_name)
        if os.path.exists(download_directory):
            shutil.rmtree(download_directory)
        os.rename(actual_download_directory, download_directory)
    # Now save the version into a file to be checked later
    with open(os.path.join(download_directory, _version_filename), "w") as f:
        f.write(release_name)
    logger.info(
        (f"{asset} from " if asset else "")
        + "%s %s downloaded and decompressed correctly.",
        repo_name,
        release_name,
    )
    return True


def pip_install(packages, upgrade=False, logger=None, options=(), **kwargs):
    """
    Takes package name or list of them.

    Returns exit status.
    """
    if hasattr(packages, "split"):
        packages = [packages]
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if upgrade:
        cmd += ["--upgrade"]
    cmd += list(options)
    # Assume that if the user has installed Cobaya on the system-wide Python,
    # then it is OK to overwrite system packages:
    kwargs["env"] = dict(kwargs.get("env", os.environ), PIP_BREAK_SYSTEM_PACKAGES="1")
    res = subprocess.call(cmd + packages, **kwargs)
    if res:
        msg = f"pip: error installing packages '{packages}'"
        logger.error(msg) if logger else print(msg, file=sys.stderr)
    return res


def check_gcc_version(min_version="6.4", error_returns=None):
    """
    Checks the version of the ``gcc`` compiler installed.

    If the installed version is higher than ``min_version``, return ``True``,
    and ``False`` otherwise.

    If an error is produced, returns ``error_returns``.
    """
    try:
        ver = (
            subprocess.check_output(
                "gcc -dumpversion", shell=True, stderr=subprocess.STDOUT
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return error_returns
    # Change in gcc >= 7: -dumpversion only dumps major version
    if "." not in ver:
        ver = (
            subprocess.check_output(
                "gcc -dumpfullversion", shell=True, stderr=subprocess.STDOUT
            )
            .decode()
            .strip()
        )
    return version.parse(str(min_version)) <= version.parse(ver)


# Command-line script ####################################################################


def install_script(args=None):
    """Command line script for the installer."""
    set_mpi_disabled()
    warn_deprecation()
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(
        prog="cobaya install",
        description="Cobaya's installation tool for external packages.",
    )
    parser.add_argument(
        "files_or_components",
        action="store",
        nargs="+",
        metavar="input_file.yaml|component_name",
        help="One or more input files or component names "
        "(or simply 'cosmo' to install all the requisites for basic"
        " cosmological runs)",
    )
    parser.add_argument(
        "-" + packages_path_arg[0],
        "--" + packages_path_arg_posix,
        action="store",
        required=False,
        metavar="/packages/path",
        default=None,
        help="Desired path where to install external packages. "
        "Optional if one has been set globally or as an env variable"
        " (run with '--show_%s' to check)." % packages_path_arg_posix,
    )
    output_show_packages_path = resolve_packages_path()
    if output_show_packages_path and os.environ.get(packages_path_env):
        output_show_packages_path += " (from env variable %r)" % packages_path_env
    elif output_show_packages_path:
        output_show_packages_path += " (from config file)"
    else:
        output_show_packages_path = "(Not currently set.)"
    parser.add_argument(
        "--show-" + packages_path_arg_posix,
        action="version",
        version=output_show_packages_path,
        help="Prints default external packages installation folder and exits.",
    )
    parser.add_argument(
        "-" + "f",
        "--" + "force",
        action="store_true",
        default=False,
        help="Force re-installation of apparently installed packages.",
    )
    parser.add_argument(
        "--%s" % "test",
        action="store_true",
        default=False,
        help="Just check whether components are installed.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        default=False,
        help="Force upgrade of obsolete components.",
    )
    parser.add_argument(
        "--skip",
        action="store",
        nargs="*",
        metavar="keyword",
        help=("Keywords of components that will be skipped during installation."),
    )
    parser.add_argument(
        "--skip-global",
        action="store_true",
        default=False,
        help="Skip installation of already-available Python modules.",
    )
    parser.add_argument(
        "-" + "d",
        "--" + "debug",
        action="store_true",
        help="Produce verbose debug output.",
    )
    group_just = parser.add_mutually_exclusive_group(required=False)
    group_just.add_argument(
        "-C",
        "--just-code",
        action="store_false",
        default=True,
        help="Install code of the components.",
        dest=data_path,
    )
    group_just.add_argument(
        "-D",
        "--just-data",
        action="store_false",
        default=True,
        help="Install data of the components.",
        dest=code_path,
    )
    parser.add_argument(
        "--no-progress-bars",
        action="store_true",
        default=False,
        help=(
            "No progress bars shown; use when output is saved into a "
            "text file (e.g. when running on a cluster)."
        ),
    )
    parser.add_argument(
        "--no-set-global",
        action="store_true",
        default=False,
        help="Do not store the installation path for later runs.",
    )
    arguments = parser.parse_args(args)
    # Configure the logger ASAP
    logger_setup(arguments.debug)
    logger = get_logger("install")
    # Gather requests
    infos: list[InfoDict | InputDict | str] = []
    for f in arguments.files_or_components:
        if f.lower() == "cosmo":
            logger.info("Installing basic cosmological packages.")
            from cobaya.cosmo_input import install_basic

            infos += [install_basic]
        elif f.lower() == "cosmo-tests":
            logger.info("Installing *tested* cosmological packages.")
            from cobaya.cosmo_input import install_tests

            infos += [install_tests]
        elif os.path.splitext(f)[1].lower() in Extension.yamls:
            from cobaya.input import load_input

            infos += [load_input(f)]
        else:  # a single component name, no kind specified
            infos += [f]
    # Launch installer
    install(
        *infos,
        path=getattr(arguments, packages_path_arg),
        logger=logger,
        **{
            arg: getattr(arguments, arg)
            for arg in [
                "force",
                code_path,
                data_path,
                "no_progress_bars",
                "test",
                "no_set_global",
                "skip",
                "skip_global",
                "debug",
                "upgrade",
            ]
        },
    )


if __name__ == "__main__":
    install_script()
