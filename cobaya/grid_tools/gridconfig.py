"""
.. module:: cobaya.grid_tools.gridconfig

:Synopsis: Grid creator (Cobaya version)
:Author: Antony Lewis and Jesus Torrado
         (based on Antony Lewis' CosmoMC version of the same code)

"""

import argparse
import importlib.util
import os
from itertools import chain

from getdist.inifile import IniFile
from getdist.paramnames import makeList as make_list

from cobaya.conventions import Extension, packages_path_input
from cobaya.cosmo_input import (
    create_input,
    get_best_covmat_ext,
    get_covmat_package_folders,
)
from cobaya.grid_tools import batchjob
from cobaya.input import merge_info, update_info
from cobaya.install import install as install_reqs
from cobaya.parameterization import is_sampled_param
from cobaya.tools import resolve_packages_path, sort_cosmetic, warn_deprecation
from cobaya.yaml import yaml_dump_file, yaml_load_file


def get_args(vals=None):
    parser = argparse.ArgumentParser(
        prog="cobaya-grid-create", description="Initialize grid using settings file"
    )
    parser.add_argument(
        "batchPath",
        help=(
            "root directory containing/to contain the grid "
            "(e.g. grid_folder where output directories are created "
            "at grid_folder/base/base_xx)"
        ),
    )
    parser.add_argument(
        "settingName",
        nargs="?",
        help=(
            "python setting file for making or updating a grid, a py filename or "
            "full name of a python module"
        ),
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help=("option to configure an already-run existing grid"),
    )
    # Arguments related to installation of requisites
    parser.add_argument(
        "--install",
        action="store_true",
        help=("install required code and data for the grid using default."),
    )
    parser.add_argument(
        "--install-reqs-at",
        help=("install required code and data for the grid in the given folder."),
    )
    parser.add_argument(
        "--install-reqs-force",
        action="store_true",
        default=False,
        help="Force re-installation of apparently installed packages.",
    )
    parser.add_argument(
        "--show-covmats",
        action="store_true",
        help="Show which covmat is assigned to each chain.",
    )

    return parser.parse_args(vals)


def path_is_grid(batchPath):
    return os.path.exists(batchjob.grid_cache_file(batchPath)) or os.path.exists(
        os.path.join(batchPath, "config", "config.ini")
    )


def grid_create(args=None):
    warn_deprecation()
    args = get_args(args)
    args.interactive = True
    makeGrid(**args.__dict__)


def import_from_path(full_path):
    # Create a module spec from the full path
    spec = importlib.util.spec_from_file_location(
        os.path.splitext(os.path.basename(full_path))[0], full_path
    )
    # Create a module from the spec
    module = importlib.util.module_from_spec(spec)
    # Execute the module to populate it
    spec.loader.exec_module(module)
    return module


def post_merge_info(*infos):
    # merge contents of add and remove, or if neither, assume should be options for "add"
    adds = []
    removes = []
    result = {}
    for info in infos:
        inf = info.copy()
        if "add" in info:
            adds.append(inf.pop("add"))
        if "remove" in info:
            removes.append(inf.pop("remove"))
        if len(inf) == len(info):
            adds.append(inf)
        else:
            result.update(inf)
    if adds:
        result["add"] = merge_info(*adds)
    if removes:
        result["remove"] = merge_info(*removes)
    return result


def set_minimize(info, minimize_info=None):
    result = dict(info, sampler={"minimize": minimize_info}, force=True)
    result.pop("resume", None)
    return result


def makeGrid(
    batchPath,
    settingName=None,
    settings=None,
    read_only=False,
    interactive=False,
    install=False,
    install_reqs_at=None,
    install_reqs_force=None,
    show_covmats=False,
):
    print("Generating grid...")
    batchPath = os.path.abspath(batchPath) + os.sep
    if not settings:
        if not settingName:
            if not path_is_grid(batchPath):
                raise Exception(
                    "Need to give name of setting file if batchPath/config does not exist"
                )
            read_only = True
            settingName = IniFile(
                os.path.join(batchPath + "config", "config.ini")
            ).params["setting_file"]
            settingName = os.path.join(batchPath + "config", settingName)
            if settingName.endswith(".py"):
                settings = import_from_path(settingName)
            else:
                settings = yaml_load_file(settingName)
        elif os.path.splitext(settingName)[-1].lower() in Extension.yamls:
            settings = yaml_load_file(settingName)
        elif settingName.endswith(".py"):
            settings = import_from_path(settingName)
        else:
            settings = __import__(settingName, fromlist=["dummy"])
            settingName = settings.__file__
    batch = batchjob.BatchJob(batchPath)
    batch.make_items(settings, messages=not read_only)
    if read_only:
        for job_item in batch.jobItems.copy():
            if not job_item.chainExists():
                batch.jobItems.remove(job_item)
        batch.save()
        print("OK, configured grid with %u existing chains" % (len(batch.jobItems)))
        return batch
    else:
        batch.make_directories(settingName or settings.__file__)
        batch.save()
    infos = {}

    from_yaml = isinstance(settings, dict)
    dic = settings if from_yaml else settings.__dict__
    yaml_dir = dic.get("yaml_dir") or ""
    if "start_at_bestfit" in dic:
        raise ValueError("start_at_bestfit not yet implemented")

    def dicts_or_load(_infos):
        if not _infos or isinstance(_infos, dict):
            return [_infos or {}]
        return [
            (
                yaml_load_file(os.path.join(yaml_dir, _info))
                if isinstance(_info, str)
                else _info
            )
            for _info in _infos
        ]

    def dict_option(_name):
        s = dic.get(_name) or {}
        if isinstance(s, str):
            return yaml_load_file(os.path.join(yaml_dir, s))
        return s

    defaults = merge_info(*dicts_or_load(dic.get("defaults")))
    importance_defaults = merge_info(*dicts_or_load(dic.get("importance_defaults")))
    minimize_defaults = merge_info(*dicts_or_load(dic.get("minimize_defaults")))
    params = dict_option("params")
    param_extra = dict_option("param_extra_opts")
    install = install or install_reqs_at

    components_infos = {}
    for job_item in batch.items(wantSubItems=False):
        # Model info
        job_item.makeChainPath()
        if (model_info := job_item.model_info) is None:
            model_info = {"params": {}}
            for par in job_item.param_set:
                if par not in params:
                    raise ValueError("params[%s] must be defined." % par)
                model_info["params"][par] = params[par]
            extra = dict(param_extra, **job_item.param_extra_opts)
            if opts := extra.get(job_item.paramtag):
                extra_infos = [opts]
            else:
                extra_infos = [extra[par] for par in job_item.param_set if par in extra]
            model_info = merge_info(job_item.defaults, model_info, *extra_infos)

        data_infos = dicts_or_load(job_item.data_set.infos)
        combined_info = merge_info(defaults, model_info, *data_infos)
        if "preset" in combined_info:
            preset = combined_info.pop("preset")
            combined_info = merge_info(create_input(**preset), combined_info)
        combined_info["output"] = job_item.chainRoot
        # Requisites
        if install_reqs_at:
            combined_info[packages_path_input] = os.path.abspath(install_reqs_at)
        # Save the info (we will write it after installation:
        # we need to install to add auto covmats
        if job_item.paramtag not in infos:
            infos[job_item.paramtag] = {}
        infos[job_item.paramtag][job_item.data_set.tag] = combined_info
        components_infos = merge_info(components_infos, combined_info)
    # Installing requisites
    if install:
        print("Installing required code and data for the grid.")
        from cobaya.log import logger_setup

        logger_setup()
        install_reqs(components_infos, path=install_reqs_at, force=install_reqs_force)
    print("Adding covmats (if necessary) and writing input files")
    cov_dir = dic.get("cov_dir")  # None means use the default from mcmc settings
    def_packages = cov_dir or install_reqs_at or resolve_packages_path()
    for job_item in batch.items(wantSubItems=False):
        info = infos[job_item.paramtag][job_item.data_set.tag]
        # Covariance matrices
        # We try to find them now, instead of at run time, to check if correctly selected
        try:
            sampler = list(info["sampler"])[0]
        except KeyError:
            raise ValueError("No sampler has been chosen: %s" % job_item.name)
        if sampler == "mcmc" and (
            cov_dir
            or cov_dir is None
            and info["sampler"][sampler].get("covmat") == "auto"
        ):
            if not (cov_dirs := make_list(cov_dir or [])):
                if not (
                    packages_path := install_reqs_at
                    or info.get(packages_path_input)
                    or def_packages
                ):
                    raise ValueError(
                        "Cannot assign automatic covariance matrices because no "
                        "external packages path has been defined."
                    )
                cov_dirs = get_covmat_package_folders(os.path.abspath(packages_path))
            # Need updated info for covmats: includes renames
            updated_info = update_info(info)
            # Ideally, we use slow+sampled parameters to look for the covariance matrix
            # but since for that we'd need to initialise a model, we approximate that set
            # as theory+sampled
            like_params = set(
                chain(
                    *[
                        list(like.get("params") or [])
                        for like in updated_info["likelihood"].values()
                    ]
                )
            )
            params_info = {
                p: v
                for p, v in updated_info["params"].items()
                if is_sampled_param(v) and p not in like_params
            }

            best_covmat = get_best_covmat_ext(
                cov_dirs,
                params_info,
                updated_info["likelihood"],
                job_item=job_item,
                cov_map=dic.get("cov_map") or {},
            )
            info["sampler"][sampler]["covmat"] = (
                os.path.join(best_covmat["folder"], best_covmat["name"])
                if best_covmat
                else None
            )
            if show_covmats:
                print(job_item.name, "->", (best_covmat or {}).get("name"))

        # Write the info for this job
        # Allow overwrite since often will want to regenerate grid with tweaks
        info = sort_cosmetic(info)
        yaml_dump_file(job_item.yaml_file(), info, error_if_exists=False)

        # Minimize
        info = set_minimize(info, minimize_defaults)
        yaml_dump_file(job_item.yaml_file("_minimize"), info, error_if_exists=False)

        # Importance sampling
        for imp in job_item.importanceJobs():
            if getattr(imp, "importanceFilter", None):
                continue
            if batch.hasName(imp.name.replace(".post.", "_")):
                raise Exception("importance sampling something you already have?")
            info_post = {
                "output": job_item.chainRoot,
                "post": post_merge_info(
                    importance_defaults, *dicts_or_load(imp.importanceSettings)
                ),
                "force": True,
            }
            info_post["post"]["suffix"] = imp.importanceTag
            yaml_dump_file(imp.yaml_file(), info_post, error_if_exists=False)
            if getattr(imp, "want_minimize", True):
                info = set_minimize(dict(info, **info_post), minimize_defaults)
                yaml_dump_file(imp.yaml_file("_minimize"), info, error_if_exists=False)

    if not interactive:
        return batch
    print("Done... to run do: cobaya-grid-run %s" % batchPath)
    print("....... for best fits: cobaya-grid-run %s --minimize" % batchPath)
    print("For importance sampled: cobaya-grid-run %s --importance" % batchPath)
    print(
        "for best-fit for importance sampled: "
        "cobaya-grid-run %s --importance_minimize" % batchPath
    )
