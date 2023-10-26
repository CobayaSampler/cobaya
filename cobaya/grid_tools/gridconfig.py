"""
.. module:: cobaya.grid_tools.gridconfig

:Synopsis: Grid creator (Cobaya version)
:Author: Antony Lewis and Jesus Torrado
         (based on Antony Lewis' CosmoMC version of the same code)

"""

# Global
import os
import copy
import argparse
import numpy as np
import importlib.util
from getdist.inifile import IniFile

# Local
from cobaya.yaml import yaml_load_file, yaml_dump_file
from cobaya.conventions import Extension, packages_path_input
from cobaya.input import get_used_components, merge_info, update_info
from cobaya.install import install as install_reqs
from cobaya.tools import sort_cosmetic, warn_deprecation, resolve_packages_path, \
    PythonPath
from cobaya.grid_tools import batchjob
from cobaya.cosmo_input import create_input, get_best_covmat_ext
from cobaya.parameterization import is_sampled_param


def getArgs(vals=None):
    parser = argparse.ArgumentParser(
        prog="cobaya-grid-create",
        description='Initialize grid using settings file')
    parser.add_argument('batchPath', help=(
        'root directory containing/to contain the grid '
        '(e.g. ./PLA where directories base, base_xx etc are under ./PLA)'))
    parser.add_argument('settingName', nargs='?', help=(
        'python setting file (without .py) for making or updating a grid, '
        'usually found as python/settingName.py'))
    parser.add_argument('--read-only', action='store_true', help=(
        'option to configure an already-run existing grid'))
    # Arguments related to installation of requisites
    parser.add_argument('--install-reqs-at', help=(
        'install required code and data for the grid in the given folder.'))
    parser.add_argument("--install-reqs-force", action="store_true", default=False,
                        help="Force re-installation of apparently installed packages.")
    return parser.parse_args(vals)


def path_is_grid(batchPath):
    return os.path.exists(batchjob.grid_cache_file(batchPath)) or os.path.exists(
        os.path.join(batchPath, 'config', 'config.ini'))


def grid_create(args=None):
    warn_deprecation()
    args = getArgs(args)
    args.interactive = True
    makeGrid(**args.__dict__)


def import_from_path(full_path):
    # Create a module spec from the full path
    spec = importlib.util.spec_from_file_location(
        os.path.splitext(os.path.basename(full_path))[0], full_path)
    # Create a module from the spec
    module = importlib.util.module_from_spec(spec)
    # Execute the module to populate it
    spec.loader.exec_module(module)
    return module


def makeGrid(batchPath, settingName=None, settings=None, read_only=False,
             interactive=False, install_reqs_at=None, install_reqs_force=None,
             random_state=None):
    print("Generating grid...")
    batchPath = os.path.abspath(batchPath) + os.sep
    if not settings:
        if not settingName:
            if not path_is_grid(batchPath):
                raise Exception('Need to give name of setting file if batchPath/config '
                                'does not exist')
            read_only = True
            settingName = IniFile(os.path.join(batchPath + 'config',
                                               'config.ini')).params['setting_file']
            settingName = os.path.join(batchPath + 'config', settingName)
            if settingName.endswith('.py'):
                settings = import_from_path(settingName)
            else:
                settings = yaml_load_file(settingName)
        elif os.path.splitext(settingName)[-1].lower() in Extension.yamls:
            settings = yaml_load_file(settingName)
        elif settingName.endswith('.py'):
            settings = import_from_path(settingName)
        else:
            settings = __import__(settingName, fromlist=['dummy'])
            settingName = settings.__file__
    batch = batchjob.BatchJob(batchPath)
    batch.makeItems(settings, messages=not read_only)
    if read_only:
        for jobItem in [b for b in batch.jobItems]:
            if not jobItem.chainExists():
                batch.jobItems.remove(jobItem)
        batch.save()
        print('OK, configured grid with %u existing chains' % (len(batch.jobItems)))
        return batch
    else:
        batch.makeDirectories(settingName or settings.__file__)
        batch.save()
    infos = {}
    components_used = {}
    random_state = np.random.default_rng(random_state)
    from_yaml = isinstance(settings, dict)
    # Default info
    if from_yaml:
        defaults = copy.deepcopy(settings)
        grid_definition = defaults.pop("grid")
        models_definitions = grid_definition["models"]
        yaml_dir = defaults.pop("yaml_dir", "") or ""
    else:
        yaml_dir = getattr(settings, 'yaml_dir', "")

    def dicts_or_load(_infos):
        return [(yaml_load_file(os.path.join(yaml_dir, _info)) if
                 isinstance(_info, str) else _info)
                for _info in _infos]

    def dict_option(_name):
        s = getattr(settings, _name, {})
        if isinstance(s, str):
            return yaml_load_file(os.path.join(yaml_dir, s))
        return s

    if not from_yaml:
        defaults = settings.defaults if isinstance(settings.defaults, dict) \
            else merge_info(*dicts_or_load(settings.defaults))
        params = dict_option('params')
        param_extra = dict_option('param_extra_opts')
        settings_extra = dict_option('extra_opts')

    for jobItem in batch.items(wantSubItems=False):
        # Model info
        jobItem.makeChainPath()
        if from_yaml:
            model_tag = "_".join(jobItem.param_set)
            try:
                model_info = copy.deepcopy(models_definitions[model_tag] or {})
            except KeyError:
                raise ValueError("Model '%s' must be defined." % model_tag)
        else:
            model_info = {'params': {}}
            for par in jobItem.param_set:
                if par not in params:
                    raise ValueError("params[%s] must be defined." % par)
                model_info['params'][par] = params[par]
            job_param_extra = getattr(jobItem, 'param_extra_opts', {}) or {}
            job_extra = getattr(jobItem, 'extra_opts', {}) or {}
            extra = dict(param_extra, **job_param_extra)
            model_info = merge_info(settings_extra, job_extra, model_info,
                                    *[extra[par]
                                      for par in jobItem.param_set if par in extra])

        model_info = merge_info(defaults, model_info)
        data_infos = dicts_or_load(jobItem.data_set.infos)
        combined_info = merge_info(defaults, model_info, *data_infos)
        if "preset" in combined_info:
            preset = combined_info.pop("preset")
            combined_info = merge_info(create_input(**preset), combined_info)
        combined_info["output"] = jobItem.chainRoot
        # Requisites
        components_used = get_used_components(components_used, combined_info)
        if install_reqs_at:
            combined_info[packages_path_input] = os.path.abspath(install_reqs_at)
        # Save the info (we will write it after installation:
        # we need to install to add auto covmats
        if jobItem.paramtag not in infos:
            infos[jobItem.paramtag] = {}
        infos[jobItem.paramtag][jobItem.data_set.tag] = combined_info
    # Installing requisites
    if install_reqs_at:
        print("Installing required code and data for the grid.")
        from cobaya.log import logger_setup
        logger_setup()
        install_reqs(components_used, path=install_reqs_at, force=install_reqs_force)
    print("Adding covmats (if necessary) and writing input files")
    for jobItem in batch.items(wantSubItems=False):
        info = infos[jobItem.paramtag][jobItem.data_set.tag]
        # Covariance matrices
        # We try to find them now, instead of at run time, to check if correctly selected
        try:
            sampler = list(info["sampler"])[0]
        except KeyError:
            raise ValueError("No sampler has been chosen")
        if sampler == "mcmc" and info["sampler"][sampler].get("covmat", "auto"):
            if not (packages_path := install_reqs_at or info.get(packages_path_input)
                                     or resolve_packages_path()):
                raise ValueError("Cannot assign automatic covariance matrices because no "
                                 "external packages path has been defined.")
            # Need updated info for covmats: includes renames
            updated_info = update_info(info)
            # Ideally, we use slow+sampled parameters to look for the covariance matrix
            # but since for that we'd need to initialise a model, we approximate that set
            # as theory+sampled
            from itertools import chain
            like_params = set(chain(*[
                list(like.get("params") or [])
                for like in updated_info["likelihood"].values()]))
            params_info = {p: v for p, v in updated_info["params"].items()
                           if is_sampled_param(v) and p not in like_params}
            best_covmat = get_best_covmat_ext(os.path.abspath(packages_path),
                                              params_info, updated_info["likelihood"],
                                              random_state, msg_context=jobItem.name)
            info["sampler"][sampler]["covmat"] = os.path.join(
                best_covmat["folder"], best_covmat["name"]) if best_covmat else None
        # Write the info for this job
        # Allow overwrite since often will want to regenerate grid with tweaks
        yaml_dump_file(jobItem.yaml_file(), sort_cosmetic(info), error_if_exists=False)

        # Non-translated old code
        # if not start_at_bestfit:
        #     setMinimize(jobItem, ini)
        #     variant = '_minimize'
        #     ini.saveFile(jobItem.iniFile(variant))
        ## NOT IMPLEMENTED: start at best fit
        ##        ini.params['start_at_bestfit'] = start_at_bestfit
        # ---
        # for deffile in settings.defaults:
        #    ini.defaults.append(batch.commonPath + deffile)
        # if hasattr(settings, 'override_defaults'):
        #    ini.defaults = [batch.commonPath + deffile for deffile in settings.override_defaults] + ini.defaults
        # ---
        # # add ini files for importance sampling runs
        # for imp in jobItem.importanceJobs():
        #     if getattr(imp, 'importanceFilter', None): continue
        #     if batch.hasName(imp.name.replace('"post"', '')):
        #         raise Exception('importance sampling something you already have?')
        #     for minimize in (False, True):
        #         if minimize and not getattr(imp, 'want_minimize', True): continue
        #         ini = IniFile()
        #         updateIniParams(ini, imp.importanceSettings, batch.commonPath)
        #         if cosmomcAction == 0 and not minimize:
        #             for deffile in settings.importanceDefaults:
        #                 ini.defaults.append(batch.commonPath + deffile)
        #             ini.params['redo_outroot'] = imp.chainRoot
        #             ini.params['action'] = 1
        #         else:
        #             ini.params['file_root'] = imp.chainRoot
        #         if minimize:
        #             setMinimize(jobItem, ini)
        #             variant = '_minimize'
        #         else:
        #             variant = ''
        #         ini.defaults.append(jobItem.iniFile())
        #         ini.saveFile(imp.iniFile(variant))
        #         if cosmomcAction != 0: break

    if not interactive:
        return batch
    print('Done... to run do: cobaya-grid-run %s' % batchPath)
#    if not start_at_bestfit:
#        print('....... for best fits: python python/gridrun.py %s --minimize'%batchPath)
#    print('')
#    print('for importance sampled: python python/gridrun.py %s --importance'%batchPath)
#    print('for best-fit for importance sampled: '
#          'python python/gridrun.py %s --importance_minimize'%batchPath)
