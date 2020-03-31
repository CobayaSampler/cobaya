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

# Local
from cobaya.yaml import yaml_load_file, yaml_dump_file
from cobaya.conventions import _output_prefix, _packages_path, _yaml_extensions
from cobaya.conventions import kinds, _params
from cobaya.input import get_used_components, merge_info, update_info
from cobaya.install import install as install_reqs
from cobaya.tools import sort_cosmetic, warn_deprecation
from cobaya.grid_tools import batchjob
from cobaya.cosmo_input import create_input, _get_best_covmat
from cobaya.parameterization import is_sampled_param


def getArgs(vals=None):
    parser = argparse.ArgumentParser('Initialize grid using settings file')
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


def pathIsGrid(batchPath):
    return os.path.exists(batchjob.grid_cache_file(batchPath)) or os.path.exists(
        os.path.join(batchPath, 'config', 'config.ini'))


def make_grid_script():
    warn_deprecation()
    args = getArgs()
    args.interactive = True
    makeGrid(**args.__dict__)


def makeGrid(batchPath, settingName=None, settings=None, read_only=False,
             interactive=False, install_reqs_at=None, install_reqs_force=None):
    print("Generating grid...")
    batchPath = os.path.abspath(batchPath) + os.sep
    if not settings:
        if not settingName:
            raise NotImplementedError("Re-using previous batch is work in progress...")
        #            if not pathIsGrid(batchPath):
        #                raise Exception('Need to give name of setting file if batchPath/config '
        #                                'does not exist')
        #            read_only = True
        #            sys.path.insert(0, batchPath + 'config')
        #            settings = __import__(IniFile(batchPath + 'config/config.ini').params['setting_file'].replace('.py', ''))
        elif os.path.splitext(settingName)[-1].lower() in _yaml_extensions:
            settings = yaml_load_file(settingName)
        else:
            raise NotImplementedError("Using a python script is work in progress...")
            # In this case, info-as-dict would be passed
            # settings = __import__(settingName, fromlist=['dummy'])
    batch = batchjob.BatchJob(batchPath)
    # batch.skip = settings.get("skip", False)
    batch.makeItems(settings, messages=not read_only)
    if read_only:
        for jobItem in [b for b in batch.jobItems]:
            if not jobItem.chainExists():
                batch.jobItems.remove(jobItem)
        batch.save()
        print('OK, configured grid with %u existing chains' % (len(batch.jobItems)))
        return batch
    else:
        batch.makeDirectories(setting_file=None)
        batch.save()
    infos = {}
    components_used = {}
    # Default info
    defaults = copy.deepcopy(settings)
    grid_definition = defaults.pop("grid")
    models_definitions = grid_definition["models"]
    datasets_definitions = grid_definition["datasets"]
    for jobItem in batch.items(wantSubItems=False):
        # Model info
        jobItem.makeChainPath()
        try:
            model_info = copy.deepcopy(models_definitions[jobItem.param_set] or {})
        except KeyError:
            raise ValueError("Model '%s' must be defined." % jobItem.param_set)
        model_info = merge_info(defaults, model_info)
        # Dataset info
        try:
            dataset_info = copy.deepcopy(datasets_definitions[jobItem.data_set.tag])
        except KeyError:
            raise ValueError("Data set '%s' must be defined." % jobItem.data_set.tag)
        # Combined info
        combined_info = merge_info(defaults, model_info, dataset_info)
        if "preset" in combined_info:
            preset = combined_info.pop("preset")
            combined_info = merge_info(create_input(**preset), combined_info)
        combined_info[_output_prefix] = jobItem.chainRoot
        # Requisites
        components_used = get_used_components(components_used, combined_info)
        if install_reqs_at:
            combined_info[_packages_path] = os.path.abspath(install_reqs_at)
        # Save the info (we will write it after installation:
        # we need to install to add auto covmats
        if jobItem.param_set not in infos:
            infos[jobItem.param_set] = {}
        infos[jobItem.param_set][jobItem.data_set.tag] = combined_info
    # Installing requisites
    if install_reqs_at:
        print("Installing required code and data for the grid.")
        from cobaya.log import logger_setup
        logger_setup()
        install_reqs(components_used, path=install_reqs_at, force=install_reqs_force)
    print("Adding covmats (if necessary) and writing input files")
    for jobItem in batch.items(wantSubItems=False):
        info = infos[jobItem.param_set][jobItem.data_set.tag]
        # Covariance matrices
        # We try to find them now, instead of at run time, to check if correctly selected
        try:
            sampler = list(info[kinds.sampler])[0]
        except KeyError:
            raise ValueError("No sampler has been chosen")
        if sampler == "mcmc" and info[kinds.sampler][sampler].get("covmat", "auto"):
            packages_path = install_reqs_at or info.get(_packages_path, None)
            if not packages_path:
                raise ValueError("Cannot assign automatic covariance matrices because no "
                                 "external packages path has been defined.")
            # Need updated info for covmats: includes renames
            updated_info = update_info(info)
            # Ideally, we use slow+sampled parameters to look for the covariance matrix
            # but since for that we'd need to initialise a model, we approximate that set
            # as theory+sampled
            from itertools import chain
            like_params = set(chain(*[
                list(like[_params])
                for like in updated_info[kinds.likelihood].values()]))
            params_info = {p: v for p, v in updated_info[_params].items()
                           if is_sampled_param(v) and p not in like_params}
            best_covmat = _get_best_covmat(
                os.path.abspath(packages_path),
                params_info, updated_info[kinds.likelihood])
            info[kinds.sampler][sampler]["covmat"] = os.path.join(
                best_covmat["folder"], best_covmat["name"])
        # Write the info for this job
        # Allow overwrite since often will want to regenerate grid with tweaks
        yaml_dump_file(jobItem.iniFile(), sort_cosmetic(info), error_if_exists=False)

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
        #     if batch.hasName(imp.name.replace('_post', '')):
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
#        print('....... for best fits: python python/runbatch.py %s --minimize'%batchPath)
#    print('')
#    print('for importance sampled: python python/runbatch.py %s --importance'%batchPath)
#    print('for best-fit for importance sampled: '
#          'python python/runbatch.py %s --importance_minimize'%batchPath)
