"""
.. module:: cobaya.grid_tools.gridconfig

:Synopsis: Grid creator (Cobaya version)
:Author: Antony Lewis and Jesus Torrado (based on Antony Lewis' CosmoMC version of the same code)

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import print_function

# Global
import os
import copy
import argparse

# Local
from cobaya.yaml import yaml_load_file, yaml_dump_file
from cobaya.conventions import _output_prefix, _path_install
from cobaya.input import get_modules, merge_info
from cobaya.install import install as install_reqs


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
                        help="Force re-installation of apparently installed modules.")
    return parser.parse_args(vals)


def MakeGridScript():
    args = getArgs()
    args.interactive = True
    makeGrid(**args.__dict__)


def makeGrid(batchPath, settingName=None, settings=None, read_only=False,
             interactive=False, install_reqs_at=None, install_reqs_force=None):
    batchPath = os.path.abspath(batchPath) + os.sep
#    # 0: chains, 1: importance sampling, 2: best-fit, 3: best-fit and Hessian
#    cosmomcAction = 0
    if not settings:
        if not settingName:
            raise NotImplementedError("Re-using previous batch is work in progress...")
#            if not pathIsGrid(batchPath):
#                raise Exception('Need to give name of setting file if batchPath/config '
#                                'does not exist')
#            read_only = True
#            sys.path.insert(0, batchPath + 'config')
#            sys.modules['batchJob'] = batchjob  # old name
#            settings = __import__(IniFile(batchPath + 'config/config.ini').params['setting_file'].replace('.py', ''))
        elif os.path.splitext(settingName)[-1].lower() in (".yml", ".yaml"):
            settings = yaml_load_file(settingName)
        else:
# ACTUALLY, in the scripted case a DICT or a YAML FILE NAME should be passed
            raise NotImplementedError("Using a python script is work in progress...")
#            settings = __import__(settingName, fromlist=['dummy'])
    from cobaya.grid_tools import batchjob
    batch = batchjob.batchJob(batchPath, settings.get("yaml_dir", None))
###    batch.skip = settings.get("skip", False)
    if "skip" in settings:
        raise NotImplementedError("Skipping not implemented yet.")
    batch.makeItems(settings, messages=not read_only)
    if read_only:
        for jobItem in [b for b in batch.jobItems]:
            if not jobItem.chainExists():
                batch.jobItems.remove(jobItem)
        batch.save()
        print('OK, configured grid with %u existing chains' % (len(batch.jobItems)))
        return batch
    else:
# WAS        batch.makeDirectories(settings.__file__)
# WHY THE DIR OF settings AND NOT THE GRID DIR GIVEN???
        batch.makeDirectories(setting_file=None)
        batch.save()

# NOT IMPLEMENTED YET: start at best fit!!!
#    start_at_bestfit = getattr(settings, 'start_at_bestfit', False)

    defaults = copy.deepcopy(settings)
    modules_used = {}
    grid_definition = defaults.pop("grid")
    models_definitions = grid_definition["models"]
    datasets_definitions = grid_definition["datasets"]
    for jobItem in batch.items(wantSubItems=False):
        jobItem.makeChainPath()
        base_info = copy.deepcopy(defaults)
        try:
            model_info = models_definitions[jobItem.param_set] or {}
        except KeyError:
            raise ValueError("Model '%s' must be defined."%jobItem.param_set)

        # COVMATS NOT IMPLEMENTED YET!!!
        # cov_dir_name = getattr(settings, 'cov_dir', 'planck_covmats')
        # covdir = os.path.join(batch.basePath, cov_dir_name)
        # covmat = os.path.join(covdir, jobItem.name + '.covmat')
        # if not os.path.exists(covmat):
        #     covNameMappings = getattr(settings, 'covNameMappings', None)
        #     mapped_name_norm = jobItem.makeNormedName(covNameMappings)[0]
        #     covmat_normed = os.path.join(covdir, mapped_name_norm + '.covmat')
        #     covmat = covmat_normed
        #     if not os.path.exists(covmat) and hasattr(jobItem.data_set,
        #                                               'covmat'): covmat = batch.basePath + jobItem.data_set.covmat
        #     if not os.path.exists(covmat) and hasattr(settings, 'covmat'): covmat = batch.basePath + settings.covmat
        # else:
        #     covNameMappings = None
        # if os.path.exists(covmat):
        #     ini.params['propose_matrix'] = covmat
        #     if getattr(settings, 'newCovmats', True): ini.params['MPI_Max_R_ProposeUpdate'] = 20
        # else:
        #     hasCov = False
        #     ini.params['MPI_Max_R_ProposeUpdate'] = 20
        #     covmat_try = []
        #     if 'covRenamer' in dir(settings):
        #         covmat_try += settings.covRenamer(jobItem.name)
        #         covmat_try += settings.covRenamer(mapped_name_norm)
        #     if hasattr(settings, 'covrenames'):
        #         for aname in [jobItem.name, mapped_name_norm]:
        #             covmat_try += [aname.replace(old, new, 1) for old, new in settings.covrenames if old in aname]
        #             for new1, old1 in settings.covrenames:
        #                 if old1 in aname:
        #                     name = aname.replace(old1, new1, 1)
        #                     covmat_try += [name.replace(old, new, 1) for old, new in settings.covrenames if old in name]
        #     if 'covWithoutNameOrder' in dir(settings):
        #         if covNameMappings:
        #             removes = copy.deepcopy(covNameMappings)
        #         else:
        #             removes = dict()
        #         for name in settings.covWithoutNameOrder:
        #             if name in jobItem.data_set.names:
        #                 removes[name] = ''
        #                 covmat_try += [jobItem.makeNormedName(removes)[0]]
        #     covdir2 = os.path.join(batch.basePath, getattr(settings, 'cov_dir_fallback', cov_dir_name))
        #     for name in covmat_try:
        #         covmat = os.path.join(batch.basePath, covdir2, name + '.covmat')
        #         if os.path.exists(covmat):
        #             ini.params['propose_matrix'] = covmat
        #             print('covmat ' + jobItem.name + ' -> ' + name)
        #             hasCov = True
        #             break
        #     if not hasCov: print('WARNING: no matching specific covmat for ' + jobItem.name)


        ## NOT IMPLEMENTED: start at best fit
        ##        ini.params['start_at_bestfit'] = start_at_bestfit


        try:
            dataset_info = datasets_definitions[jobItem.data_set.tag]
        except KeyError:
            raise ValueError("Data set '%s' must be defined."%jobItem.data_set.tag)
        combined_info = merge_info(base_info, model_info, dataset_info)
        combined_info[_output_prefix] = jobItem.chainRoot

# ???
#        for deffile in settings.defaults:
#            ini.defaults.append(batch.commonPath + deffile)
#        if hasattr(settings, 'override_defaults'):
#            ini.defaults = [batch.commonPath + deffile for deffile in settings.override_defaults] + ini.defaults

        # requisites
        modules_used = get_modules(modules_used, combined_info)
        if install_reqs_at:
            combined_info[_path_install] = os.path.abspath(install_reqs_at)
        # Write the info for this job
        yaml_dump_file(combined_info, jobItem.iniFile())

        # if not start_at_bestfit:
        #     setMinimize(jobItem, ini)
        #     variant = '_minimize'
        #     ini.saveFile(jobItem.iniFile(variant))


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

    # Installing requisites
    print("Installing required code and data for the grid.")
    if install_reqs_at:
        install_reqs(modules_used, path=install_reqs_at, force=install_reqs_force)
    if not interactive:
        return batch
    print('Done... to run do: cobaya-grid-run %s'%batchPath)
#    if not start_at_bestfit:
#        print('....... for best fits: python python/runbatch.py %s --minimize'%batchPath)
#    print('')
#    print('for importance sampled: python python/runbatch.py %s --importance'%batchPath)
#    print('for best-fit for importance sampled: '
#          'python python/runbatch.py %s --importance_minimize'%batchPath)

