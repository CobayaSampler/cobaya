"""
.. module:: cobaya.grid_tools.batchjob

:Synopsis: Classes for jobs and datasets
:Author: Antony Lewis, modified by Jesus Torrado

"""

import copy
import os
import pickle
import shutil
import sys
import time
from collections.abc import Callable
from typing import Any

from getdist import IniFile, types
from getdist.mcsamples import loadMCSamples
from getdist.paramnames import makeList as make_list

import cobaya
from cobaya.conventions import Extension
from cobaya.output import use_portalocker
from cobaya.tools import PythonPath
from cobaya.yaml import yaml_load_file

from .conventions import input_folder, input_folder_post, script_folder, yaml_ext


def grid_cache_file(directory):
    return os.path.abspath(directory) + os.sep + "batch.pyobj"


def resetGrid(directory):
    fname = grid_cache_file(directory)
    if os.path.exists(fname):
        os.remove(fname)


def readobject(directory=None):
    # load this here to prevent circular
    from cobaya.grid_tools import gridconfig

    if directory is None:
        directory = sys.argv[1]
    fname = grid_cache_file(directory)
    if not os.path.exists(fname):
        if gridconfig.path_is_grid(directory):
            return gridconfig.makeGrid(directory, read_only=True, interactive=False)
        return None
    try:
        config_dir = os.path.abspath(directory) + os.sep + "config"
        with PythonPath(config_dir, when=os.path.exists(config_dir)):
            # set path in case using functions defined
            # and hence imported from in settings file
            with open(fname, "rb") as inp:
                grid = pickle.load(inp)
        if not os.path.exists(grid.batchPath):
            raise FileNotFoundError("Directory not found %s" % grid.batchPath)
        return grid
    except Exception as e:
        print("Error loading cached batch object: ", e)
        resetGrid(directory)
        if gridconfig.path_is_grid(directory):
            return gridconfig.makeGrid(directory, read_only=True, interactive=False)
        raise


def saveobject(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def nonEmptyFile(fname):
    return os.path.exists(fname) and os.path.getsize(fname) > 0


class PropertiesItem:
    propertiesIniFile: Callable

    def propertiesIni(self):
        if os.path.exists(self.propertiesIniFile()):
            return IniFile(self.propertiesIniFile())
        else:
            ini = IniFile()
            ini.original_filename = self.propertiesIniFile()
            return ini


class DataSet:
    importanceNames: list
    importanceParams: list

    def __init__(self, names, option_dicts=None, covmat=None, dist_settings=None):
        if not dist_settings:
            dist_settings = {}
        if isinstance(names, str):
            names = [names]
        if option_dicts is None:
            option_dicts = [(name + yaml_ext) for name in names]
        else:
            option_dicts = self.standardizeParams(option_dicts)
        if covmat is not None:
            self.covmat = covmat
        self.names = names
        self.infos = option_dicts
        # can be an array of items, either ini file name or dictionaries of parameters
        self.tag = "_".join(self.names)
        self.dist_settings = dist_settings

    def add(self, name, params=None, overrideExisting=True, dist_settings=None):
        if params is None:
            params = [name]
        params = self.standardizeParams(params)
        if dist_settings:
            self.dist_settings.update(dist_settings)
        if overrideExisting:
            self.infos = params + self.infos
            # can be an array of items, either ini file name or dictionaries of parameters
        else:
            self.infos += params
        if name is not None:
            self.names += [name]
            self.tag = "_".join(self.names)
        return self

    def addEnd(self, name, params, dist_settings=None):
        if not dist_settings:
            dist_settings = {}
        return self.add(name, params, overrideExisting=False, dist_settings=dist_settings)

    def copy(self):
        return copy.deepcopy(self)

    def extendForImportance(self, names, params):
        data = copy.deepcopy(self)
        if ".post." not in data.tag:
            data.tag += ".post." + "_".join(names)
        else:
            data.tag += "_" + "_".join(names)
        data.importanceNames = names
        data.importanceParams = data.standardizeParams(params)
        data.names += data.importanceNames
        # data.infos += data.importanceParams
        return data

    def standardizeParams(self, params):
        if isinstance(params, dict) or isinstance(params, str):
            params = [params]
        for i in range(len(params)):
            if isinstance(params[i], str) and yaml_ext not in params[i]:
                params[i] += yaml_ext
        return params

    def hasName(self, name):
        if isinstance(name, str):
            return name in self.names
        else:
            return any(True for i in name if i in self.names)

    def hasAll(self, name):
        if isinstance(name, str):
            return name in self.names
        else:
            return all((i in self.names) for i in name)

    def tagReplacing(self, x, y):
        items = []
        for name in self.names:
            if name == x:
                if y != "":
                    items.append(y)
            else:
                items.append(name)
        return "_".join(items)

    def namesReplacing(self, dic):
        if dic is None:
            return self.names
        items = []
        for name in self.names:
            if name in dic:
                val = dic[name]
                if val:
                    items.append(val)
            else:
                items.append(name)
        return items

    def makeNormedDatatag(self, dic):
        return "_".join(sorted(self.namesReplacing(dic)))


# not currently used
class JobGroup:
    def __init__(self, name, params=None, importanceRuns=None, datasets=None):
        if importanceRuns is None:
            importanceRuns = []
        if params is None:
            params = [[]]
        if datasets is None:
            self.params = params
            self.groupName = name
            self.importanceRuns = importanceRuns
            self.datasets = []


class ImportanceSetting:
    def __init__(self, names, inis=None, dist_settings=None, minimize=True):
        if not inis:
            inis = []
        self.names = make_list(names)
        self.inis = make_list(inis)
        self.dist_settings = dist_settings or {}
        self.want_minimize = minimize

    def want_importance(self, _jobItem):
        return True


class ImportanceFilter(ImportanceSetting):
    # class for trivial importance sampling filters that can be done in python,
    # e.g. restricting a parameter to a new range

    def __init__(self, names, dist_settings=None, minimize=False):
        super().__init__(
            names, inis=[self], dist_settings=dist_settings, minimize=minimize
        )


class JobItem(PropertiesItem):
    importanceTag: str
    importanceSettings: list
    importanceFilter: ImportanceFilter

    def __init__(
        self, path, param_set, data_set, base="base", group_name=None, minimize=True
    ):
        self.param_set = param_set
        if not isinstance(data_set, DataSet):
            data_set = DataSet(data_set[0], data_set[1])
        self.data_set = data_set
        self.base = base
        self.paramtag = "_".join([base] + param_set)
        self.datatag = data_set.tag
        self.name = self.paramtag + "_" + self.datatag
        self.batchPath = path
        self.relativePath = self.paramtag + os.sep + self.datatag + os.sep
        self.chainPath = path + self.relativePath
        self.chainRoot = self.chainPath + self.name
        self.distPath = self.chainPath + "dist" + os.sep
        self.distRoot = self.distPath + self.name
        self.isImportanceJob = False
        self.importanceItems = []
        self.want_minimize = minimize
        self.result_converge = None
        self.group = group_name
        self.parent = None
        self.dist_settings = copy.copy(data_set.dist_settings)
        self.makeIDs()

    def yaml_file(self, variant=""):
        if not self.isImportanceJob:
            return self.batchPath + input_folder + os.sep + self.name + variant + yaml_ext
        else:
            return (
                self.batchPath
                + input_folder_post
                + os.sep
                + self.name
                + variant
                + yaml_ext
            )

    def propertiesIniFile(self):
        return self.chainRoot + ".properties.ini"

    def isBurnRemoved(self):
        return self.propertiesIni().bool("burn_removed")

    def makeImportance(self, importance_runs):
        for imp_run in importance_runs:
            if isinstance(imp_run, ImportanceSetting):
                if not imp_run.want_importance(self):
                    continue
            else:
                if len(imp_run) not in (2, 3):
                    raise ValueError(
                        "importance_runs must be list of tuples of "
                        "(names, infos, [ImportanceFilter]) or "
                        "ImportanceSetting instances"
                    )
                if len(imp_run) > 2 and not imp_run[2].want_importance(self):
                    continue
                imp_run = ImportanceSetting(imp_run[0], imp_run[1])
            if len(set(imp_run.names).intersection(self.data_set.names)) > 0:
                print(
                    "importance job duplicating parent data set: {} with {}".format(
                        self.name, imp_run.names
                    )
                )
                continue
            data = self.data_set.extendForImportance(imp_run.names, imp_run.inis)
            job = JobItem(
                self.batchPath, self.param_set, data, minimize=imp_run.want_minimize
            )
            job.importanceTag = "_".join(imp_run.names)
            job.importanceSettings = imp_run.inis
            if ".post." not in self.name:
                tag = ".post." + job.importanceTag
            else:
                tag = "_" + job.importanceTag
            job.name = self.name + tag
            job.chainRoot = self.chainRoot + tag
            job.distPath = self.distPath
            job.chainPath = self.chainPath
            job.relativePath = self.relativePath
            job.distRoot = self.distRoot + tag
            job.datatag = self.datatag + tag
            job.isImportanceJob = True
            job.parent = self
            job.group = self.group
            job.dist_settings.update(imp_run.dist_settings)
            if isinstance(imp_run, ImportanceFilter):
                job.importanceFilter = imp_run
            job.makeIDs()
            self.importanceItems.append(job)

    def makeNormedName(self, dataSubs=None):
        normed_params = "_".join(sorted(self.param_set))
        normed_data = self.data_set.makeNormedDatatag(dataSubs)
        normed_name = self.base
        if len(normed_params) > 0:
            normed_name += "_" + normed_params
        normed_name += "_" + normed_data
        return normed_name, normed_params, normed_data

    def makeIDs(self):
        self.normed_name, self.normed_params, self.normed_data = self.makeNormedName()

    def matchesDatatag(self, tagList):
        if self.datatag in tagList or self.normed_data in tagList:
            return True
        return self.datatag.replace(".post.", "_") in [
            tag.replace(".post.", "_") for tag in tagList
        ]

    def hasParam(self, name):
        if isinstance(name, str):
            return name in self.param_set
        else:
            return any(True for i in name if i in self.param_set)

    def importanceJobs(self):
        return self.importanceItems

    def importanceJobsRecursive(self):
        res = copy.copy(self.importanceItems)
        for r in self.importanceItems:
            res += r.importanceJobsRecursive()
        return res

    def removeImportance(self, job):
        if job in self.importanceItems:
            self.importanceItems.remove(job)
        else:
            for j in self.importanceItems:
                j.removeImportance(job)

    def makeChainPath(self):
        os.makedirs(self.chainPath, exist_ok=True)
        return self.chainPath

    def chainName(self, chain=1):
        return self.chainRoot + "." + str(chain) + ".txt"

    def chainExists(self, chain=1):
        fname = self.chainName(chain)
        return nonEmptyFile(fname)

    def chainNames(self, num_chains=None):
        if num_chains:
            return [self.chainName(i) for i in range(num_chains)]
        else:
            i = 1
            chains = []
            while self.chainExists(i):
                chains.append(self.chainName(i))
                i += 1
            return chains

    def allChainExists(self, num_chains):
        return all(self.chainExists(i + 1) for i in range(num_chains))

    def chainFileDate(self, chain=1):
        return os.path.getmtime(self.chainName(chain))

    def chainsDodgy(self, interval=600):
        dates = []
        i = 1
        while os.path.exists(self.chainName(i)):
            dates.append(os.path.getmtime(self.chainName(i)))
            i += 1
        return os.path.exists(self.chainName(i + 1)) or max(dates) - min(dates) > interval

    def notRunning(self, age_qualify_minutes=5):
        if not self.chainExists():
            return False  # might be in queue
        if use_portalocker():
            lock_file = self.chainRoot + ".input.yaml.locked"
            if not os.path.exists(lock_file):
                return True
            h: Any = None
            import portalocker

            try:
                h = open(lock_file, "wb")
                portalocker.lock(h, portalocker.LOCK_EX + portalocker.LOCK_NB)
            except (portalocker.exceptions.BaseLockException, OSError):
                if h:
                    h.close()
                return False
            else:
                h.close()
                del h
                try:
                    os.remove(lock_file)
                except OSError:
                    return False
        # if no locking, just check if files updated recently
        lastWrite = self.chainFileDate()
        return lastWrite < time.time() - age_qualify_minutes * 60

    def chainMinimumExists(self):
        fname = self.chainRoot + ".minimum"
        return nonEmptyFile(fname)

    def chainBestfit(self, paramNameFile=None):
        bf_file = self.chainRoot + ".minimum"
        if nonEmptyFile(bf_file):
            return types.BestFit(bf_file, paramNameFile)
        return None

    def chainMinimumConverged(self):
        bf = self.chainBestfit()
        if bf is None:
            return False
        return bf.logLike < 1e29

    def convergeStat(self):
        fname = self.chainRoot + Extension.checkpoint
        if not nonEmptyFile(fname):
            return None, None
        yaml = yaml_load_file(fname)
        try:
            sampler = next(iter(yaml["sampler"].values()))
            R = float(sampler.get("Rminus1_last"))
            return R, sampler.get("converged")
        except Exception:
            return None, None

    def chainFinished(self):
        if self.isImportanceJob:
            done = self.parent.convergeStat()[1]
            if done is None or self.parentChanged() or not self.notRunning():
                return False
        else:
            done = self.convergeStat()[1]
        if done is None:
            return False
        return done

    def wantCheckpointContinue(self, minR=0):
        R, done = self.convergeStat()
        if R is None:
            return False
        if not os.path.exists(self.chainRoot + Extension.covmat):
            return False
        return not done and R > minR

    def getDistExists(self):
        return os.path.exists(self.distRoot + ".margestats")

    def getDistNeedsUpdate(self):
        return self.chainExists() and (
            not self.getDistExists()
            or self.chainFileDate() > os.path.getmtime(self.distRoot + ".margestats")
        )

    def parentChanged(self):
        return (
            not self.chainExists() or self.chainFileDate() < self.parent.chainFileDate()
        )

    def R(self):
        if self.result_converge is None:
            fname = self.distRoot + ".converge"
            if not nonEmptyFile(fname):
                return None
            self.result_converge = types.ConvergeStats(fname)
        try:
            return float(self.result_converge.worstR())
        except (TypeError, IndexError):
            return None

    def hasConvergeBetterThan(self, R, returnNotExist=False):
        try:
            chainR = self.R()
            if chainR is None:
                return returnNotExist
            return chainR <= R
        except Exception:
            print("WARNING: Bad .converge for " + self.name)
            return returnNotExist

    def loadJobItemResults(
        self,
        paramNameFile=None,
        bestfit=True,
        bestfitonly=False,
        noconverge=False,
        silent=False,
    ):
        self.result_converge = None
        self.result_marge = None
        self.result_likemarge = None
        self.result_bestfit = self.chainBestfit(paramNameFile)
        if not bestfitonly:
            marge_root = self.distRoot
            if self.getDistExists():
                if not noconverge:
                    self.result_converge = types.ConvergeStats(marge_root + ".converge")
                self.result_marge = types.MargeStats(
                    marge_root + ".margestats", paramNameFile
                )
                self.result_likemarge = types.LikeStats(marge_root + ".likestats")
                if self.result_bestfit is not None and bestfit:
                    self.result_marge.addBestFit(self.result_bestfit)
            elif not silent:
                print("missing: " + marge_root)

    def getMCSamples(self, ini=None, settings=None):
        return loadMCSamples(self.chainRoot, jobItem=self, ini=ini, settings=settings)


class BatchJob(PropertiesItem):
    def __init__(self, path):
        self.batchPath = path
        self.skip = []
        self.subBatches = []
        # self.jobItems = None
        self.getdist_options = {}

    def propertiesIniFile(self):
        return os.path.join(self.batchPath, "config", "config.ini")

    def make_items(self, settings, messages=True, base_name="base"):
        self.jobItems = []
        dic = settings if isinstance(settings, dict) else settings.__dict__
        self.getdist_options = dic.get("getdist_options") or self.getdist_options
        all_importance = dic.get("importance_runs") or []
        self.skip = dic.get("skip") or []

        dataset_infos = dic.get("datasets") or {}
        model_infos = dic.get("models") or {}
        for group_name, group in dic["groups"].items():
            skip = group.get("skip") or {}
            data_used = set()
            for data_set in group["datasets"]:
                if isinstance(data_set, str):
                    if data_set not in dataset_infos:
                        raise ValueError("Dataset name '%s' must be defined." % data_set)
                    info = (dataset_infos.get(data_set) or {}).copy()
                    dataset = DataSet(info.pop("tags", data_set.split("_")), info)
                else:
                    dataset = data_set

                if (
                    data_tags := frozenset(
                        dataset.names if isinstance(dataset, DataSet) else dataset[0]
                    )
                ) in data_used:
                    raise ValueError("Duplicate dataset tags %s" % data_tags)
                data_used.add(data_tags)
                models_used = set()
                for model in group["models"]:
                    model_info = None
                    if isinstance(model, str):
                        if (
                            isinstance(skip, dict)
                            and isinstance(data_set, str)
                            and data_set in skip.get(model, {})
                        ):
                            continue
                        if model not in model_infos:
                            raise ValueError("Model '%s' must be defined." % model)
                        model_info = (model_infos[model] or {}).copy()
                        model = (
                            (model_info.pop("tags", []) or [])
                            if "tags" in model_info
                            else model.split("_")
                        )
                    elif not isinstance(model, (list, tuple)):
                        raise ValueError(
                            "models must be model name strings or list of model names"
                        )
                    if frozenset(model) in models_used:
                        raise ValueError("Duplicate model (parameter) tags %s" % model)
                    models_used.add(frozenset(model))
                    item = JobItem(
                        self.batchPath,
                        model,
                        dataset,
                        base=group.get("base") or dic.get("base") or base_name,
                        group_name=group_name,
                    )
                    item.model_info = model_info
                    item.defaults = group.get("defaults") or {}
                    item.param_extra_opts = group.get("param_extra_opts") or {}
                    if item.name not in self.skip and item.name not in skip:
                        item.makeImportance(group.get("importance_runs") or [])
                        item.makeImportance(all_importance)
                        self.jobItems.append(item)

        for item in dic.get("job_items") or []:
            self.jobItems.append(item)
            item.makeImportance(all_importance)
        if filters := dic.get("importance_filters"):
            for job in self.jobItems:
                for item in job.importanceJobs():
                    item.makeImportance(filters)
                job.makeImportance(filters)

        for item in list(self.items()):
            for x in [imp for imp in item.importanceJobsRecursive()]:
                if self.has_normed_name(x.normed_name):
                    if messages:
                        print(
                            "replacing importance sampling run with full run: %s" % x.name
                        )
                    item.removeImportance(x)
        for item in list(self.items()):
            for x in [imp for imp in item.importanceJobsRecursive()]:
                if self.has_normed_name(x.normed_name, wantImportance=True, exclude=x):
                    if messages:
                        print("removing duplicate importance sampling run: " + x.name)
                    item.removeImportance(x)

    def items(self, wantSubItems=True, wantImportance=False):
        for item in self.jobItems:
            yield item
            if wantImportance:
                for imp in item.importanceJobsRecursive():
                    if imp.name not in self.skip:
                        yield imp
        if wantSubItems:
            for subBatch in self.subBatches:
                for item in subBatch.items(wantSubItems, wantImportance):
                    yield item

    def hasName(self, name, wantSubItems=True):
        for jobItem in self.items(wantSubItems):
            if jobItem.name == name:
                return True
        return False

    def has_normed_name(
        self, name, wantSubItems=True, wantImportance=False, exclude=None
    ):
        return (
            self.normed_name_item(name, wantSubItems, wantImportance, exclude) is not None
        )

    def normed_name_item(
        self, name, wantSubItems=True, wantImportance=False, exclude=None
    ):
        for jobItem in self.items(wantSubItems, wantImportance):
            if jobItem.normed_name == name and jobItem is not exclude:
                return jobItem
        return None

    @staticmethod
    def normalizeDataTag(tag):
        return "_".join(sorted(tag.replace(".post.", "_").split("_")))

    def resolveName(
        self,
        paramtag,
        datatag,
        wantSubItems=True,
        wantImportance=True,
        raiseError=True,
        base="base",
        returnJobItem=False,
    ):
        if paramtag:
            if isinstance(paramtag, str):
                paramtag = paramtag.split("_")
            paramtags = [base] + sorted(paramtag)
        else:
            paramtag = [base]
            paramtags = [base]
        name = "_".join(paramtags) + "_" + self.normalizeDataTag(datatag)

        if jobItem := self.normed_name_item(name, wantSubItems, wantImportance):
            return jobItem if returnJobItem else jobItem.name
        if raiseError:
            raise Exception(
                "No match for paramtag, datatag... " + "_".join(paramtag) + ", " + datatag
            )
        else:
            return None

    def resolve_root(self, root):
        for jobItem in self.items(True, True):
            if jobItem.name == root:
                return jobItem
        return self.normed_name_item(root, True, True)

    def save(self, filename=""):
        saveobject(self, filename if filename else grid_cache_file(self.batchPath))

    def make_directories(self, setting_file=None):
        os.makedirs(self.batchPath, exist_ok=True)
        if setting_file:
            s = self.batchPath + "config"
            os.makedirs(s, exist_ok=True)
            setting_file = setting_file.replace(".pyc", ".py")
            shutil.copy(setting_file, self.batchPath + "config")
            props = self.propertiesIni()
            props.params["setting_file"] = os.path.split(setting_file)[-1]
            props.params["cobaya_version"] = cobaya.__version__
            props.saveFile()
        for p in (input_folder, input_folder_post, script_folder):
            os.makedirs(self.batchPath + p, exist_ok=True)
