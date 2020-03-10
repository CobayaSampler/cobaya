# Global
import os
from random import choice
from itertools import chain
import numpy as np
from copy import deepcopy

# Local
from cobaya.yaml import yaml_load_file, yaml_dump_file
from cobaya.conventions import _covmats_file, _aliases, _path_install, partag, _params
from cobaya.conventions import kinds
from cobaya.tools import str_to_list
from cobaya.parameterization import is_sampled_param
from cobaya.input import get_default_info

# Logger
import logging

log = logging.getLogger(__name__.split(".")[-1])

covmat_folders = ["{%s}/data/planck_supp_data_and_covmats/covmats/" % _path_install,
                  "{%s}/data/bicep_keck_2015/BK15_cosmomc/planck_covmats/" % _path_install]

# Global instance of loaded database, for fast calls to get_best_covmat in GUI
_loaded_covmats_database = None


def get_covmat_database(modules, cached=True):
    # Get folders with corresponding modules installed
    installed_folders = [folder for folder in covmat_folders
                         if os.path.exists(folder.format(**{_path_install: modules}))]
    covmats_database_fullpath = os.path.join(modules, _covmats_file)
    # Check if there is a usable cached one
    if cached:
        try:
            covmat_database = yaml_load_file(covmats_database_fullpath)
            assert set(covmat_database) == set(installed_folders)
            return covmat_database
        except:
            log.info("No cached covmat database present, not usable or not up-to-date. "
                     "Will be re-created and cached.")
            pass
    # Create it (again)
    covmat_database = {}
    for folder in installed_folders:
        covmat_database[folder] = []
        folder_full = folder.format(**{_path_install: modules}).replace("/", os.sep)
        for filename in os.listdir(folder_full):
            try:
                with open(os.path.join(folder_full, filename), encoding="utf-8") as covmat:
                    header = covmat.readline()
                assert header.strip().startswith("#")
                params = header.strip().lstrip("#").split()
            except:
                continue
            covmat_database[folder].append({"name": filename, "params": params})
    if cached:
        yaml_dump_file(covmats_database_fullpath, covmat_database, error_if_exists=False)
    return covmat_database


def prepare_info_best_covmat(info):
    """
    Helper function to prepare input for `get_best_covmat`. Updated bare input info with
    defaults, including default parameter renames of the given theory code.

    Returns `(info_params, info_likelihood)` ready to use as arguments of
    `get_best_covmat`.
    """
    sampled_params_info = {
        p: deepcopy(v) for p, v in info[_params].items() if is_sampled_param(v)}
    theory_renames = get_default_info(list(info[kinds.theory])[0])[partag.renames]
    theory_renames_sets = [
        set([k] + str_to_list(v)) for k, v in theory_renames.items()]
    for p, pinfo in sampled_params_info.items():
        try:
            this_set = next(s for s in theory_renames_sets if p in s)
        except StopIteration:
            this_set = set()
        this_set = this_set.union(pinfo.get(partag.renames, set()))
        try:
            this_set.remove(p)
        except:
            pass
        pinfo[partag.renames] = list(this_set)
    info_likelihood = {l: get_default_info(l) for l in info[kinds.likelihood]}
    return sampled_params_info, info_likelihood


def get_best_covmat(modules, slow_params_info, likelihoods_info, cached=True):
    if cached:
        global _loaded_covmats_database
        covmats_database = (
            _loaded_covmats_database or get_covmat_database(modules, cached=cached))
        _loaded_covmats_database = covmats_database
    # Select first based on number of slow parameters
    params_renames = set(chain(*[
        [p] + str_to_list(info.get(partag.renames, [])) for p, info in
        slow_params_info.items()]))
    get_score_params = (
        lambda covmat_params: len(set(covmat_params).intersection(params_renames)))
    highest_score = 0
    best = []
    for folder, covmats in covmats_database.items():
        for covmat in covmats:
            score = get_score_params(covmat["params"])
            if score > highest_score:
                highest_score = score
                best = []
            if score == highest_score:
                best.append({
                    "folder": folder, "name": covmat["name"], "params": covmat["params"]})
    if highest_score == 0:
        log.warning(
            "No covariance matrix found including at least one of the given parameters")
        return None
    # Sub-select by number of likelihoods
    likes_renames = set(chain(*[[like] + str_to_list((info or {}).get(_aliases, []))
                                for like, info in likelihoods_info.items()]))
    get_score_likes = (
        lambda covmat_name: len([0 for like in likes_renames if like in covmat_name]))
    highest_score = 0
    best_2 = []
    for covmat in best:
        score = get_score_likes(covmat["name"])
        if score > highest_score:
            highest_score = score
            best_2 = []
        if score == highest_score:
            best_2.append(covmat)
    # Finally, in case there is more than one, select shortest #params and name (simpler!)
    # #params first, to avoid extended models with shorter covmat name
    get_score_simpler_params = lambda covmat_params: -len(covmat_params)
    highest_score = -np.inf
    best_3 = []
    for covmat in best_2:
        score = get_score_simpler_params(covmat["params"])
        if score > highest_score:
            highest_score = score
            best_3 = []
        if score == highest_score:
            best_3.append(covmat)
    get_score_simpler_name = (
        lambda covmat_name: -len(covmat_name.replace("_", " ").replace("-", " ").split()))
    highest_score = -np.inf
    best_4 = []
    for covmat in best_3:
        score = get_score_simpler_name(covmat["name"])
        if score > highest_score:
            highest_score = score
            best_4 = []
        if score == highest_score:
            best_4.append(covmat)
    # if there is more than one (unlikely), just pick one at random
    if len(best_4) > 1:
        log.warning("WARNING: >1 possible best covmats: %r" % [b["name"] for b in best_4])
    return best_4[choice(range(len(best_4)))]
