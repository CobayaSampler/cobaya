# Global
import os
import pickle
from random import choice
from itertools import chain
import numpy as np
import re

# Local
from cobaya.conventions import _covmats_file, _aliases, _packages_path, partag, _params
from cobaya.conventions import kinds, _covmat_extension
from cobaya.tools import str_to_list, get_translated_params, get_cache_path
from cobaya.parameterization import is_sampled_param
from cobaya.input import update_info
from cobaya.log import LoggedError

# Logger
import logging

log = logging.getLogger(__name__.split(".")[-1])

covmat_folders = [
    "{%s}/data/planck_supp_data_and_covmats/covmats/" % _packages_path,
    "{%s}/data/bicep_keck_2015/BK15_cosmomc/planck_covmats/" % _packages_path]

# Global instance of loaded database, for fast calls to get_best_covmat in GUI
_loaded_covmats_database = None


def get_covmat_database(packages_path, cached=True):
    # Get folders with corresponding components installed
    installed_folders = [folder for folder in covmat_folders
                         if os.path.exists(
                             folder.format(**{_packages_path: packages_path}))]
    covmats_database_fullpath = os.path.join(get_cache_path(), _covmats_file)
    # Check if there is a usable cached one
    if cached:
        try:
            with open(covmats_database_fullpath, "rb") as f:
                covmat_database = pickle.load(f)
            # quick and dirty hash for regeneration: check number of .covmat files
            num_files = len(list(chain(
                *[[filename for filename in os.listdir(
                    folder.format(**{_packages_path: packages_path}))
                   if filename.endswith(_covmat_extension)]
                  for folder in installed_folders])))
            assert num_files == len(covmat_database)
            log.debug("Loaded cached covmats database")
            return covmat_database
        except:
            log.info("No cached covmat database present, not usable or not up-to-date. "
                     "Will be re-created and cached.")
            pass
    # Create it (again)
    covmat_database = []
    for folder in installed_folders:
        folder_full = folder.format(
            **{_packages_path: packages_path}).replace("/", os.sep)
        for filename in os.listdir(folder_full):
            try:
                with open(os.path.join(folder_full, filename), encoding="utf-8") as covmat:
                    header = covmat.readline()
                assert header.strip().startswith("#")
                params = header.strip().lstrip("#").split()
            except:
                continue
            covmat_database.append({"folder": folder, "name": filename, "params": params})
    if cached:
        with open(covmats_database_fullpath, "wb") as f:
            pickle.dump(covmat_database, f)
    return covmat_database


def get_best_covmat(info, packages_path=None, cached=True):
    """
    Chooses optimal covmat from a database, based on common parameters and likelihoods.

    Returns a dict `{folder: [folder_of_covmat], name: [file_name_of_covmat],
    params: [parameters_in_covmat], covmat: [covariance_matrix]}`.
    """
    packages_path = packages_path or info.get(_packages_path)
    if not packages_path:
        raise LoggedError(log, "Needs a path to the external packages installation.")
    updated_info = update_info(info)
    for p, pinfo in list(updated_info[_params].items()):
        if not is_sampled_param(pinfo):
            updated_info[_params].pop(p)
    info_sampled_params = updated_info[_params]
    covmat_data = _get_best_covmat(packages_path, updated_info[_params],
                                   updated_info[kinds.likelihood], cached=cached)
    covmat = np.atleast_2d(np.loadtxt(os.path.join(
        covmat_data["folder"].format(packages_path=packages_path), covmat_data["name"])))
    params_in_covmat = get_translated_params(info_sampled_params, covmat_data["params"])
    indices = [covmat_data["params"].index(p) for p in params_in_covmat.values()]
    covmat_data["covmat"] = covmat[indices][:, indices]
    covmat_data["params"] = params_in_covmat
    return covmat_data


def _get_best_covmat(packages_path, params_info, likelihoods_info, cached=True):
    """
    Actual covmat finder used by `get_best_covmat`. Call directly for more control on
    the parameters used.

    Returns the same dict as `get_best_covmat`, except for the covariance matrix itself.
    """
    if cached:
        global _loaded_covmats_database
        covmats_database = (
            _loaded_covmats_database or get_covmat_database(packages_path, cached=cached))
        _loaded_covmats_database = covmats_database
    # Prepare params and likes aliases
    params_renames = set(chain(*[
        [p] + str_to_list(info.get(partag.renames, [])) for p, info in
        params_info.items()]))
    likes_renames = set(chain(*[[like] + str_to_list((info or {}).get(_aliases, []))
                                for like, info in likelihoods_info.items()]))
    delimiters = r"[_\.]"
    likes_regexps = [re.compile(delimiters + re.escape(l) + delimiters)
                             for l in likes_renames]
    # Match number of params
    score_params = (
        lambda covmat: len(set(covmat["params"]).intersection(params_renames)))
    best_p = get_best_score(covmats_database, score_params)
    if not best_p:
        log.warning(
            "No covariance matrix found including at least one of the given parameters")
        return None
    # Match likelihood names / keywords
    # No debug print here: way too many!
    score_likes = (
        lambda covmat: len([0 for r in likes_regexps if r.search(covmat["name"])]))
    best_p_l = get_best_score(best_p, score_likes)
    if log.getEffectiveLevel() <= logging.DEBUG:
        log.debug("Subset based on params + likes:\n - " +
                  "\n - ".join([b["name"] for b in best_p_l]))
    # Finally, in case there is more than one, select shortest #params and name (simpler!)
    # #params first, to avoid extended models with shorter covmat name
    score_simpler_params = lambda covmat: -len(covmat["params"])
    best_p_l_sp = get_best_score(best_p_l, score_simpler_params)
    if log.getEffectiveLevel() <= logging.DEBUG:
        log.debug("Subset based on params + likes + fewest params:\n - " +
                  "\n - ".join([b["name"] for b in best_p_l_sp]))
    score_simpler_name = (
        lambda covmat: -len(covmat["name"].replace("_", " ").replace("-", " ").split()))
    best_p_l_sp_sn = get_best_score(best_p_l_sp, score_simpler_name)
    if log.getEffectiveLevel() <= logging.DEBUG:
        log.debug("Subset based on params + likes + fewest params + shortest name:\n - " +
                  "\n - ".join([b["name"] for b in best_p_l_sp_sn]))
    # if there is more than one (unlikely), just pick one at random
    if len(best_p_l_sp_sn) > 1:
        log.warning("WARNING: >1 possible best covmats: %r",
                    [b["name"] for b in best_p_l_sp_sn])
    return best_p_l_sp_sn[choice(range(len(best_p_l_sp_sn)))].copy()


def get_best_score(covmats, score_func):
    scores = np.array([score_func(covmat) for covmat in covmats])
    i_max = np.argwhere(scores == np.max(scores)).T[0]
    return [covmats[i] for i in i_max]
