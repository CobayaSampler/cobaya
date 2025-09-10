import hashlib
import os
import pickle
import re
from itertools import chain
from typing import NamedTuple

import numpy as np

from cobaya.conventions import Extension, packages_path_input
from cobaya.input import update_info
from cobaya.log import LoggedError, get_logger, is_debug
from cobaya.parameterization import is_sampled_param
from cobaya.tools import get_cache_path, get_translated_params, str_to_list
from cobaya.typing import empty_dict

_covmats_file = "covmat_%s.pkl"

log = get_logger(__name__)

covmat_folders = [
    "{%s}/data/planck_supp_data_and_covmats/covmats/" % packages_path_input,
    "{%s}/data/bicep_keck_2018/BK18_cosmomc/planck_covmats/" % packages_path_input,
]


class CovmatFileKey(NamedTuple):
    paramtags: frozenset
    datatags: frozenset
    base: str


def covmat_file_key(paramtags, datatags, base):
    return CovmatFileKey(frozenset(paramtags), frozenset(datatags), base)


# Global instance of loaded database, for fast calls to get_best_covmat in GUI
_loaded_covmats_database: dict[str, dict[CovmatFileKey, dict]] = {}


def get_covmat_package_folders(packages_path) -> list[str]:
    install_folders = []
    for folder in covmat_folders:
        folder_full = folder.format(**{packages_path_input: packages_path}).replace(
            "/", os.sep
        )
        if os.path.exists(folder_full):
            install_folders.append(folder_full)
    return install_folders


def get_covmat_database(installed_folders, cached=True) -> dict[CovmatFileKey, dict]:
    # Get folders with corresponding components installed
    _hash = hashlib.md5(str(installed_folders).encode("utf8")).hexdigest()
    covmats_database_fullpath = os.path.join(get_cache_path(), _covmats_file % _hash)
    # Check if there is a usable cached one
    if cached:
        if covmats_database := _loaded_covmats_database.get(_hash):
            return covmats_database
        try:
            with open(covmats_database_fullpath, "rb") as f:
                covmat_database = pickle.load(f)
            # quick and dirty hash for regeneration: check number of .covmat files
            num_files = len(
                list(
                    chain(
                        *[
                            [
                                filename
                                for filename in os.listdir(folder)
                                if filename.endswith(Extension.covmat)
                            ]
                            for folder in installed_folders
                        ]
                    )
                )
            )
            assert num_files == len(covmat_database)
            log.debug("Loaded cached covmats database")
            _loaded_covmats_database[_hash] = covmat_database
            return covmat_database
        except Exception:
            log.info(
                "No cached covmat database present, not usable or not up-to-date. "
                "Will be re-created and cached."
            )
    # Create it (again)
    covmat_database = {}
    for folder_full in installed_folders:
        for filename in os.listdir(folder_full):
            try:
                with open(
                    os.path.join(folder_full, filename), encoding="utf-8-sig"
                ) as covmat:
                    header = covmat.readline()
                assert header.strip().startswith("#")
                params = header.strip().lstrip("#").split()
            except Exception:
                continue
            name = os.path.splitext(filename)[0]
            tags = name.replace(".post.", "_").replace("_post", "").split("_")
            partags = set(tags).intersection(params)
            datatags = set(tags[1:]) - partags
            key = covmat_file_key(partags, datatags, tags[0])
            covmat_database[key] = {
                "folder": folder_full,
                "name": filename,
                "params": params,
            }
    if cached:
        with open(covmats_database_fullpath, "wb") as f:
            pickle.dump(covmat_database, f)
        _loaded_covmats_database[_hash] = covmat_database
    return covmat_database


def get_best_covmat(info, packages_path=None, cached=True):
    """
    Chooses optimal covmat from a database, based on common parameters and likelihoods.
    Only used by GUI.

    Returns a dict `{folder: [folder_of_covmat], name: [file_name_of_covmat],
    params: [parameters_in_covmat], covmat: [covariance_matrix]}`.
    """

    if not (packages_path := packages_path or info.get(packages_path_input)):
        raise LoggedError(log, "Needs a path to the external packages' installation.")
    updated_info = update_info(info, strict=False)
    for p, pinfo in list(updated_info["params"].items()):
        if not is_sampled_param(pinfo):
            updated_info["params"].pop(p)
    info_sampled_params = updated_info["params"]
    if not (
        covmat_data := get_best_covmat_ext(
            get_covmat_package_folders(packages_path),
            updated_info["params"],
            updated_info["likelihood"],
            cached,
        )
    ):
        return None
    covmat = np.atleast_2d(
        np.loadtxt(os.path.join(covmat_data["folder"], covmat_data["name"]))
    )
    params_in_covmat = get_translated_params(info_sampled_params, covmat_data["params"])
    indices = [covmat_data["params"].index(p) for p in params_in_covmat.values()]
    covmat_data["covmat"] = covmat[indices][:, indices]
    covmat_data["params"] = params_in_covmat
    return covmat_data


def get_best_covmat_ext(
    covmat_dirs,
    params_info,
    likelihoods_info,
    cached=True,
    job_item=None,
    cov_map=empty_dict,
) -> dict | None:
    """
    Actual covmat finder used by `get_best_covmat`. Call directly for more control on
    the parameters used.

    Returns the same dict as `get_best_covmat`, except for the covariance matrix itself.
    """
    if not (covmats_database := get_covmat_database(covmat_dirs, cached=cached)):
        log.warning("No covariance matrices found at %s" % covmat_dirs)
        return None

    if job_item:
        key_tuple = covmat_file_key(
            job_item.param_set, job_item.data_set.names, job_item.base
        )

        # match all data tags and param tags independent of order
        if match := covmats_database.get(key_tuple):
            return match
        # match without base
        for tup, item in covmats_database.items():
            if tup[:2] == key_tuple[:2]:
                return item
        # match dropping "without" names
        keys = {key_tuple}
        for remove in cov_map.get("without") or []:
            for param, data, base in keys.copy():
                key = covmat_file_key(set(param) - {remove}, set(data) - {remove}, base)
                if match := covmats_database.get(key):
                    return match
                keys.add(key)
        # match using rename dict
        if rename := cov_map.get("rename"):
            renames = {x: (v,) if isinstance(v, str) else v for x, v in rename.items()}
            for param, data, base in keys.copy():
                key = covmat_file_key(
                    chain(*[renames.get(p, [p]) for p in param]),
                    chain(*[renames.get(p, [p]) for p in data]),
                    rename.get(base, base),
                )
                if match := covmats_database.get(key):
                    return match
                keys.add(key)
        # include all renamed tag variants
        key_tuple = covmat_file_key(
            chain(*[x.paramtags for x in keys]),
            chain(*[x.datatags for x in keys]),
            key_tuple.base,
        )
    else:
        key_tuple = None

    # Prepare params and likes aliases
    params_renames = set(
        chain(
            *[
                [p] + str_to_list(info.get("renames", []))
                for p, info in params_info.items()
            ]
        )
    )
    likes_renames = set(
        chain(
            *[
                [like] + str_to_list((info or {}).get("aliases", []))
                for like, info in likelihoods_info.items()
            ]
        )
    )
    delimiters = r"[_\.]"
    likes_regexps = [
        re.compile(delimiters + re.escape(_like) + delimiters) for _like in likes_renames
    ]

    # Match number of params
    def score_params(_key, covmat):
        return len(set(covmat["params"]).intersection(params_renames))

    if not (best_p := get_best_score(covmats_database, score_params, 0)):
        log.warning(
            ((job_item.name + ":\n") if job_item else "")
            + "No covariance matrix found including at least "
            "one of the given parameters"
        )
        return None

    # Match likelihood names / keywords
    # No debug print here: way too many!
    def score_likes(_key: CovmatFileKey, covmat):
        if key_tuple:
            return len(
                _key.datatags.intersection(likes_renames.union(key_tuple.datatags))
            )
        return len([0 for r in likes_regexps if r.search(covmat["name"])])

    best_p_l = get_best_score(best_p, score_likes)
    if is_debug(log):
        log.debug(
            "Subset based on params + likes:\n - "
            + "\n - ".join([b["name"] for b in best_p_l.values()])
        )

    if key_tuple:

        def score_left_params(_key, _covmat):
            return -len(_key.paramtags - params_renames.union(key_tuple.paramtags))

        best_p_l = get_best_score(best_p_l, score_left_params)

    # Finally, in case there is more than one, select shortest #params and name (simpler!)
    # #params first, to avoid extended models with shorter covmat name
    def score_simpler_params(_key, _covmat):
        return -len(_covmat["params"])

    best_p_l_sp = get_best_score(best_p_l, score_simpler_params)
    if is_debug(log):
        log.debug(
            "Subset based on params + likes + fewest params:\n - "
            + "\n - ".join([b["name"] for b in best_p_l_sp.values()])
        )

    def score_simpler_name(_key, _covmat):
        return -len(_key.datatags)

    best_p_l_sp_sn = get_best_score(best_p_l_sp, score_simpler_name)
    if is_debug(log):
        log.debug(
            "Subset based on params + likes + fewest params + shortest name:\n - "
            + "\n - ".join([b["name"] for b in best_p_l_sp_sn.values()])
        )
    # if there is more than one (unlikely), just take first
    if len(best_p_l_sp_sn) > 1:
        log.warning(
            ((job_item.name + ":\n") if job_item else "")
            + "WARNING: using first of >1 possible best covmats: %r",
            [b["name"] for b in best_p_l_sp_sn.values()],
        )
    return next(iter(best_p_l_sp_sn.values())).copy()


def get_best_score(covmats, score_func, min_score=None) -> dict:
    scores = np.array([score_func(*x) for x in covmats.items()])
    if min_score is not None and np.max(scores) <= min_score:
        return {}
    i_max = np.argwhere(scores == np.max(scores)).T[0]
    return dict(val for i, val in enumerate(covmats.items()) if i in i_max)
