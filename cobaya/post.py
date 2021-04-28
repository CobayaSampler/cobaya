"""
.. module:: post

:Synopsis: Post-processing functions
:Author: Jesus Torrado and Antony Lewis

"""

# Global
import os
import logging
from itertools import chain
import numpy as np
import sys
from typing import List, Union, NamedTuple

# Local
from cobaya.parameterization import Parameterization
from cobaya.parameterization import is_fixed_or_function_param, is_sampled_param, \
    is_derived_param
from cobaya.conventions import _prior_1d_name, _debug, _debug_file, _output_prefix, \
    _post, _params, _prior, kinds, _weight, _resume, _separator, _get_chi2_name, \
    _minuslogpost, _force, partag, _minuslogprior, _packages_path, \
    _separator_files, _post_add, _post_remove, _post_suffix, _undo_chi2_name
from cobaya.conventions import ParamValuesDict, InputDict, InfoDict
from cobaya.collection import Collection
from cobaya.log import logger_setup, LoggedError
from cobaya.input import update_info, add_aggregated_chi2_params, load_input_dict
from cobaya.output import get_output
from cobaya import mpi
from cobaya.tools import progress_bar, recursive_update, deepcopy_where_possible, \
    check_deprecated_modules_path, str_to_list
from cobaya.model import Model
from cobaya.prior import Prior

if sys.version_info >= (3, 8):
    from typing import TypedDict


    class ResultDict(TypedDict):
        sample: Union[Collection, List[Collection]]
        stats: ParamValuesDict
        weights: Union[np.ndarray, List[np.ndarray]]
else:
    ResultDict = InfoDict

_minuslogprior_1d_name = _minuslogprior + _separator + _prior_1d_name
_default_post_cache_size = 2000


class PostTuple(NamedTuple):
    info: InputDict
    products: ResultDict


def value_or_list(lst: list):
    if len(lst) == 1:
        return lst[0]
    else:
        return lst


# Dummy classes for loading chains for post processing

class DummyModel:

    def __init__(self, info_params, info_likelihood, info_prior=None):
        self.parameterization = Parameterization(info_params, ignore_unused_sampled=True)
        self.prior = [_prior_1d_name] + list(info_prior or [])
        self.likelihood = list(info_likelihood)


@mpi.sync_state
def post(info_or_yaml_or_file: Union[InputDict, str, os.PathLike],
         sample: Union[Collection, List[Collection], None] = None
         ) -> PostTuple:
    info = load_input_dict(info_or_yaml_or_file)
    logger_setup(info.get(_debug), info.get(_debug_file))
    log = logging.getLogger(__name__.split(".")[-1])
    # MARKED FOR DEPRECATION IN v3.0
    # BEHAVIOUR TO BE REPLACED BY ERROR:
    check_deprecated_modules_path(info)
    # END OF DEPRECATION BLOCK
    info_post = info.get(_post)
    if not info_post:
        raise LoggedError(log, "No 'post' block given. Nothing to do!")
    if mpi.is_main_process() and info.get(_resume):
        log.warning("Resuming not implemented for post-processing. Re-starting.")
    if not info.get(_output_prefix) and info_post.get(_output_prefix) \
            and not info.get(_params):
        raise LoggedError(log, "The input dictionary must have be a full option "
                               "dictionary, or have an existing 'output' root to load "
                               "previous settings from ('output' to read from is in the "
                               "main block not under 'post'). ")
    # 1. Load existing sample
    output_in = get_output(prefix=info.get(_output_prefix))
    if output_in:
        info_in = output_in.load_updated_info()
        if info_in is None:
            info_in = update_info(info)
    else:
        info_in = update_info(info)
    dummy_model_in = DummyModel(info_in[_params], info_in.get(kinds.likelihood, {}),
                                info_in.get(_prior))
    in_collections = []
    thin = info_post.get("thin", 1)
    skip = info_post.get("skip", 0)
    if info.get('thin') is not None or info.get('skip') is not None:
        raise LoggedError(log, "'thin' and 'skip' should be "
                               "parameters of the 'post' block")

    if sample:
        # If MPI, assume for each MPI process post is passed in the list of
        # collections that should be processed by that process
        # (e.g. single chain output from sampler)
        if isinstance(sample, Collection):
            in_collections = [sample]
        else:
            in_collections = sample
        for i, collection in enumerate(in_collections):
            if skip:
                if 0 < skip < 1:
                    skip = int(round(skip * len(collection)))
                collection = collection.filtered_copy(slice(skip, None))
            if thin != 1:
                collection = collection.thin_samples(thin)
            in_collections[i] = collection
    elif output_in:
        files = output_in.find_collections()
        numbered = files
        if not numbered:
            # look for un-numbered output files
            files = output_in.find_collections(name=False)
        if files:
            if mpi.size() > len(files):
                raise LoggedError(log, "Number of MPI processes (%s) is larger than "
                                       "the number of sample files (%s)",
                                  mpi.size(), len(files))
            for num in range(mpi.rank(), len(files), mpi.size()):
                in_collections += [Collection(dummy_model_in, output_in,
                                              onload_thin=thin, onload_skip=skip,
                                              load=True, file_name=files[num],
                                              name=str(num + 1) if numbered else "")]
        else:
            raise LoggedError(log, "No samples found for the input model with prefix %s",
                              os.path.join(output_in.folder, output_in.prefix))

    else:
        raise LoggedError(log, "No output from where to load from, "
                               "nor input collections given.")
    if any(len(c) <= 1 for c in in_collections):
        raise LoggedError(
            log, "Not enough samples for post-processing. Try using a larger sample, "
                 "or skipping or thinning less.")
    mpi.sync_processes()
    log.info("Will process %d sample points.", sum(len(c) for c in in_collections))

    # 2. Compare old and new info: determine what to do
    add = info_post.get(_post_add) or {}
    if _post_remove in add:
        raise LoggedError(log, "remove block should be under 'post', not 'add'")
    remove = info_post.get(_post_remove) or {}
    # Add a dummy 'one' likelihood, to absorb unused parameters
    if not add.get(kinds.likelihood):
        add[kinds.likelihood] = {}
    add[kinds.likelihood]["one"] = None
    # Expand the "add" info, but don't add new default sampled parameters
    orig_params = set(add.get(_params) or [])
    add = update_info(add)
    for p in set(add[_params]) - orig_params:
        if p in info_in[_params]:
            add[_params].pop(p)

    # 2.1 Adding/removing derived parameters and changes in priors of sampled parameters
    out_combined = {_params: deepcopy_where_possible(info_in[_params])}
    remove_params = list(str_to_list(remove.get(_params)) or [])
    for p in remove_params:
        pinfo = info_in[_params].get(p)
        if pinfo is None or not is_derived_param(pinfo):
            raise LoggedError(
                log,
                "You tried to remove parameter '%s', which is not a derived parameter. "
                "Only derived parameters can be removed during post-processing.", p)
        out_combined[_params].pop(p)
    # Force recomputation of aggregated chi2
    for p in list(out_combined[_params]):
        if p.startswith(_get_chi2_name("")):
            out_combined[_params].pop(p)
    prior_recompute_1d = False
    for p, pinfo in (add.get(_params) or {}).items():
        pinfo_in = info_in[_params].get(p)
        if is_sampled_param(pinfo):
            if not is_sampled_param(pinfo_in):
                # No added sampled parameters (de-marginalisation not implemented)
                if pinfo_in is None:
                    raise LoggedError(
                        log, "You added a new sampled parameter %r (maybe accidentally "
                             "by adding a new likelihood that depends on it). "
                             "Adding new sampled parameters is not possible. Try fixing "
                             "it to some value.", p)
                else:
                    raise LoggedError(
                        log,
                        "You tried to change the prior of parameter '%s', "
                        "but it was not a sampled parameter. "
                        "To change that prior, you need to define as an external one.", p)
            # recompute prior if potentially changed sampled parameter priors
            prior_recompute_1d = True
        elif is_derived_param(pinfo):
            if p in out_combined[_params]:
                raise LoggedError(
                    log, "You tried to add derived parameter '%s', which is already "
                         "present. To force its recomputation, 'remove' it too.", p)
        elif is_fixed_or_function_param(pinfo):
            # Only one possibility left "fixed" parameter that was not present before:
            # input of new likelihood, or just an argument for dynamical derived (dropped)
            if ((p in info_in[_params] and
                 pinfo[partag.value] != (pinfo_in or {}).get(partag.value, None))):
                raise LoggedError(
                    log,
                    "You tried to add a fixed parameter '%s: %r' that was already present"
                    " but had a different value or was not fixed. This is not allowed. "
                    "The old info of the parameter was '%s: %r'",
                    p, dict(pinfo), p, dict(pinfo_in))
        elif not pinfo_in:  # OK as long as we have known value for it
            raise LoggedError(log, "Parameter %s no known value. ", p)
        out_combined[_params][p] = pinfo
    # Turn the rest of *derived* parameters into constants,
    # so that the likelihoods do not try to recompute them
    # But be careful to exclude *input* params that have a "derived: True" value
    # (which in "updated info" turns into "derived: 'lambda [x]: [x]'")
    # Don't assign to derived parameters to theories, only likelihoods, so they can be
    # recomputed if needed. If the theory does not need to be computed, it doesn't matter
    # if it is already assigned parameters in the usual way; likelihoods can get
    # the required derived parameters from the stored sample derived parameter inputs.
    out_params_with_computed = deepcopy_where_possible(out_combined[_params])
    dropped_theory = set()
    for p, pinfo in out_params_with_computed.items():
        if (is_derived_param(pinfo) and not (partag.value in pinfo)
                and p not in add.get(_params, {})):
            out_params_with_computed[p] = {partag.value: np.nan}
            dropped_theory.add(p)
    # 2.2 Manage adding/removing priors and likelihoods
    warn_remove = False
    for kind in [_prior, kinds.likelihood, kinds.theory]:
        out_combined[kind] = deepcopy_where_possible(info_in.get(kind)) or {}
        for remove_item in str_to_list(remove.get(kind)) or []:
            try:
                out_combined[kind].pop(remove_item, None)
                if remove_item not in add.get(kind, {}) or [] and kind != kinds.theory:
                    warn_remove = True
            except ValueError:
                raise LoggedError(
                    log, "Trying to remove %s '%s', but it is not present. "
                         "Existing ones: %r", kind, remove_item, list(out_combined[kind]))
        if kind != kinds.theory and kind in add:
            dups = set(add.get(kind)).intersection(out_combined[kind]) - {"one"}
            if dups:
                raise LoggedError(
                    log, "You have added %s '%s', which was already present. If you "
                         "want to force its recomputation, you must also 'remove' it.",
                    kind, dups)
            out_combined[kind].update(add[kind])

    if warn_remove and mpi.is_main_process():
        log.warning("You are removing a prior or likelihood pdf. "
                    "Notice that if the resulting posterior is much wider "
                    "than the original one, or displaced enough, "
                    "it is probably safer to explore it directly.")

    mlprior_names_add = [_minuslogprior + _separator + name for name in
                         (add.get(_prior) or [])]
    chi2_names_add = [_get_chi2_name(name) for name in add[kinds.likelihood] if
                      name != "one"]
    out_combined[kinds.likelihood].pop("one", None)

    add_theory = add.get(kinds.theory)
    if add_theory:
        if len(add[kinds.likelihood]) == 1 and not any(
                is_derived_param(pinfo) for pinfo in add.get(_params, {}).values()):
            log.warning("You are adding a theory, but this does not force recomputation "
                        "of any likelihood or derived parameters unless explicitly "
                        "removed+added.")
        # Inherit from the original chain (input|output_params, renames, etc)
        added_theory = add_theory.copy()
        for theory, theory_info in out_combined[kinds.theory].items():
            if theory in list(added_theory):
                out_combined[kinds.theory][theory] = \
                    recursive_update(theory_info, added_theory.pop(theory))
        out_combined[kinds.theory].update(added_theory)

    # Prepare recomputation of aggregated chi2
    # (they need to be recomputed by hand, because its auto-computation won't pick up
    #  old likelihoods for a given type)
    all_types = {like: str_to_list(opts.get("type") or [])
                 for like, opts in out_combined[kinds.likelihood].items()}
    types = set(chain(*all_types.values()))
    inv_types = {t: [like for like, like_types in all_types.items() if t in like_types]
                 for t in sorted(types)}
    add_aggregated_chi2_params(out_combined[_params], types)

    # 3. Create output collection
    if _post_suffix not in info_post:
        raise LoggedError(log, "You need to provide a '%s' for your output chains.",
                          _post_suffix)
    # Use default prefix if it exists. If it does not, produce no output by default.
    # {post: {output: None}} suppresses output, and if it's a string, updates it.
    out_prefix = info_post.get(_output_prefix, info.get(_output_prefix))
    if out_prefix not in [None, False]:
        out_prefix += _separator_files + _post + _separator_files + info_post[
            _post_suffix]
    output_out = get_output(prefix=out_prefix, force=info.get(_force))
    output_out.set_lock()

    if output_out and not output_out.force and output_out.find_collections():
        raise LoggedError(log, "Found existing post-processing output with prefix %r. "
                               "Delete it manually or re-run with `force: True` "
                               "(or `-f`, `--force` from the shell).", out_prefix)
    elif output_out and output_out.force and mpi.is_main_process():
        output_out.delete_infos()
        for _file in output_out.find_collections():
            output_out.delete_file_or_folder(_file)
    info_out = deepcopy_where_possible(info)
    info_post = info_post.copy()
    info_out[_post] = info_post
    # Updated with input info and extended (updated) add info
    info_out.update(info_in)
    info_post[_post_add] = add

    dummy_model_out = DummyModel(out_combined[_params], out_combined[kinds.likelihood],
                                 info_prior=out_combined[_prior])
    out_func_parameterization = Parameterization(out_params_with_computed)

    # TODO: check allow_renames=False?
    model_add = Model(out_params_with_computed, add[kinds.likelihood],
                      info_prior=add.get(_prior), info_theory=out_combined[kinds.theory],
                      packages_path=info_post.get(_packages_path) or
                                    info.get(_packages_path),
                      allow_renames=False, post=True,
                      stop_at_error=info.get('stop_at_error', False),
                      skip_unused_theories=True, dropped_theory_params=dropped_theory)
    # Remove auxiliary "one" before dumping -- 'add' *is* info_out[_post][_post_add]
    add[kinds.likelihood].pop("one")
    out_collections = [Collection(dummy_model_out, output_out, name=c.name,
                                  cache_size=_default_post_cache_size)
                       for c in in_collections]
    # TODO: should maybe add skip/thin to out_combined, so can tell post-processed?
    output_out.check_and_dump_info(info_out, out_combined, check_compatible=False)
    collection_in = in_collections[0]
    collection_out = out_collections[0]

    last_percent = None
    known_constants = dummy_model_out.parameterization.constant_params()
    known_constants.update(dummy_model_in.parameterization.constant_params())
    missing_params = dummy_model_in.parameterization.sampled_params().keys() - set(
        collection_in.columns)
    if missing_params:
        raise LoggedError(log, "Input samples do not contain expected sampled parameter "
                               "values: %s", missing_params)

    missing_priors = set(name for name in collection_out.minuslogprior_names if
                         name not in mlprior_names_add
                         and name not in collection_in.columns)
    if _minuslogprior_1d_name in missing_priors:
        prior_recompute_1d = True
    if prior_recompute_1d:
        missing_priors.discard(_minuslogprior_1d_name)
        mlprior_names_add.insert(0, _minuslogprior_1d_name)
    if missing_priors and _prior in info_in:
        # in case there are input priors that are not stored in input samples
        # e.g. when postprocessing GetDist/CosmoMC-format chains
        info_prior = {piname: info_in[_prior][piname] for piname in info_in[_prior] if
                      (_minuslogprior + _separator + piname in missing_priors)}
        regenerated_prior_names = [_minuslogprior + _separator + piname for piname in
                                   info_prior]
        missing_priors.difference_update(regenerated_prior_names)
        prior_regenerate = Prior(dummy_model_in.parameterization, info_prior)
    else:
        prior_regenerate = None
        regenerated_prior_names = None
    if missing_priors:
        raise LoggedError(log, "Missing priors: %s", missing_priors)

    mpi.sync_processes()
    output_in.check_lock()

    # 4. Main loop! Loop over input samples and adjust as required.
    if mpi.is_main_process():
        log.info("Running post-processing...")
    difflogmax = -np.inf
    to_do = sum(len(c) for c in in_collections)
    done = 0
    for collection_in, collection_out in zip(in_collections, out_collections):
        for i, point in collection_in.data.iterrows():
            all_params = point.to_dict()
            for p in remove_params:
                all_params.pop(p, None)
            log.debug("Point: %r", point)
            sampled = np.array([all_params[param] for param in
                                dummy_model_in.parameterization.sampled_params()])
            all_params = out_func_parameterization.to_input(all_params)

            # Add/remove priors
            if prior_recompute_1d:
                priors_add = [model_add.prior.logps_internal(sampled)]
                if priors_add[0] == -np.inf:
                    continue
            else:
                priors_add = []
            if model_add.prior.external:
                priors_add.extend(model_add.prior.logps_external(all_params))

            logpriors_add = dict(zip(mlprior_names_add, priors_add))
            logpriors_new = [logpriors_add.get(name, - point.get(name, 0))
                             for name in collection_out.minuslogprior_names]
            if prior_regenerate:
                regenerated = dict(
                    zip(regenerated_prior_names,
                        prior_regenerate.logps_external(all_params)))
                for _i, name in enumerate(collection_out.minuslogprior_names):
                    if name in regenerated_prior_names:
                        logpriors_new[_i] = regenerated[name]

            if log.getEffectiveLevel() <= logging.DEBUG:
                log.debug("New set of priors: %r",
                          dict(zip(dummy_model_out.prior, logpriors_new)))
            if -np.inf in logpriors_new:
                continue
            # Add/remove likelihoods and/or (re-)calculate derived parameters
            loglikes_add, output_derived = model_add.logps(all_params)
            loglikes_add = dict(zip(chi2_names_add, loglikes_add))
            output_derived = dict(zip(model_add.output_params, output_derived))
            loglikes_new = [loglikes_add.get(name, -0.5 * point.get(name, 0))
                            for name in collection_out.chi2_names]
            if log.getEffectiveLevel() <= logging.DEBUG:
                log.debug(
                    "New set of likelihoods: %r",
                    dict(zip(dummy_model_out.likelihood, loglikes_new)))
                if output_derived:
                    log.debug("New set of derived parameters: %r", output_derived)
            if -np.inf in loglikes_new:
                continue
            all_params.update(output_derived)

            all_params.update(out_func_parameterization.to_derived(all_params))
            derived = {param: all_params.get(param)
                       for param in dummy_model_out.parameterization.derived_params()}
            # We need to recompute the aggregated chi2 by hand
            for type_, likes in inv_types.items():
                derived[_get_chi2_name(type_)] = sum(
                    -2 * lvalue for lname, lvalue
                    in zip(collection_out.chi2_names, loglikes_new)
                    if _undo_chi2_name(lname) in likes)
            if log.getEffectiveLevel() <= logging.DEBUG:
                log.debug("New derived parameters: %r",
                          {p: derived[p]
                           for p in dummy_model_out.parameterization.derived_params()
                           if p in add[_params]})
            # Save to the collection (keep old weight for now)
            collection_out.add(
                sampled, derived=derived.values(), weight=point.get(_weight),
                logpriors=logpriors_new, loglikes=loglikes_new)
            mpi.check_errors()
            # Display progress
            percent = int(np.round((i + done) / to_do * 100))
            if percent != last_percent and not percent % 5:
                last_percent = percent
                progress_bar(log, percent, " (%d/%d)" % (i + done, to_do))
        if not collection_out.data.last_valid_index():
            raise LoggedError(
                log, "No elements in the final sample. Possible causes: "
                     "added a prior or likelihood valued zero over the full sampled "
                     "domain, or the computation of the theory failed everywhere, etc.")

        difflogmax = max(difflogmax, max(collection_in[_minuslogpost]
                                         - collection_out[_minuslogpost]))
        done += len(collection_in)

        # Reweight -- account for large dynamic range!
        #   Prefer to rescale +inf to finite, and ignore final points with -inf.
        #   Remove -inf's (0-weight), and correct indices

    difflogmax = max(mpi.allgather(difflogmax))

    points = 0
    tot_weight = 0
    min_weight = np.inf
    max_weight = -np.inf
    sum_w2 = 0
    points_removed = 0
    weights = []
    for collection_in, collection_out in zip(in_collections, out_collections):
        importance_weights = np.exp(
            collection_in[_minuslogpost] - collection_out[_minuslogpost] - difflogmax)
        weights.append(importance_weights)
        collection_out.reweight(importance_weights)
        # Write!
        collection_out.out_update()
        output_weights = collection_out[_weight]
        points += len(collection_out)
        tot_weight += np.sum(output_weights)
        points_removed += len(importance_weights) - len(output_weights)
        min_weight = min(min_weight, np.min(importance_weights))
        max_weight = max(max_weight, np.max(output_weights))
        sum_w2 += np.dot(output_weights, output_weights)
    tot_weight, min_weight, max_weight, sum_w2, points, points_removed = \
        mpi.zip_gather([tot_weight, min_weight, max_weight, sum_w2,
                        points, points_removed])
    if mpi.is_main_process():
        output_out.clear_lock()
        log.info("Finished! Final number of distinct sample points: %s", sum(points))
        log.info("Minimum scaled importance weight: %.4g", min(min_weight))
        if sum(points_removed):
            log.info("Points deleted due to zero weight: %s", sum(points_removed))
        log.info("Effective number of single samples if independent (sum w)/max(w): %s",
                 int(sum(tot_weight) / max(max_weight)))
        log.info(
            "Effective number of weighted samples if independent (sum w)^2/sum(w^2): "
            "%s", int(sum(tot_weight) ** 2 / sum(sum_w2)))
    products = {"sample": value_or_list(out_collections),
                "stats": {'min_weight': min(min_weight),
                          'points_removed': sum(points_removed),
                          'tot_weight': sum(tot_weight),
                          'max_weight': max(max_weight),
                          'sum_w2': sum(sum_w2),
                          'points': sum(points)},
                "weights": value_or_list(weights)}
    return PostTuple(info=out_combined, products=products)
