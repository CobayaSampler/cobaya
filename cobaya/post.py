"""
.. module:: post

:Synopsis: Post-processing functions
:Author: Jesus Torrado and Antony Lewis

"""

import os
import sys
import time
from itertools import chain
from typing import List, Union, Optional, Tuple
import numpy as np

from cobaya import mpi
from cobaya.collection import SampleCollection
from cobaya.conventions import prior_1d_name, OutPar, get_chi2_name, \
    undo_chi2_name, get_minuslogpior_name, separator_files, minuslogprior_names, \
    packages_path_input
from cobaya.input import update_info, add_aggregated_chi2_params, load_input_dict
from cobaya.log import logger_setup, get_logger, is_debug, LoggedError
from cobaya.model import Model
from cobaya.output import get_output
from cobaya.parameterization import Parameterization
from cobaya.parameterization import is_fixed_or_function_param, is_sampled_param, \
    is_derived_param
from cobaya.prior import Prior
from cobaya.tools import progress_bar, recursive_update, deepcopy_where_possible, \
    str_to_list
from cobaya.typing import ExpandedParamsDict, ModelBlock, ParamValuesDict, InputDict, \
    InfoDict, PostDict

if sys.version_info >= (3, 8):
    from typing import TypedDict


    class PostResultDict(TypedDict):
        sample: Union[SampleCollection, List[SampleCollection]]
        stats: ParamValuesDict
        logpost_weight_offset: float
        weights: Union[np.ndarray, List[np.ndarray]]
else:
    globals()['PostResultDict'] = InfoDict

_minuslogprior_1d_name = get_minuslogpior_name(prior_1d_name)


class OutputOptions:
    default_post_cache_size = 2000
    # to reweight we need to know absolute size of log likelihoods
    # but want to dump out regularly, so set _reweight_after as minimum to check first
    reweight_after = 100
    output_inteveral_s = 60


def value_or_list(lst: list):
    if len(lst) == 1:
        return lst[0]
    else:
        return lst


# Dummy classes for loading chains for post processing

class DummyModel:

    def __init__(self, info_params, info_likelihood, info_prior=None):
        self.parameterization = Parameterization(info_params, ignore_unused_sampled=True)
        self.prior = [prior_1d_name] + list(info_prior or [])
        self.likelihood = list(info_likelihood)


@mpi.sync_state
def post(info_or_yaml_or_file: Union[InputDict, str, os.PathLike],
         sample: Union[SampleCollection, List[SampleCollection], None] = None
         ) -> Tuple[InputDict, PostResultDict]:
    info = load_input_dict(info_or_yaml_or_file)
    # MARKED FOR DEPRECATION IN v3.2
    if info.get("debug_file"):
        print("*WARNING* 'debug_file' will soon be deprecated. If you want to "
              "save the debug output to a file, use 'debug: [filename]'.")
        # BEHAVIOUR TO BE REPLACED BY AN ERROR
        if info.get("debug"):
            info["debug"] = info.pop("debug_file")
    # END OF DEPRECATION BLOCK
    logger_setup(info.get("debug"))
    log = get_logger(__name__)
    info_post: PostDict = info.get("post") or {}
    if not info_post:
        raise LoggedError(log, "No 'post' block given. Nothing to do!")
    if mpi.is_main_process() and info.get("resume"):
        log.warning("Resuming not implemented for post-processing. Re-starting.")
    if not info.get("output") and info_post.get("output") \
            and not info.get("params"):
        raise LoggedError(log, "The input dictionary must have be a full option "
                               "dictionary, or have an existing 'output' root to load "
                               "previous settings from ('output' to read from is in the "
                               "main block not under 'post'). ")
    # 1. Load existing sample
    output_in = get_output(prefix=info.get("output"))
    if output_in:
        info_in = output_in.load_updated_info() or update_info(info)
    else:
        info_in = update_info(info)
    params_in: ExpandedParamsDict = info_in["params"]  # type: ignore
    dummy_model_in = DummyModel(params_in, info_in.get("likelihood", {}),
                                info_in.get("prior"))

    in_collections = []
    thin = info_post.get("thin", 1)
    skip = info_post.get("skip", 0)
    if info.get('thin') is not None or info.get('skip') is not None:  # type: ignore
        raise LoggedError(log, "'thin' and 'skip' should be "
                               "parameters of the 'post' block")

    if sample:
        # If MPI, assume for each MPI process post is passed in the list of
        # collections that should be processed by that process
        # (e.g. single chain output from sampler)
        if isinstance(sample, SampleCollection):
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
                in_collections += [SampleCollection(
                    dummy_model_in, output_in,
                    onload_thin=thin, onload_skip=skip, load=True, file_name=files[num],
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
    add = info_post.get("add") or {}
    if "remove" in add:
        raise LoggedError(log, "remove block should be under 'post', not 'add'")
    remove = info_post.get("remove") or {}
    # Add a dummy 'one' likelihood, to absorb unused parameters
    if not add.get("likelihood"):
        add["likelihood"] = {}
    add["likelihood"]["one"] = None
    # Expand the "add" info, but don't add new default sampled parameters
    orig_params = set(add.get("params") or [])
    add = update_info(add, add_aggr_chi2=False)
    add_params: ExpandedParamsDict = add["params"]  # type: ignore
    for p in set(add_params) - orig_params:
        if p in params_in:
            add_params.pop(p)

    # 2.1 Adding/removing derived parameters and changes in priors of sampled parameters
    out_combined_params = deepcopy_where_possible(params_in)
    remove_params = list(str_to_list(remove.get("params")) or [])
    for p in remove_params:
        pinfo = params_in.get(p)
        if pinfo is None or not is_derived_param(pinfo):
            raise LoggedError(
                log,
                "You tried to remove parameter '%s', which is not a derived parameter. "
                "Only derived parameters can be removed during post-processing.", p)
        out_combined_params.pop(p)
    # Force recomputation of aggregated chi2
    for p in list(out_combined_params):
        if p.startswith(get_chi2_name("")):
            out_combined_params.pop(p)
    prior_recompute_1d = False
    for p, pinfo in add_params.items():
        pinfo_in = params_in.get(p)
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
            if p in out_combined_params:
                raise LoggedError(
                    log, "You tried to add derived parameter '%s', which is already "
                         "present. To force its recomputation, 'remove' it too.", p)
        elif is_fixed_or_function_param(pinfo):
            # Only one possibility left "fixed" parameter that was not present before:
            # input of new likelihood, or just an argument for dynamical derived (dropped)
            if pinfo_in and p in params_in and pinfo["value"] != pinfo_in.get("value"):
                raise LoggedError(
                    log,
                    "You tried to add a fixed parameter '%s: %r' that was already present"
                    " but had a different value or was not fixed. This is not allowed. "
                    "The old info of the parameter was '%s: %r'",
                    p, dict(pinfo), p, dict(pinfo_in))
        elif not pinfo_in:  # OK as long as we have known value for it
            raise LoggedError(log, "Parameter %s no known value. ", p)
        out_combined_params[p] = pinfo

    out_combined: InputDict = {"params": out_combined_params}  # type: ignore
    # Turn the rest of *derived* parameters into constants,
    # so that the likelihoods do not try to recompute them
    # But be careful to exclude *input* params that have a "derived: True" value
    # (which in "updated info" turns into "derived: 'lambda [x]: [x]'")
    # Don't assign to derived parameters to theories, only likelihoods, so they can be
    # recomputed if needed. If the theory does not need to be computed, it doesn't matter
    # if it is already assigned parameters in the usual way; likelihoods can get
    # the required derived parameters from the stored sample derived parameter inputs.
    out_params_with_computed = deepcopy_where_possible(out_combined_params)

    dropped_theory = set()
    for p, pinfo in out_params_with_computed.items():
        if (is_derived_param(pinfo) and "value" not in pinfo
                and p not in add_params):
            out_params_with_computed[p] = {"value": np.nan}
            dropped_theory.add(p)
    # 2.2 Manage adding/removing priors and likelihoods
    warn_remove = False
    kind: ModelBlock
    for kind in ("prior", "likelihood", "theory"):
        out_combined[kind] = deepcopy_where_possible(info_in.get(kind)) or {}
        for remove_item in str_to_list(remove.get(kind)) or []:
            try:
                out_combined[kind].pop(remove_item, None)
                if remove_item not in (add.get(kind) or []) and kind != "theory":
                    warn_remove = True
            except ValueError:
                raise LoggedError(
                    log, "Trying to remove %s '%s', but it is not present. "
                         "Existing ones: %r", kind, remove_item, list(out_combined[kind]))
        if kind != "theory" and kind in add:
            dups = set(add.get(kind) or []).intersection(out_combined[kind]) - {"one"}
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

    mlprior_names_add = minuslogprior_names(add.get("prior") or [])
    chi2_names_add = [get_chi2_name(name) for name in add["likelihood"] if
                      name != "one"]
    out_combined["likelihood"].pop("one", None)

    add_theory = add.get("theory")
    if add_theory:
        if len(add["likelihood"]) == 1 and not any(
                is_derived_param(pinfo) for pinfo in add_params.values()):
            log.warning("You are adding a theory, but this does not force recomputation "
                        "of any likelihood or derived parameters unless explicitly "
                        "removed+added.")
        # Inherit from the original chain (input|output_params, renames, etc)
        added_theory = add_theory.copy()
        for theory, theory_info in out_combined["theory"].items():
            if theory in list(added_theory):
                out_combined["theory"][theory] = \
                    recursive_update(theory_info, added_theory.pop(theory))
        out_combined["theory"].update(added_theory)

    # Prepare recomputation of aggregated chi2
    # (they need to be recomputed by hand, because auto-computation won't pick up
    #  old likelihoods for a given type)
    all_types = {like: str_to_list(opts.get("type") or [])
                 for like, opts in out_combined["likelihood"].items()}
    types = set(chain(*all_types.values()))
    inv_types = {t: [like for like, like_types in all_types.items() if t in like_types]
                 for t in sorted(types)}
    add_aggregated_chi2_params(out_combined_params, types)

    # 3. Create output collection
    # Use default prefix if it exists. If it does not, produce no output by default.
    # {post: {output: None}} suppresses output, and if it's a string, updates it.
    out_prefix = info_post.get("output", info.get("output"))
    if out_prefix:
        suffix = info_post.get("suffix")
        if not suffix:
            raise LoggedError(log, "You need to provide a '%s' for your output chains.",
                              "suffix")
        out_prefix += separator_files + "post" + separator_files + suffix
    output_out = get_output(prefix=out_prefix, force=info.get("force"))
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
    info_out["post"] = info_post
    # Updated with input info and extended (updated) add info
    info_out.update(info_in)  # type: ignore
    info_post["add"] = add

    dummy_model_out = DummyModel(out_combined_params, out_combined["likelihood"],
                                 info_prior=out_combined["prior"])
    out_func_parameterization = Parameterization(out_params_with_computed)

    # TODO: check allow_renames=False?
    model_add = Model(out_params_with_computed, add["likelihood"],
                      info_prior=add.get("prior"), info_theory=out_combined["theory"],
                      packages_path=(info_post.get(packages_path_input) or
                                     info.get(packages_path_input)),
                      allow_renames=False, post=True,
                      stop_at_error=info.get('stop_at_error', False),
                      skip_unused_theories=True, dropped_theory_params=dropped_theory)
    # Remove auxiliary "one" before dumping -- 'add' *is* info_out["post"]["add"]
    add["likelihood"].pop("one")
    out_collections = [SampleCollection(dummy_model_out, output_out, name=c.name,
                                        cache_size=OutputOptions.default_post_cache_size)
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
    prior_regenerate: Optional[Prior]
    if missing_priors and "prior" in info_in:
        # in case there are input priors that are not stored in input samples
        # e.g. when postprocessing GetDist/CosmoMC-format chains
        in_names = minuslogprior_names(info_in["prior"])
        info_prior = {piname: inf for (piname, inf), in_name in
                      zip(info_in["prior"].items(), in_names) if
                      in_name in missing_priors}
        regenerated_prior_names = minuslogprior_names(info_prior)
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
    difflogmax: Optional[float] = None
    to_do = sum(len(c) for c in in_collections)
    weights = []
    done = 0
    last_dump_time = time.time()
    for collection_in, collection_out in zip(in_collections, out_collections):
        importance_weights = []

        def set_difflogmax():
            nonlocal difflogmax
            difflog = (collection_in[OutPar.minuslogpost].to_numpy(
                dtype=np.float64)[:len(collection_out)]
                       - collection_out[OutPar.minuslogpost].to_numpy(dtype=np.float64))
            difflogmax = np.max(difflog)
            if abs(difflogmax) < 1:
                difflogmax = 0  # keep simple when e.g. very similar
            log.debug("difflogmax: %g", difflogmax)
            if mpi.more_than_one_process():
                difflogmax = max(mpi.allgather(difflogmax))
            if mpi.is_main_process():
                log.debug("Set difflogmax: %g", difflogmax)
            _weights = np.exp(difflog - difflogmax)
            importance_weights.extend(_weights)
            collection_out.reweight(_weights)

        for i, point in collection_in.data.iterrows():
            all_params = point.to_dict()
            for p in remove_params:
                all_params.pop(p, None)
            log.debug("Point: %r", point)
            sampled = np.array([all_params[param] for param in
                                dummy_model_in.parameterization.sampled_params()])
            all_params = out_func_parameterization.to_input(all_params).copy()

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
                regenerated = dict(zip(regenerated_prior_names,
                                       prior_regenerate.logps_external(all_params)))
                for _i, name in enumerate(collection_out.minuslogprior_names):
                    if name in regenerated_prior_names:
                        logpriors_new[_i] = regenerated[name]

            if is_debug(log):
                log.debug("New set of priors: %r",
                          dict(zip(dummy_model_out.prior, logpriors_new)))
            if -np.inf in logpriors_new:
                continue
            # Add/remove likelihoods and/or (re-)calculate derived parameters
            loglikes_add, output_derived = model_add._loglikes_input_params(
                all_params, return_output_params=True)
            loglikes_add = dict(zip(chi2_names_add, loglikes_add))
            output_derived = dict(zip(model_add.output_params, output_derived))
            loglikes_new = [loglikes_add.get(name, -0.5 * point.get(name, 0))
                            for name in collection_out.chi2_names]
            if is_debug(log):
                log.debug("New set of likelihoods: %r",
                          dict(zip(dummy_model_out.likelihood, loglikes_new)))
                if output_derived:
                    log.debug("New set of derived parameters: %r", output_derived)
            if -np.inf in loglikes_new:
                continue
            all_params.update(output_derived)

            all_params.update(out_func_parameterization.to_derived(all_params))
            derived = {param: all_params.get(param) for param in
                       dummy_model_out.parameterization.derived_params()}
            # We need to recompute the aggregated chi2 by hand
            for type_, likes in inv_types.items():
                derived[get_chi2_name(type_)] = sum(
                    -2 * lvalue for lname, lvalue
                    in zip(collection_out.chi2_names, loglikes_new)
                    if undo_chi2_name(lname) in likes)
            if is_debug(log):
                log.debug("New derived parameters: %r",
                          {p: derived[p]
                           for p in dummy_model_out.parameterization.derived_params()
                           if p in add["params"]})
            # Save to the collection (keep old weight for now)
            weight = point.get(OutPar.weight)
            mpi.check_errors()
            if difflogmax is None and i > OutputOptions.reweight_after and \
                    time.time() - last_dump_time > OutputOptions.output_inteveral_s / 2:
                set_difflogmax()
                collection_out.out_update()

            if difflogmax is not None:
                logpost_new = sum(logpriors_new) + sum(loglikes_new)
                importance_weight = np.exp(logpost_new + point.get(OutPar.minuslogpost)
                                           - difflogmax)
                weight = weight * importance_weight
                importance_weights.append(importance_weight)
                if time.time() - last_dump_time > OutputOptions.output_inteveral_s:
                    collection_out.out_update()
                    last_dump_time = time.time()

            if weight > 0:
                collection_out.add(sampled, derived=derived.values(), weight=weight,
                                   logpriors=logpriors_new, loglikes=loglikes_new)

            # Display progress
            percent = int(np.round((i + done) / to_do * 100))
            if percent != last_percent and not percent % 5:
                last_percent = percent
                progress_bar(log, percent, " (%d/%d)" % (i + done, to_do))

        if difflogmax is None:
            set_difflogmax()
        if not collection_out.data.last_valid_index():
            raise LoggedError(
                log, "No elements in the final sample. Possible causes: "
                     "added a prior or likelihood valued zero over the full sampled "
                     "domain, or the computation of the theory failed everywhere, etc.")
        collection_out.out_update()
        weights.append(np.array(importance_weights))
        done += len(collection_in)

    assert difflogmax is not None
    points = 0
    tot_weight = 0
    min_weight = np.inf
    max_weight = -np.inf
    max_output_weight = -np.inf
    sum_w2 = 0
    points_removed = 0
    for collection_in, collection_out, importance_weights in zip(in_collections,
                                                                 out_collections,
                                                                 weights):
        output_weights = collection_out[OutPar.weight]
        points += len(collection_out)
        tot_weight += np.sum(output_weights)
        points_removed += len(importance_weights) - len(output_weights)
        min_weight = min(min_weight, np.min(importance_weights))
        max_weight = max(max_weight, np.max(importance_weights))
        max_output_weight = max(max_output_weight, np.max(output_weights))
        sum_w2 += np.dot(output_weights, output_weights)

    (tot_weights, min_weights, max_weights, max_output_weights, sum_w2s, points_s,
     points_removed_s) = mpi.zip_gather(
        [tot_weight, min_weight, max_weight, max_output_weight, sum_w2,
         points, points_removed])

    if mpi.is_main_process():
        output_out.clear_lock()
        log.info("Finished! Final number of distinct sample points: %s", sum(points_s))
        log.info("Importance weight range: %.4g -- %.4g",
                 min(min_weights), max(max_weights))
        if sum(points_removed_s):
            log.info("Points deleted due to zero weight: %s", sum(points_removed_s))
        log.info("Effective number of single samples if independent (sum w)/max(w): %s",
                 int(sum(tot_weights) / max(max_output_weights)))
        log.info(
            "Effective number of weighted samples if independent (sum w)^2/sum(w^2): "
            "%s", int(sum(tot_weights) ** 2 / sum(sum_w2s)))
    products: PostResultDict = {"sample": value_or_list(out_collections),
                                "stats": {'min_importance_weight': (min(min_weights) /
                                                                    max(max_weights)),
                                          'points_removed': sum(points_removed_s),
                                          'tot_weight': sum(tot_weights),
                                          'max_weight': max(max_output_weights),
                                          'sum_w2': sum(sum_w2s),
                                          'points': sum(points_s)},
                                "logpost_weight_offset": difflogmax,
                                "weights": value_or_list(weights)}
    return out_combined, products
