"""
.. module:: post

:Synopsis: Post-processing functions
:Author: Jesus Torrado and Antony Lewis

"""

import os
import time
from itertools import chain
from typing import TYPE_CHECKING, TypedDict, Union

import numpy as np

from cobaya import mpi
from cobaya.collection import SampleCollection
from cobaya.conventions import (
    OutPar,
    get_chi2_name,
    get_minuslogpior_name,
    minuslogprior_names,
    prior_1d_name,
    separator_files,
    undo_chi2_name,
)
from cobaya.input import add_aggregated_chi2_params, load_input_dict, update_info
from cobaya.log import LoggedError, get_logger, is_debug, logger_setup
from cobaya.model import DummyModel, Model
from cobaya.output import get_output
from cobaya.parameterization import (
    Parameterization,
    is_derived_param,
    is_fixed_or_function_param,
    is_sampled_param,
)
from cobaya.prior import Prior
from cobaya.tools import (
    deepcopy_where_possible,
    progress_bar,
    recursive_update,
    str_to_list,
)
from cobaya.typing import (
    ExpandedParamsDict,
    InputDict,
    ModelBlock,
    ParamValuesDict,
    PostDict,
)

if TYPE_CHECKING:
    from getdist import MCSamples


class PostResultDict(TypedDict):
    sample: Union[SampleCollection, list[SampleCollection], "MCSamples"]
    stats: ParamValuesDict
    logpost_weight_offset: float
    weights: np.ndarray | list[np.ndarray]


class PostResult:
    def __init__(self, post_results: PostResultDict):
        self.results = post_results

    # For backwards compatibility
    def __getitem__(self, key):
        return self.results[key]

    # For compatibility with Sampler, when returned by run()
    def samples(
        self,
        combined: bool = False,
        skip_samples: float = 0,
        to_getdist: bool = False,
    ) -> Union[SampleCollection, list[SampleCollection], "MCSamples"]:
        """
        Returns the post-processed sample.

        Parameters
        ----------
        combined: bool, default: False
            If ``True`` and running more than one MPI process, returns for all processes
            a single sample collection including, all parallel chains concatenated,
            instead of the chain of the current process only. For this to work, this
            method needs to be called from all MPI processes simultaneously.
        skip_samples: int or float, default: 0
            Skips some amount of initial samples (if ``int``), or an initial fraction of
            them (if ``float < 1``). If concatenating (``combined=True``), skipping is
            applied before concatenation. Forces the return of a copy.
        to_getdist: bool, default: False
            If ``True``, returns a single :class:`getdist.MCSamples` instance, containing
            all samples (``combined`` is ignored).

        Returns
        -------
        SampleCollection, list[SampleCollection], getdist.MCSamples
            The post-processed samples.
        """
        # Difference with MCMC: self.results["sample"] may contain one collection or a
        # list of them pre-process
        collections = self.results["sample"]
        if not isinstance(collections, list):
            collections = [collections]
        collections = [c.skip_samples(skip_samples, inplace=False) for c in collections]
        if not (to_getdist or combined):
            return collections
        # In all the remaining cases, we'll concatenate the chains
        collection = None
        all_collections = mpi.gather(collections)
        if mpi.is_main_process():
            all_collections = list(chain(*all_collections))
            if to_getdist:
                collection = all_collections[0].to_getdist(
                    combine_with=all_collections[1:]
                )
            else:
                if len(all_collections) > 1:
                    for collection in all_collections[1:]:
                        all_collections[0]._append(collection)
                collection = all_collections[0]
        return mpi.share_mpi(collection)  # type: ignore

    # For compatibility with Sampler, when returned by run()
    def products(
        self,
        combined: bool = False,
        skip_samples: float = 0,
        to_getdist: bool = False,
    ) -> PostResultDict:
        """
        Returns the products of post-processing.

        Parameters
        ----------
        combined: bool, default: False
            If ``True`` and running more than one MPI process, the ``sample`` key of the
            returned dictionary contains a concatenated sample including all parallel
            chains concatenated, instead of the chain of the current process only. For
            this to work, this method needs to be called from all MPI processes
            simultaneously.
        skip_samples: int or float, default: 0
            Skips some amount of initial samples (if ``int``), or an initial fraction of
            them (if ``float < 1``). If concatenating (``combined=True``), skipping is
            applied previously to concatenation. Forces the return of a copy.
        to_getdist: bool, default: False
            If ``True``, returns a single :class:`getdist.MCSamples` instance, containing
            all samples (``combined`` is ignored).

        Returns
        -------
        PostResultDict
            A dictionary containing the :class:`cobaya.collection.SampleCollection` of
            accepted steps under ``"sample"``, and stats about the post-processing.
        """
        products_dict: PostResultDict = {
            "sample": self.samples(
                combined=combined, skip_samples=skip_samples, to_getdist=to_getdist
            ),
            "stats": self.results["stats"],
            "logpost_weight_offset": self.results["logpost_weight_offset"],
            "weights": self.results["weights"],
        }
        return products_dict


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


def get_collections(info, output_in, info_post, sample, dummy_model_in, log):
    in_collections = []
    thin = info_post.get("thin", 1)
    skip = info_post.get("skip", 0)
    if info.get("thin") is not None or info.get("skip") is not None:  # type: ignore
        raise LoggedError(
            log, "'thin' and 'skip' should be parameters of the 'post' block"
        )

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
                collection = collection.skip_samples(skip)
            if thin != 1:
                collection = collection.thin_samples(thin or 0)
            in_collections[i] = collection
    elif output_in:
        files = output_in.find_collections()
        numbered = files
        if not numbered:
            # look for un-numbered output files
            files = output_in.find_collections(name=False)
        if files:
            if mpi.size() > len(files):
                raise LoggedError(
                    log,
                    "Number of MPI processes (%s) is larger than "
                    "the number of sample files (%s)",
                    mpi.size(),
                    len(files),
                )
            for num in range(mpi.rank(), len(files), mpi.size()):
                in_collections += [
                    SampleCollection(
                        dummy_model_in,
                        output_in,
                        onload_thin=thin,
                        onload_skip=skip,
                        load=True,
                        file_name=files[num],
                        name=str(num + 1) if numbered else "",
                    )
                ]
        else:
            raise LoggedError(
                log,
                "No samples found for the input model with prefix %s",
                os.path.join(output_in.folder, output_in.prefix),
            )

    else:
        raise LoggedError(
            log, "No output from where to load from, nor input collections given."
        )
    # Let's make sure we work on a copy if the chain is going to be altered
    already_copied = bool(output_in) or (sample is not None and (skip or thin != 1))
    for i, collection in enumerate(in_collections):
        if not already_copied:
            collection = collection.copy()
        in_collections[i] = collection
    # A note on tempered chains: detempering happens automatically when reweighting,
    # which is done later in this function in most cases.
    # But for the sake of robustness, we detemper all chains at init.
    # In order not to introduce reweighting errors coming from subtractions of the max
    # log-posterior at detempering, we need to detemper all samples at once
    all_in_collections = mpi.gather(in_collections)
    if mpi.is_main_process():
        flat_in_collections = list(chain(*all_in_collections))
        if any(c.is_tempered for c in flat_in_collections):
            log.info("Starting from tempered chains. Will detemper before proceeding.")
        flat_in_collections[0].reset_temperature(with_batch=flat_in_collections[1:])
    # Detempering happens in place, so one can scatter back the original
    # all_in_collections object to preserve the in_collection dist across processes
    in_collections = mpi.scatter(all_in_collections)
    if any(len(c) <= 1 for c in in_collections):
        raise LoggedError(
            log,
            "Not enough samples for post-processing. Try using a larger sample, "
            "or skipping or thinning less.",
        )
    mpi.sync_processes()
    log.info("Will process %d sample points.", sum(len(c) for c in in_collections))
    return in_collections


@mpi.sync_state
def post(
    info_or_yaml_or_file: InputDict | str | os.PathLike,
    sample: SampleCollection | list[SampleCollection] | None = None,
) -> tuple[InputDict, PostResult]:
    info = load_input_dict(info_or_yaml_or_file)
    logger_setup(info.get("debug"))
    log = get_logger(__name__)
    info_post: PostDict | None = info.get("post")
    if not info_post:
        raise LoggedError(log, "No 'post' block given. Nothing to do!")
    if mpi.is_main_process() and info.get("resume"):
        log.warning("Resuming not implemented for post-processing. Re-starting.")
    if not info.get("output") and info_post.get("output") and not info.get("params"):
        raise LoggedError(
            log,
            "The input dictionary must be a full option "
            "dictionary, or have an existing 'output' root to load "
            "previous settings from ('output' to read from is in the "
            "main block not under 'post'). ",
        )
    # 1. Load existing sample
    if output_in := get_output(prefix=info.get("output")):
        info_in = output_in.get_updated_info() or update_info(info)
    else:
        info_in = update_info(info)
    params_in: ExpandedParamsDict = info_in["params"]  # type: ignore
    dummy_model_in = DummyModel(
        params_in, info_in.get("likelihood", {}), info_in.get("prior")
    )

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
                "Only derived parameters can be removed during post-processing.",
                p,
            )
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
                        log,
                        "You added a new sampled parameter %r (maybe accidentally "
                        "by adding a new likelihood that depends on it). "
                        "Adding new sampled parameters is not possible. Try fixing "
                        "it to some value.",
                        p,
                    )
                else:
                    raise LoggedError(
                        log,
                        "You tried to change the prior of parameter '%s', "
                        "but it was not a sampled parameter. "
                        "To change that prior, you need to define as an external one.",
                        p,
                    )
            # recompute prior if potentially changed sampled parameter priors
            prior_recompute_1d = True
        elif is_derived_param(pinfo):
            if p in out_combined_params:
                raise LoggedError(
                    log,
                    "You tried to add derived parameter '%s', which is already "
                    "present. To force its recomputation, 'remove' it too.",
                    p,
                )
        elif is_fixed_or_function_param(pinfo):
            # Only one possibility left "fixed" parameter that was not present before:
            # input of new likelihood, or just an argument for dynamical derived (dropped)
            if pinfo_in and p in params_in and pinfo["value"] != pinfo_in.get("value"):
                raise LoggedError(
                    log,
                    "You tried to add a fixed parameter '%s: %r' that was already present"
                    " but had a different value or was not fixed. This is not allowed. "
                    "The old info of the parameter was '%s: %r'",
                    p,
                    dict(pinfo),
                    p,
                    dict(pinfo_in),
                )
        elif not pinfo_in:  # OK as long as we have known value for it
            raise LoggedError(log, "Parameter %s no known value. ", p)
        out_combined_params[p] = pinfo

    out_combined: InputDict = {"params": out_combined_params}  # type: ignore
    # Turn the rest of *derived* parameters into constants,
    # so that the likelihoods do not try to recompute them
    # But be careful to exclude *input* params that have a "derived: True" value
    # (which in "updated info" turns into "derived: 'lambda [x]: [x]'")
    # Don't assign derived parameters to theories, only likelihoods, so they can be
    # recomputed if needed. If the theory does not need to be computed, it doesn't matter
    # if it is already assigned parameters in the usual way; likelihoods can get
    # the required derived parameters from the stored sample derived parameter inputs.
    out_params_with_computed = deepcopy_where_possible(out_combined_params)

    dropped_theory = set()
    for p, pinfo in out_params_with_computed.items():
        if is_derived_param(pinfo) and "value" not in pinfo and p not in add_params:
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
            except ValueError as excpt:
                raise LoggedError(
                    log,
                    "Trying to remove %s '%s', but it is not present. Existing ones: %r",
                    kind,
                    remove_item,
                    list(out_combined[kind]),
                ) from excpt
        if kind != "theory" and kind in add:
            dups = set(add.get(kind) or []).intersection(out_combined[kind]) - {"one"}
            if dups:
                raise LoggedError(
                    log,
                    "You have added %s '%s', which was already present. If you "
                    "want to force its recomputation, you must also 'remove' it.",
                    kind,
                    dups,
                )
            out_combined[kind].update(add[kind])

    if warn_remove and mpi.is_main_process():
        log.warning(
            "You are removing a prior or likelihood pdf. "
            "Notice that if the resulting posterior is much wider "
            "than the original one, or displaced enough, "
            "it is probably safer to explore it directly."
        )

    mlprior_names_add = minuslogprior_names(add.get("prior") or [])

    out_combined["likelihood"].pop("one", None)

    add_theory = add.get("theory")
    if add_theory:
        if len(add["likelihood"]) == 1 and not any(
            is_derived_param(pinfo) for pinfo in add_params.values()
        ):
            log.warning(
                "You are adding a theory, but this does not force recomputation "
                "of any likelihood or derived parameters unless explicitly "
                "removed+added."
            )
        # Inherit from the original chain (input|output_params, renames, etc)
        added_theory = add_theory.copy()
        for theory, theory_info in out_combined["theory"].items():
            if theory in list(added_theory):
                out_combined["theory"][theory] = recursive_update(
                    theory_info, added_theory.pop(theory)
                )
        out_combined["theory"].update(added_theory)

    # Use default prefix if it exists. If it does not, produce no output by default.
    # {post: {output: None}} suppresses output, and if it's a string, updates it.
    out_prefix = info_post.get("output", info.get("output"))
    if out_prefix:
        suffix = info_post.get("suffix")
        if not suffix:
            raise LoggedError(
                log, "You need to provide a '%s' for your output chains.", "suffix"
            )
        out_prefix += separator_files + "post" + separator_files + suffix

    if "minimize" in (info.get("sampler") or []):
        # actually minimizing with importance-sampled combination of likelihoods
        out_combined: InputDict = dict(info, **out_combined)  # type: ignore
        out_combined.pop("post")
        out_combined["output"] = out_prefix
        from cobaya.run import run

        return run(out_combined)  # type: ignore

    in_collections = get_collections(
        info, output_in, info_post, sample, dummy_model_in, log
    )

    # Prepare recomputation of aggregated chi2
    # (they need to be recomputed by hand, because auto-computation won't pick up
    #  old likelihoods for a given type)
    all_types = {
        like: str_to_list(opts.get("type") or [])
        for like, opts in out_combined["likelihood"].items()
    }
    types = set(chain(*all_types.values()))
    inv_types = {
        t: [like for like, like_types in all_types.items() if t in like_types]
        for t in sorted(types)
    }
    add_aggregated_chi2_params(out_combined_params, types)

    # 3. Create output collection
    output_out = get_output(prefix=out_prefix, force=info.get("force"))
    output_out.set_lock()

    if output_out and not output_out.force and output_out.find_collections():
        raise LoggedError(
            log,
            "Found existing post-processing output with prefix %r. "
            "Delete it manually or re-run with `force: True` "
            "(or `-f`, `--force` from the shell).",
            out_prefix,
        )
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

    dummy_model_out = DummyModel(
        out_combined_params, out_combined["likelihood"], info_prior=out_combined["prior"]
    )
    out_func_parameterization = Parameterization(out_params_with_computed)

    # TODO: check allow_renames=False?
    model_add = Model(
        out_params_with_computed,
        add["likelihood"],
        info_prior=add.get("prior"),
        info_theory=out_combined["theory"],
        packages_path=(info_post.get("packages_path") or info.get("packages_path")),
        allow_renames=False,
        post=True,
        stop_at_error=info.get("stop_at_error", False),
        skip_unused_theories=True,
        dropped_theory_params=dropped_theory,
    )
    # Remove auxiliary "one" before dumping -- 'add' *is* info_out["post"]["add"]
    add["likelihood"].pop("one")
    out_collections = [
        SampleCollection(
            dummy_model_out,
            output_out,
            name=c.name,
            cache_size=OutputOptions.default_post_cache_size,
        )
        for c in in_collections
    ]
    # TODO: should maybe add skip/thin to out_combined, so can tell post-processed?
    output_out.check_and_dump_info(info_out, out_combined, check_compatible=False)
    collection_in = in_collections[0]
    collection_out = out_collections[0]

    last_percent = None
    known_constants = dummy_model_out.parameterization.constant_params()
    known_constants.update(dummy_model_in.parameterization.constant_params())

    if missing_params := dummy_model_in.parameterization.sampled_params().keys() - set(
        collection_in.columns
    ):
        raise LoggedError(
            log,
            "Input samples do not contain expected sampled parameter values: %s",
            missing_params,
        )

    missing_priors = {
        name
        for name in collection_out.minuslogprior_names
        if name not in mlprior_names_add and name not in collection_in.columns
    }
    if _minuslogprior_1d_name in missing_priors:
        prior_recompute_1d = True
    if prior_recompute_1d:
        missing_priors.discard(_minuslogprior_1d_name)
        mlprior_names_add.insert(0, _minuslogprior_1d_name)
    prior_regenerate: Prior | None
    if missing_priors and "prior" in info_in:
        # in case there are input priors that are not stored in input samples
        # e.g. when postprocessing GetDist/CosmoMC-format chains
        in_names = minuslogprior_names(info_in["prior"])
        info_prior = {
            piname: inf
            for (piname, inf), in_name in zip(info_in["prior"].items(), in_names)
            if in_name in missing_priors
        }
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
    difflogmax: float | None = None
    to_do = sum(len(c) for c in in_collections)
    weights = []
    done = 0
    last_dump_time = time.time()
    for collection_in, collection_out in zip(in_collections, out_collections):
        importance_weights = []

        def set_difflogmax():
            nonlocal difflogmax
            difflog = collection_in[OutPar.minuslogpost].to_numpy(dtype=np.float64)[
                : len(collection_out)
            ] - collection_out[OutPar.minuslogpost].to_numpy(dtype=np.float64)
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

        sampled_params = dummy_model_in.parameterization.sampled_params()

        for i, point in collection_in.data.iterrows():
            all_params = point.to_dict()
            for p in remove_params:
                all_params.pop(p, None)
            log.debug("Point: %r", point)
            sampled = np.array([all_params[param] for param in sampled_params])
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
            logpriors_new = [
                logpriors_add.get(name, -point.get(name, 0))
                for name in collection_out.minuslogprior_names
            ]
            if prior_regenerate:
                regenerated = dict(
                    zip(
                        regenerated_prior_names,
                        prior_regenerate.logps_external(all_params),
                    )
                )
                for _i, name in enumerate(collection_out.minuslogprior_names):
                    if name in regenerated_prior_names:
                        logpriors_new[_i] = regenerated[name]

            if is_debug(log):
                log.debug(
                    "New set of priors: %r",
                    dict(zip(dummy_model_out.prior, logpriors_new)),
                )
            if -np.inf in logpriors_new:
                continue
            # Add/remove likelihoods and/or (re-)calculate derived parameters
            loglikes_add, output_derived = model_add._loglikes_input_params(
                all_params, return_output_params=True, as_dict=True
            )
            loglikes_add = {
                get_chi2_name(name): loglikes_add[name]
                for name in model_add.likelihood
                if name != "one"
            }
            output_derived = {_p: output_derived[_p] for _p in model_add.output_params}

            loglikes_new = [
                loglikes_add.get(name, -0.5 * point.get(name, 0))
                for name in collection_out.chi2_names
            ]
            if is_debug(log):
                log.debug(
                    "New set of likelihoods: %r",
                    dict(zip(dummy_model_out.likelihood, loglikes_new)),
                )
                if output_derived:
                    log.debug("New set of derived parameters: %r", output_derived)
            if -np.inf in loglikes_new:
                continue
            all_params.update(output_derived)

            all_params.update(out_func_parameterization.to_derived(all_params))
            derived = {
                param: all_params.get(param)
                for param in dummy_model_out.parameterization.derived_params()
            }
            # We need to recompute the aggregated chi2 by hand
            for type_, likes in inv_types.items():
                derived[get_chi2_name(type_)] = sum(
                    -2 * lvalue
                    for lname, lvalue in zip(collection_out.chi2_names, loglikes_new)
                    if undo_chi2_name(lname) in likes
                )
            if is_debug(log):
                log.debug(
                    "New derived parameters: %r",
                    {
                        p: derived[p]
                        for p in dummy_model_out.parameterization.derived_params()
                        if p in add["params"]
                    },
                )
            # Save to the collection (keep old weight for now)
            weight = point.get(OutPar.weight)
            mpi.check_errors()
            if (
                difflogmax is None
                and i > OutputOptions.reweight_after
                and time.time() - last_dump_time > OutputOptions.output_inteveral_s / 2
            ):
                set_difflogmax()
                collection_out.out_update()

            if difflogmax is not None:
                logpost_new = sum(logpriors_new) + sum(loglikes_new)
                importance_weight = np.exp(
                    logpost_new + point.get(OutPar.minuslogpost) - difflogmax
                )
                weight = weight * importance_weight
                importance_weights.append(importance_weight)
                if time.time() - last_dump_time > OutputOptions.output_inteveral_s:
                    collection_out.out_update()
                    last_dump_time = time.time()

            if weight > 0:
                collection_out.add(
                    sampled,
                    derived=derived.values(),
                    weight=weight,
                    logpriors=logpriors_new,
                    loglikes=loglikes_new,
                )

            # Display progress
            percent = int(np.round((i + done) / to_do * 100))
            if percent != last_percent and not percent % 5:
                last_percent = percent
                progress_bar(log, percent, " (%d/%d)" % (i + done, to_do))

        if difflogmax is None:
            set_difflogmax()
        if not collection_out.data.last_valid_index():
            raise LoggedError(
                log,
                "No elements in the final sample. Possible causes: "
                "added a prior or likelihood valued zero over the full sampled "
                "domain, or the computation of the theory failed everywhere, etc.",
            )
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
    for collection_in, collection_out, importance_weights in zip(
        in_collections, out_collections, weights
    ):
        output_weights = collection_out[OutPar.weight]
        points += len(collection_out)
        tot_weight += np.sum(output_weights)
        points_removed += len(importance_weights) - len(output_weights)
        min_weight = min(min_weight, np.min(importance_weights))
        max_weight = max(max_weight, np.max(importance_weights))
        max_output_weight = max(max_output_weight, np.max(output_weights))
        sum_w2 += np.dot(output_weights, output_weights)

    (
        tot_weights,
        min_weights,
        max_weights,
        max_output_weights,
        sum_w2s,
        points_s,
        points_removed_s,
    ) = mpi.zip_gather(
        [
            tot_weight,
            min_weight,
            max_weight,
            max_output_weight,
            sum_w2,
            points,
            points_removed,
        ]
    )

    if mpi.is_main_process():
        output_out.clear_lock()
        log.info("Finished! Final number of distinct sample points: %s", sum(points_s))
        log.info(
            "Importance weight range: %.4g -- %.4g", min(min_weights), max(max_weights)
        )
        if sum(points_removed_s):
            log.info("Points deleted due to zero weight: %s", sum(points_removed_s))
        log.info(
            "Effective number of single samples if independent (sum w)/max(w): %s",
            int(sum(tot_weights) / max(max_output_weights)),
        )
        log.info(
            "Effective number of weighted samples if independent (sum w)^2/sum(w^2): %s",
            int(sum(tot_weights) ** 2 / sum(sum_w2s)),
        )
    products_dict: PostResultDict = {
        "sample": value_or_list(out_collections),
        "stats": {
            "min_importance_weight": (min(min_weights) / max(max_weights)),
            "points_removed": sum(points_removed_s),
            "tot_weight": sum(tot_weights),
            "max_weight": max(max_output_weights),
            "sum_w2": sum(sum_w2s),
            "points": sum(points_s),
        },
        "logpost_weight_offset": difflogmax,
        "weights": value_or_list(weights),
    }
    return out_combined, PostResult(products_dict)
