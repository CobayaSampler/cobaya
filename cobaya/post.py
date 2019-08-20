"""
.. module:: post

:Synopsis: Post-processing functions
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import, division

# Global
import numpy as np
import logging
from collections import OrderedDict as odict
from numbers import Number
from copy import deepcopy

# Local
from cobaya.input import load_input
from cobaya.parameterization import Parameterization, expand_info_param
from cobaya.parameterization import is_fixed_param, is_sampled_param, is_derived_param
from cobaya.conventions import _prior_1d_name, _debug, _debug_file, _output_prefix, _post
from cobaya.conventions import _params, _prior, _likelihood, _theory, _p_drop, _weight
from cobaya.conventions import _chi2, _separator, _minuslogpost, _force, _p_value, _p_drop
from cobaya.conventions import _minuslogprior, _path_install, _input_params
from cobaya.conventions import _separator_files, _post_add, _post_remove, _post_suffix
from cobaya.collection import Collection
from cobaya.log import logger_setup, LoggedError
from cobaya.input import update_info
from cobaya.output import get_Output as Output
from cobaya.prior import Prior
from cobaya.likelihood import LikelihoodCollection as Likelihood
from cobaya.mpi import get_mpi_rank
from cobaya.tools import progress_bar, recursive_update


# Dummy classes for loading chains for post processing

class DummyModel(object):

    def __init__(self, info_params, info_likelihood, info_prior=None, info_theory=None):
        self.parameterization = Parameterization(info_params, ignore_unused_sampled=True)
        self.prior = [_prior_1d_name] + list(info_prior or [])
        self.likelihood = list(info_likelihood)


def post(info, sample=None):
    logger_setup(info.get(_debug), info.get(_debug_file))
    log = logging.getLogger(__name__.split(".")[-1])
    try:
        info_post = info[_post]
    except KeyError:
        raise LoggedError(log, "No 'post' block given. Nothing to do!")
    if get_mpi_rank():
        log.warning(
            "Post-processing is not yet MPI-able. Doing nothing for rank > 1 processes.")
        return
    # 1. Load existing sample
    output_in = Output(output_prefix=info.get(_output_prefix), resume=True)
    info_in = load_input(output_in.file_updated) if output_in else deepcopy(info)
    dummy_model_in = DummyModel(info_in[_params], info_in[_likelihood],
                                info_in.get(_prior, None), info_in.get(_theory, None))
    if output_in:
        collection_in = output_in.load_collections(
            dummy_model_in, skip=info_post.get("skip", 0), thin=info_post.get("thin", 1),
            concatenate=True)
    elif sample:
        if isinstance(sample, Collection):
            sample = [sample]
        collection_in = deepcopy(sample[0])
        for s in sample[1:]:
            try:
                collection_in._append(s)
            except:
                raise LoggedError(log, "Failed to load some of the input samples.")
    else:
        raise LoggedError(log, "Not output from where to load from or input collections given.")
    log.info("Will process %d samples.", collection_in.n())
    if collection_in.n() <= 1:
        raise LoggedError(
            log, "Not enough samples for post-processing. Try using a larger sample, "
            "or skipping or thinning less.")
    # 2. Compare old and new info: determine what to do
    add = info_post.get(_post_add, {}) or {}
    remove = info_post.get(_post_remove, {})
    # Add a dummy 'one' likelihood, to absorb unused parameters
    if not add.get(_likelihood):
        add[_likelihood] = odict()
    add[_likelihood].update({"one": None})
    # Expand the "add" info
    add = update_info(add)
    # 2.1 Adding/removing derived parameters and changes in priors of sampled parameters
    out = {_params: deepcopy(info_in[_params])}
    for p in remove.get(_params, {}):
        pinfo = info_in[_params].get(p)
        if pinfo is None or not is_derived_param(pinfo):
            raise LoggedError(
                log,
                "You tried to remove parameter '%s', which is not a derived paramter. "
                "Only derived parameters can be removed during post-processing.", p)
        out[_params].pop(p)
    mlprior_names_add = []
    for p, pinfo in add.get(_params, {}).items():
        pinfo_in = info_in[_params].get(p)
        if is_sampled_param(pinfo):
            if not is_sampled_param(pinfo_in):
                # No added sampled parameters (de-marginalisation not implemented)
                if pinfo_in is None:
                    raise LoggedError(
                        log, "You added a new sampled parameter %r (maybe accidentaly "
                        "by adding a new likelihood that depends on it). "
                        "Adding new sampled parameters is not possible. Try fixing "
                        "it to some value.", p)
                else:
                    raise LoggedError(
                        log,
                        "You tried to change the prior of parameter '%s', "
                        "but it was not a sampled parameter. "
                        "To change that prior, you need to define as an external one.", p)
            if mlprior_names_add[:1] != _prior_1d_name:
                mlprior_names_add = (
                        [_minuslogprior + _separator + _prior_1d_name] + mlprior_names_add)
        elif is_derived_param(pinfo):
            if p in out[_params]:
                raise LoggedError(
                    log, "You tried to add derived parameter '%s', which is already "
                    "present. To force its recomputation, 'remove' it too.", p)
        elif is_fixed_param(pinfo):
            # Only one possibility left "fixed" parameter that was not present before:
            # input of new likelihood, or just an argument for dynamical derived (dropped)
            if ((p in info_in[_params] and
                 pinfo[_p_value] != (pinfo_in or {}).get(_p_value, None))):
                raise LoggedError(
                    log,
                    "You tried to add a fixed parameter '%s: %r' that was already present"
                    " but had a different value or was not fixed. This is not allowed. "
                    "The old info of the parameter was '%s: %r'",
                    p, dict(pinfo), p, dict(pinfo_in))
        else:
            raise LoggedError(log, "This should not happen. Contact the developers.")
        out[_params][p] = pinfo
    # For the likelihood only, turn the rest of *derived* parameters into constants,
    # so that the likelihoods do not try to compute them)
    # But be careful to exclude *input* params that have a "derived: True" value
    # (which in "updated info" turns into "derived: 'lambda [x]: [x]'")
    out_params_like = deepcopy(out[_params])
    for p, pinfo in out_params_like.items():
        if ((is_derived_param(pinfo) and not (_p_value in pinfo)
             and p not in add.get(_params, {}))):
            out_params_like[p] = {_p_value: np.nan, _p_drop: True}
    parameterization_like = Parameterization(out_params_like, ignore_unused_sampled=True)
    # 2.2 Manage adding/removing priors and likelihoods
    warn_remove = False
    for level in [_prior, _likelihood]:
        out[level] = getattr(dummy_model_in, level)
        if level == _prior:
            out[level].remove(_prior_1d_name)
        for pdf in info_post.get(_post_remove, {}).get(level, []) or []:
            try:
                out[level].remove(pdf)
                warn_remove = True
            except ValueError:
                raise LoggedError(
                    log, "Trying to remove %s '%s', but it is not present. "
                    "Existing ones: %r", level, pdf, out[level])
    if warn_remove:
        log.warning("You are removing a prior or likelihood pdf. "
                    "Notice that if the resulting posterior is much wider "
                    "than the original one, or displaced enough, "
                    "it is probably safer to explore it directly.")
    if _prior in add:
        mlprior_names_add += [_minuslogprior + _separator + name for name in add[_prior]]
        out[_prior] += list(add[_prior])
    prior_recompute_1d = (
            mlprior_names_add[:1] == [_minuslogprior + _separator + _prior_1d_name])
    # Don't initialise the theory code if not adding/recomputing theory,
    # theory-derived params or likelihoods
    recompute_theory = info_in.get(_theory) and not (
            list(add[_likelihood]) == ["one"] and
            not any([is_derived_param(pinfo) for pinfo in add.get(_params, {}).values()]))
    if recompute_theory:
        # Inherit from the original chain (needs input|output_params, renames, etc
        theory = list(info_in[_theory].keys())[0]
        info_theory_out = odict([
            [theory, recursive_update(deepcopy(info_in[_theory][theory]),
                                      add.get(_theory, {theory: {}})[theory])]])
    else:
        info_theory_out = None
    chi2_names_add = [_chi2 + _separator + name for name in add[_likelihood]
                      if name is not "one"]
    out[_likelihood] += [l for l in add[_likelihood] if l is not "one"]
    if recompute_theory:
        log.warning("You are recomputing the theory, but in the current version this does"
                    " not force recomputation of any likelihood or derived parameter, "
                    "unless explicitly removed+added.")
    for level in [_prior, _likelihood]:
        for i, x_i in enumerate(out[level]):
            if x_i in list(out[level])[i + 1:]:
                raise LoggedError(
                    log, "You have added %s '%s', which was already present. If you "
                    "want to force its recomputation, you must also 'remove' it.",
                    level, x_i)
    # 3. Create output collection
    if _post_suffix not in info_post:
        raise LoggedError(
            log, "You need to provide a '%s' for your chains.", _post_suffix)
    # Use default prefix if it exists. If it does not, produce no output by default.
    # {post: {output: None}} suppresses output, and if it's a string, updates it.
    out_prefix = info_post.get(_output_prefix, info.get(_output_prefix))
    if out_prefix not in [None, False]:
        out_prefix += _separator_files + _post + _separator_files + info_post[_post_suffix]
    output_out = Output(output_prefix=out_prefix, force_output=info.get(_force))
    info_out = deepcopy(info)
    info_out[_post] = info_post
    # Updated with input info and extended (updated) add info
    info_out.update(info_in)
    info_out[_post][_post_add] = add
    dummy_model_out = DummyModel(
        out[_params], out[_likelihood], info_prior=out[_prior])
    if recompute_theory:
        theory = list(info_theory_out.keys())[0]
        if _input_params not in info_theory_out[theory]:
            raise LoggedError(
                log,
                "You appear to be post-processing a chain generated with an older "
                "version of Cobaya. For post-processing to work, please edit the "
                "'[root].updated.yaml' file of the original chain to add, inside the "
                "theory code block, the list of its input parameters. E.g.\n----\n"
                "theory:\n  %s:\n    input_params: [param1, param2, ...]\n"
                "----\nIf you get strange errors later, it is likely that you did not "
                "specify the correct set of theory parameters.\n"
                "The full set of input parameters are %s.",
                theory, list(dummy_model_out.parameterization.input_params()))
    prior_add = Prior(dummy_model_out.parameterization, add.get(_prior))
    likelihood_add = Likelihood(
        add[_likelihood], parameterization_like,
        info_theory=info_theory_out, modules=info.get(_path_install))
    # Remove auxiliary "one" before dumping -- 'add' *is* info_out[_post][_post_add]
    add[_likelihood].pop("one")
    if likelihood_add.theory:
        # Make sure that theory.needs is called at least once, for adjustments
        likelihood_add.theory.needs()
    collection_out = Collection(dummy_model_out, output_out, name="1")
    output_out.dump_info({}, info_out)
    # 4. Main loop!
    log.info("Running post-processing...")
    last_percent = 0
    for i, point in collection_in.data.iterrows():
        log.debug("Point: %r", point)
        sampled = [point[param] for param in
                   dummy_model_in.parameterization.sampled_params()]
        derived = odict(
            [[param, point.get(param, None)]
             for param in dummy_model_out.parameterization.derived_params()])
        inputs = odict([
            [param, point.get(
                param, dummy_model_in.parameterization.constant_params().get(
                    param, dummy_model_out.parameterization.constant_params().get(
                        param, None)))]
            for param in dummy_model_out.parameterization.input_params()])
        # Solve inputs that depend on a function and were not saved
        # (we don't use the Parameterization_to_input method in case there are references
        #  to functions that cannot be loaded at the moment)
        for p, value in inputs.items():
            if value is None:
                func = dummy_model_out.parameterization._input_funcs[p]
                args = dummy_model_out.parameterization._input_args[p]
                inputs[p] = func(*[point.get(arg) for arg in args])
        # Add/remove priors
        priors_add = prior_add.logps(sampled)
        if not prior_recompute_1d:
            priors_add = priors_add[1:]
        logpriors_add = odict(zip(mlprior_names_add, priors_add))
        logpriors_new = [logpriors_add.get(name, - point.get(name, 0))
                         for name in collection_out.minuslogprior_names]
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(
                "New set of priors: %r", dict(zip(dummy_model_out.prior, logpriors_new)))
        if -np.inf in logpriors_new:
            continue
        # Add/remove likelihoods
        output_like = []
        if likelihood_add:
            # Notice "one" (last in likelihood_add) is ignored: not in chi2_names
            loglikes_add = odict(
                zip(chi2_names_add, likelihood_add.logps(inputs, _derived=output_like)))
            output_like = dict(zip(likelihood_add.output_params, output_like))
        else:
            loglikes_add = dict()
        loglikes_new = [loglikes_add.get(name, -0.5 * point.get(name, 0))
                        for name in collection_out.chi2_names]
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(
                "New set of likelihoods: %r",
                dict(zip(dummy_model_out.likelihood, loglikes_new)))
            if output_like:
                log.debug("New set of likelihood-derived parameters: %r", output_like)
        if -np.inf in loglikes_new:
            continue
        # Add/remove derived parameters and change priors of sampled parameters
        for p in add[_params]:
            if p in dummy_model_out.parameterization._directly_output:
                derived[p] = output_like[p]
            elif p in dummy_model_out.parameterization._derived_funcs:
                func = dummy_model_out.parameterization._derived_funcs[p]
                args = dummy_model_out.parameterization._derived_args[p]
                derived[p] = func(
                    *[point.get(arg, output_like.get(arg, None)) for arg in args])
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug("New derived parameters: %r",
                      dict([[p, derived[p]]
                            for p in dummy_model_out.parameterization.derived_params()
                            if p in add[_params]]))
        # Save to the collection (keep old weight for now)
        collection_out.add(
            sampled, derived=derived.values(), weight=point.get(_weight),
            logpriors=logpriors_new, loglikes=loglikes_new)
        # Display progress
        percent = np.round(i / collection_in.n() * 100)
        if percent != last_percent and not percent % 5:
            last_percent = percent
            progress_bar(log, percent, " (%d/%d)" % (i, collection_in.n()))
    if not collection_out.data.last_valid_index():
        raise LoggedError(
            log, "No elements in the final sample. Possible causes: "
            "added a prior or likelihood valued zero over the full sampled domain, "
            "or the computation of the theory failed everywhere, etc.")
    # Reweight -- account for large dynamic range!
    #   Prefer to rescale +inf to finite, and ignore final points with -inf.
    #   Remove -inf's (0-weight), and correct indices
    difflogmax = max(collection_in[_minuslogpost] - collection_out[_minuslogpost])
    collection_out.data[_weight] *= np.exp(
        collection_in[_minuslogpost] - collection_out[_minuslogpost] - difflogmax)
    collection_out.data = (
        collection_out.data[collection_out.data.weight > 0].reset_index(drop=True))
    collection_out._n = collection_out.data.last_valid_index() + 1
    # Write!
    collection_out._out_update()
    log.info("Finished! Final number of samples: %d", collection_out.n())
    return info_out, {"sample": collection_out}
