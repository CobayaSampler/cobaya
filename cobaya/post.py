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
from cobaya.conventions import _chi2, _separator, _minuslogpost, _force, _p_value
from cobaya.conventions import _minuslogprior, _path_install
from cobaya.collection import Collection
from cobaya.log import logger_setup, HandledException
from cobaya.input import get_full_info
from cobaya.output import Output
from cobaya.prior import Prior
from cobaya.likelihood import LikelihoodCollection as Likelihood
from cobaya.mpi import get_mpi_rank
from cobaya.tools import progress_bar


# Dummy classes for loading chains for post processing

class DummyModel(object):

    def __init__(self, info_params, info_likelihood, info_prior=None, info_theory=None):

        self.parameterization = Parameterization(info_params, ignore_unused_sampled=True)
        self.prior = [_prior_1d_name] + list(info_prior or [])
        self.likelihood = list(info_likelihood)


def post(info):
    logger_setup(info.get(_debug), info.get(_debug_file))
    log = logging.getLogger(__name__.split(".")[-1])
    if get_mpi_rank():
        log.warning(
            "Post-processing is not yet MPI-able. Doing nothing for rank > 1 processes.")
        return
    # 1. Load existing sample
    output_in = Output(output_prefix=info.get(_output_prefix), resume=True)
    info_in = load_input(output_in.file_full)
    dummy_model_in = DummyModel(info_in[_params], info_in[_likelihood],
                                info_in.get(_prior, None), info_in.get(_theory, None))
    i = 0
    while True:
        try:
            collection = Collection(
                dummy_model_in, output_in, name="%d" % (1 + i),
                load=True, onload_skip=info[_post].get("skip", 0),
                onload_thin=info[_post].get("thin", 1))
            if i == 0:
                collection_in = collection
            else:
                collection_in._append(collection)
            i += 1
        except IOError:
            break
    log.info("Loaded %d chain%s containing %d samples.",
             i, "s" if i - 1 else "", collection_in.n())
    if collection_in.n() <= 1:
        log.error("Not enough samples for post-processing. Try using a larger sample, "
                  "or skipping or thinning less.")
        raise HandledException
    # 2. Compare old and new info: determine what to do
    add = info[_post].get("add", {})
    remove = info[_post].get("remove", {})
    # Add a dummy 'one' likelihood, to absorb unused parameters
    if not add.get(_likelihood):
        add[_likelihood] = odict()
    add[_likelihood].update({"one": None})
    # Expand the "add" info
    add = get_full_info(add)
    # 2.1 Adding/removing derived parameters and changes in priors of sampled parameters
    out = {_params: deepcopy(info_in[_params])}
    for p in remove.get(_params, {}):
        pinfo = info_in[_params].get(p)
        if pinfo is None or not is_derived_param(pinfo):
            log.error(
                "You tried to remove parameter '%s', which is not a derived paramter. "
                "Only derived parameters can be removed during post-processing.", p)
            raise HandledException
        out[_params].pop(p)
    mlprior_names_add = []
    for p, pinfo in add.get(_params, {}).items():
        if is_sampled_param(pinfo):
            pinfo_in = info_in[_params].get(p)
            if not is_sampled_param(pinfo_in):
                log.error(
                    "You tried to change the prior of parameter '%s', "
                    "but it was not a sampled parameter. "
                    "To change that prior, you need to define as an external one.", p)
                raise HandledException
            if mlprior_names_add[:1] != _prior_1d_name:
                mlprior_names_add = (
                    [_minuslogprior + _separator + _prior_1d_name] + mlprior_names_add)
        elif not is_derived_param(pinfo):
            log.error(
                "You tried to add parameter '%s', which is not a derived paramter. "
                "Only derived parameters can be added during post-processing.", p)
            raise HandledException
        out[_params][p] = pinfo
    # For the likelihood only, turn the rest of derived parameters into constants,
    # so that the likelihoods do not try to compute them)
    out_params_like = deepcopy(out[_params])
    for p, pinfo in out_params_like.items():
        if is_derived_param(pinfo) and p not in add.get(_params, {}):
            out_params_like[p] = {_p_value: np.nan, _p_drop: True}
    parameterization_like = Parameterization(out_params_like, ignore_unused_sampled=True)
    # 2.2 Manage adding/removing priors and likelihoods
    warn_remove = False
    for level in [_prior, _likelihood]:
        out[level] = getattr(dummy_model_in, level)
        if level == _prior:
            out[level].remove(_prior_1d_name)
        for pdf in info[_post].get("remove", {}).get(level, []) or []:
            try:
                out[level].remove(pdf)
                warn_remove = True
            except ValueError:
                log.error("Trying to remove %s '%s', but it is not present. "
                          "Existing ones: %r", level, pdf, out[level])
                raise HandledException
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
    if _likelihood in add:
        # Don't initialise the theory code if not adding/recomputing theory,
        # theory-derived params or likelihoods
        recompute_theory = not (
            list(add[_likelihood]) == ["one"] and
            not any([is_derived_param(pinfo) for pinfo in add.get(_params, {}).values()]))
        info_theory_out = (
            add.get(_theory, info_in.get(_theory, None)) if recompute_theory else None)
        chi2_names_add = [_chi2 + _separator + name for name in add[_likelihood]
                          if name is not "one"]
        out[_likelihood] += [l for l in add[_likelihood] if l is not "one"]
    # 3. Create output collection
    if "suffix" not in info[_post]:
        log.error("You need to provide a 'suffix' for your chains.")
        raise HandledException
    output_out = Output(output_prefix=info.get(_output_prefix, "") +
                        "_" + _post + "_" + info[_post]["suffix"],
                        force_output=info.get(_force))
    info_out = deepcopy(info)
    # Updated with input info and extended (full) add info
    info_out.update(info_in)
    info_out[_post]["add"] = deepcopy(add)
    info_out[_post].get("add", {}).get(_likelihood, {}).pop("one", None)
    output_out.dump_info({}, info_out)
    dummy_model_out = DummyModel(
        out[_params], out[_likelihood], info_prior=out[_prior])
    prior_add = Prior(dummy_model_out.parameterization, add.get(_prior))
    likelihood_add = Likelihood(
        add[_likelihood], parameterization_like,
        info_theory=info_theory_out, modules=info.get(_path_install))
    if likelihood_add.theory:
        likelihood_add.theory.input_params = info[_post]["theory_params"]["input"]
    collection_out = Collection(dummy_model_out, output_out, name="1")
    # 4. Main loop!
    log.info("Running post-processing...")
    last_percent = 0
    for i, point in enumerate(collection_in.data.itertuples()):
        log.debug("Point: %r", point)
        sampled = [getattr(point, param) for param in
                   dummy_model_in.parameterization.sampled_params()]
        derived = odict(
            [[param, getattr(point, param, None)]
             for param in dummy_model_out.parameterization.derived_params()])
        inputs = odict([
            [param, getattr(
                point, param,
                dummy_model_in.parameterization.constant_params().get(param, None))]
            for param in dummy_model_in.parameterization.input_params()])
        # Solve inputs that depend on a function and were not saved
        # (we don't use the Parameterization_to_input method in case there are references
        #  to functions that cannot be loaded at the moment)
        for p, value in inputs.items():
            if value is None:
                func = dummy_model_out.parameterization._input_funcs[p]
                args = dummy_model_out.parameterization._input_args[p]
                inputs[p] = func(*[getattr(point, arg) for arg in args])
        # Add/remove priors
        priors_add = prior_add.logps(sampled)
        if not prior_recompute_1d:
            priors_add = priors_add[1:]
        logpriors_add = odict(zip(mlprior_names_add, priors_add))
        logpriors_new = [logpriors_add.get(name, - getattr(point, name, 0))
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
        loglikes_new = [loglikes_add.get(name, -0.5 * getattr(point, name, 0))
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
                derived[p] = func(*[getattr(point, arg) for arg in args])
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug("New derived parameters: %r",
                      dict([[p, derived[p]]
                            for p in dummy_model_out.parameterization.derived_params()
                            if p in add[_params]]))
        # Save to the collection (keep old weight for now)
        collection_out.add(
            sampled, derived=derived.values(), weight=getattr(point, _weight),
            logpriors=logpriors_new, loglikes=loglikes_new)
        # Display progress
        percent = np.round(i / collection_in.n() * 100)
        if percent != last_percent and not percent % 5:
            last_percent = percent
            progress_bar(log, percent, " (%d/%d)" % (i, collection_in.n()))
    if not collection_out.data.last_valid_index():
        log.error("No elements in the final sample. Possible causes: "
                  "added a prior or likelihood valued zero over the full sampled domain, "
                  "or the computation of the theory failed everywhere, etc.")
        raise HandledException
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
    return deepcopy(info_out), {"post": collection_out}
