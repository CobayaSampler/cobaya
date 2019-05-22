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
from cobaya.parameterization import Parameterization
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

        self.parameterization = Parameterization(info_params)
        self.prior = [_prior_1d_name] + list(info_prior or [])
        self.likelihood = list(info_likelihood)


def post(info):
    logger_setup(info.get(_debug), info.get(_debug_file))
    log = logging.getLogger(__name__.split(".")[-1])
    if get_mpi_rank():
        log.warning(
            "Post-processing is not MPI-able. Doing nothing for rank > 1 processes.")
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
    out = {}
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
    add = info[_post].get("add", {})
    remove = info[_post].get("remove", {})
    # Add a dummy 'one' likelihood, to absorb unused parameters
    if not add.get(_likelihood):
        add[_likelihood] = odict()
    add[_likelihood].update({"one": None})
    add = get_full_info(add)
    prior_add, likelihood_add = None, None
    if _prior in add:
        prior_add = Prior(dummy_model_in.parameterization, add[_prior])
        mlprior_names_add = [_minuslogprior + _separator + name for name in prior_add
                             if name is not _prior_1d_name]
        out[_prior] += [p for p in prior_add if p is not _prior_1d_name]
    if _likelihood in add:
        # Don't initialise the theory code if not adding/recomputing theory or likelihoods
        info_theory_out = (
            add.get(_theory, (info_in.get(_theory, None)
                              if list(add[_likelihood]) != ["one"] else None)))
        likelihood_add = Likelihood(
            add[_likelihood], dummy_model_in.parameterization,
            info_theory=info_theory_out, modules=info.get(_path_install)
        )
        # TEMPORARY (BETA ONLY): specify theory parameters
        if likelihood_add.theory:
            likelihood_add.theory.input_params = info[_post]["theory_params"]["input"]
        chi2_names_add = [_chi2 + _separator + name for name in likelihood_add
                          if name is not "one"]
        out[_likelihood] += [l for l in likelihood_add if l is not "one"]
    out[_params] = deepcopy(info_in[_params])
    for p in remove.get(_params, {}):
        pinfo = info_in[_params].get(p)
        if pinfo is None or not is_derived_param(pinfo):
            log.error(
                "You tried to remove parameter '%s', which is not a derived paramter. "
                "Only derived parameters can be removed during post-processing.", p)
            raise HandledException
        out[_params].pop(p)
    for p, pinfo in add.get(_params, {}).items():
        if not is_derived_param(pinfo):
            log.error(
                "You tried to add parameter '%s', which is not a derived paramter. "
                "Only derived parameters can be added during post-processing.", p)
            raise HandledException
        out[_params][p] = pinfo
    # 3. Create output collection
    if "suffix" not in info[_post]:
        log.error("You need to provide a 'suffix' for your chains.")
        raise HandledException
    output_out = Output(output_prefix=info.get(_output_prefix, "") +
                        "_" + _post + "_" + info[_post]["suffix"],
                        force_output=info.get(_force))
    info_out = deepcopy(info)
    info_out.update(info_in)
    info_out[_post].get("add", {}).get(_likelihood, {}).pop("one", None)
    output_out.dump_info({}, info_out)
    dummy_model_out = DummyModel(
        out[_params], out[_likelihood], info_prior=out[_prior]
    )
    collection_out = Collection(dummy_model_out, output_out, name="1")
    # 4. Main loop!
    log.info("Running post-processing...")
    last_percent = 0
    for i, point in enumerate(collection_in.data.itertuples()):
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
        # Add/remove priors
        if prior_add:
            # Notice "0" (first prior in prior_add) is ignored: not in mlprior_names_add
            logpriors_add = odict(zip(mlprior_names_add, prior_add.logps(sampled)[1:]))
        else:
            logpriors_add = dict()
        logpriors_new = [logpriors_add.get(name, - getattr(point, name, 0))
                         for name in collection_out.minuslogprior_names]
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(
                "New set of priors: %r", dict(zip(dummy_model_out.prior, logpriors_new)))
        if -np.inf in logpriors_new:
            continue
        # Add/remove likelihoods
        if likelihood_add:
            # Notice "one" (last in likelihood_add) is ignored: not in chi2_names
            loglikes_add = odict(zip(chi2_names_add, likelihood_add.logps(inputs)))
        else:
            loglikes_add = dict()
        loglikes_new = [loglikes_add.get(name, -0.5 * getattr(point, name, 0))
                        for name in collection_out.chi2_names]
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(
                "New set of likelihoods: %r",
                dict(zip(dummy_model_out.likelihood, loglikes_new)))
        if -np.inf in loglikes_new:
            continue
        # Add/remove derived parameters
        for p in add[_params]:
            func = dummy_model_out.parameterization._derived_funcs[p]
            args = dummy_model_out.parameterization._derived_args[p]
            derived[p] = func(*[getattr(point, arg) for arg in args])
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
                  "added a prior or likelihood valued zero over the full sampled domain; "
                  "the computation of the theory failed everywhere...")
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
    log.info("Finished!")
    return deepcopy(info_out), {"post": collection_out}
