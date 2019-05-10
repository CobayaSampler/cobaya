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

# Local
from cobaya.input import load_input
from cobaya.parameterization import is_fixed_param, is_sampled_param, is_derived_param
from cobaya.conventions import _prior_1d_name, _debug, _debug_file, _output_prefix, _post
from cobaya.conventions import _params, _prior, _likelihood, _theory, _p_drop, _weight
from cobaya.conventions import _chi2, _separator, _minuslogpost, _force, _p_value
from cobaya.conventions import _minuslogprior
from cobaya.collection import Collection
from cobaya.log import logger_setup, HandledException
from cobaya.input import get_full_info
from cobaya.output import Output
from cobaya.prior import Prior
from cobaya.likelihood import LikelihoodCollection as Likelihood


# Dummy classes for loading chains for post processing

class DummyParameterization(object):

    def __init__(self, params_info):
        self._sampled_params = []
        self._derived_params = []
        self._input_params = []
        self._output_params = []
        self._constant_params = []
        for param, info in params_info.items():
            if is_fixed_param(info):
                self._input_params.append(param)
                if isinstance(info[_p_value], Number):
                    self._constant_params.append(p)
            if is_sampled_param(info):
                self._sampled_params.append(param)
                if not info.get(_p_drop):
                    self._input_params.append(param)
            elif is_derived_param(info):
                self._derived_params.append(param)

    def input_params(self):
        return self._input_params

    def output_params(self):
        return self._output_params

    def sampled_params(self):
        return self._sampled_params

    def derived_params(self):
        return self._derived_params

    def constant_params(self):
        return self._constant_params

    def sampled_params_info(self):
        return odict([[p, {_prior: {"dist": "uniform", "min": 0, "max": 1}}]
                      for p in self.sampled_params()])

    def sampled_input_dependence(self):
        return {}

class DummyModel(object):

    def __init__(self, info_params, info_likelihood, info_prior=None, info_theory=None):

        self.parameterization = DummyParameterization(info_params)
        self.prior = [_prior_1d_name] + list(info_prior or [])
        self.likelihood = list(info_likelihood)


def post(info):
    logger_setup(info.get(_debug), info.get(_debug_file))
    log = logging.getLogger(__name__.split(".")[-1])

    # TODO: 0. some input check here!

    # 1. Load existing sample
    # TODO: remove logging (talks about "resuming")
    output_in = Output(output_prefix=info.get(_output_prefix), resume=True)
    info_in = load_input(output_in.file_full)
    dummy_model_in = DummyModel(info_in[_params], info_in[_likelihood], info_in.get(_prior, None), info_in.get(_theory, None))
    # TODO: generalise NAME for multiple chains!!!!!!
    collection_in = Collection(dummy_model_in, output_in, name="1", resuming=True,
                               onload_skip=info[_post]["skip"], onload_thin=info[_post]["thin"])
    if collection_in.n() <= 1:
        log.error("Not enough samples for post-processing. Try using a larger sample, "
                  "or skipping or thinning less.")
        raise HandledException
    # 2. Compare old and new info: determine what to do
    out = {}
    for level in [_prior, _likelihood]:
        out[level] = getattr(dummy_model_in, level)
        for pdf in info[_post].get("remove", {}).get(level, []):
            try:
                out[level].remove(pdf)
            except ValueError:
                existing = out[level]
                if level == _prior:
                    existing.remove(_prior_1d_name)
                log.error("Trying to remove %s '%s', but it is not present. "
                          "Existing ones: %r", level, pdf, out[level])
                raise HandledException
    add = info[_post].get("add")
        # Add a dummy 'one' likelihood, to absorb unused parameters
    add.get(_likelihood, {}).update({"one": None})
    add = get_full_info(add)
    prior_add, likelihood_add = None, None
    if _prior in add:
        prior_add = Prior(dummy_model_in.parameterization, add[_prior])
        mlprior_names_add = [_minuslogprior + _separator + name for name in prior_add
                             if name is not _prior_1d_name]
        out[_prior] += [p for p in prior_add if p is not _prior_1d_name]
    if _likelihood in add:
        likelihood_add = Likelihood(
            add[_likelihood], dummy_model_in.parameterization,
        )
        chi2_names_add = [_chi2 + _separator + name for name in likelihood_add
                          if name is not "one"]
        out[_likelihood] += [l for l in likelihood_add if l is not "one"]

    # 3. Create output collection
    dummy_model_out = DummyModel(info_in[_params], out[_likelihood])#, info_in.get(_prior, None), info_in.get(_theory, None))
    output_out = Output(output_prefix=info.get(_output_prefix, "") +
                        "_" + _post + "_" + info[_post]["suffix"],
                        force_output=info.get(_force))
    ## TODO: generalise NAME for multiple chains!!!!!!
    info.update(info_in)
    info[_post].get("add", {}).get(_likelihood, {}).pop("one", None)
    output_out.dump_info({}, info)
    collection_out = Collection(dummy_model_out, output_out, name="1")

    # 4. Main loop!
    for i, point in collection_in:
        sampled = point[dummy_model_in.parameterization.sampled_params()]
        derived = point[dummy_model_in.parameterization.derived_params()]
        inputs = point[dummy_model_in.parameterization.input_params()]
        weight_old = point[_weight]
        logpriors_old = -point[collection_in.minuslogprior_names]
        loglikes_old = odict(-0.5*point[collection_in.chi2_names])
        # Add/remove priors
        logpriors_new = logpriors_old
        if prior_add:
            # Notice "0" (first prior in prior_add) is ignored: not in mlprior_names_add
            logpriors_add = odict(zip(mlprior_names_add, prior_add.logps(sampled)[1:]))
        else:
            logpriors_add = dict()
        logpriors_new = [logpriors_add.get(name, logpriors_old.get(name))
                         for name in collection_out.minuslogprior_names]
        if -np.inf in logpriors_new:
            continue
        # Add/remove likelihoods
        if likelihood_add:
            # Notice "one" (last in likelihood_add) is ignored: not in chi2_names
            loglikes_add = odict(zip(chi2_names_add, likelihood_add.logps(inputs)))
        else:
            loglikes_add = dict()
        loglikes_new = [loglikes_add.get(name, loglikes_old.get(name)) for name in collection_out.chi2_names]
        if -np.inf in loglikes_new:
            continue
        # Save to the collection
        collection_out.add(
            sampled, derived=derived,
            weight=weight_old, logpriors=logpriors_old, loglikes=loglikes_new)
        # Reweight
        collection_out[-1][_weight] *= np.exp(
            point[_minuslogpost] - collection_out[-1][_minuslogpost])
        # maybe I have to do everything in memory and only write at the end?
        # in that case, use output_update property of collection

        collection_out._out_update()
