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

# Local
from cobaya.input import load_input
from cobaya.parameterization import is_fixed_param, is_sampled_param, is_derived_param
from cobaya.conventions import _prior_1d_name, _debug, _debug_file, _output_prefix, _post
from cobaya.conventions import _params, _prior, _likelihood, _theory, _p_drop, _weight
from cobaya.conventions import _chi2, _separator, _minuslogpost
from cobaya.collection import Collection
from cobaya.log import logger_setup, HandledException
from cobaya.input import get_full_info
from cobaya.output import Output
from cobaya.likelihood import LikelihoodCollection as Likelihood


# Dummy classes for loading chains for post processing

class DummyParameterization(object):

    def __init__(self, params_info):
        self._sampled_params = []
        self._derived_params = []
        self._input_params = []
        self._output_params = []
        for param, info in params_info.items():
            if is_fixed_param(info):
                self._input_params.append(param)
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
    ## TODO: checks should happen BEFORE loading the old chain maybe???
    add = info[_post].get("add")
    likelihoods_out = list(dummy_model_in.likelihood)
    if _likelihood in add:
        # Add a dummy 'one' likelihood, to absorb unused parameters
        add[_likelihood].update({"one": None})
    add_full = get_full_info(add)
    if _likelihood in add_full:
        likelihood_add = Likelihood(
            add_full[_likelihood], dummy_model_in.parameterization,
                                    #info_theory, # modules=modules
            # TODO: how do we get to know whether we have to initialise the theory code?
            # i.e. whether one of the likelihoods needs it.
        )
        chi2_add = [_chi2 + _separator + name for name in likelihood_add]
        likelihoods_out += list(likelihood_add)
        likelihoods_out.remove("one")
        ## TODO: manage output/derived parameters!!!
    else:
        likelihood_add = None

    # 3. Create output collection
    dummy_model_out = DummyModel(info_in[_params], likelihoods_out)#, info_in.get(_prior, None), info_in.get(_theory, None))
    output_out = Output(output_prefix=info.get(_output_prefix, "")+info[_post]["suffix"])
    ## TODO: generalise NAME for multiple chains!!!!!!
    collection_out = Collection(dummy_model_out, output_out, name="post")

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
        if -np.inf in logpriors_new:
            continue

        # Add/remove likelihoods
        if likelihood_add:
            loglikes_add = odict(zip(chi2_add, likelihood_add.logps(inputs)))
        else:
            loglikes_add = dict()
        loglikes_new = [loglikes_add.get(name, loglikes_old.get(name)) for name in collection_out.chi2_names]
        if -np.inf in loglikes_new:
            continue

        collection_out.add(
            sampled, derived=derived,
            weight=weight_old, logpriors=logpriors_old, loglikes=loglikes_new)
        # Reweight
        collection_out[-1][_weight] *= np.exp(
            point[_minuslogpost] - collection_out[-1][_minuslogpost])
        # maybe I have to do everything in memory and only write at the end?
        # in that case, use output_update property of collection

        collection_out._out_update()
