"""
.. module:: post

:Synopsis: Post-processing functions
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import, division

# Global
import logging

# Local
from cobaya.input import load_input
from cobaya.parameterization import is_sampled_param, is_derived_param
from cobaya.conventions import _prior_1d_name, _debug, _debug_file, _output_prefix, _post
from cobaya.conventions import _params, _prior, _likelihood, _theory
from cobaya.collection import Collection
from cobaya.log import logger_setup, HandledException
from cobaya.output import Output


# Dummy classes for loading chains for post processing

class DummyParameterization(object):

    def __init__(self, params_info):
        self._sampled_params = []
        self._derived_params = []
        for param, info in params_info.items():
            if is_sampled_param(info):
                self._sampled_params.append(param)
            elif is_derived_param(info):
                self._derived_params.append(param)

    def sampled_params(self):
        return self._sampled_params

    def derived_params(self):
        return self._derived_params


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
    # 3. Create necessary instances
