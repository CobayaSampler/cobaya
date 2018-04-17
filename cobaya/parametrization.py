# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from six import string_types

# Global
from collections import OrderedDict as odict
from numbers import Number
from inspect import getargspec
from itertools import chain
from ast import parse

# Local
from cobaya.conventions import _prior, _p_drop, _p_derived, _p_label
from cobaya.tools import get_external_function, ensure_nolatex
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__.split(".")[-1])


def is_fixed_param(info_param):
    """
    Returns True if `info_param` is a number, a string or a function.
    """
    return (isinstance(info_param, Number) or
            isinstance(info_param, string_types) or
            callable(info_param))


def is_sampled_param(info_param):
    """
    Returns True if `info_param` has a `%s` field.
    """%_prior
    return _prior in (info_param if hasattr(info_param, "get") else {})


def is_derived_param(info_param):
    """
    Returns False if `info_param` is "fixed" or "sampled".
    """
    return not(is_fixed_param(info_param) or is_sampled_param(info_param))


class Parametrization(object):
    """
    Class managing parametrization.
    Translates parameter between sampler+prior and likelihood
    """
    def __init__(self, info_params):
        # First, we load the parameters,
        # not caring about whether they are understood by any likelihood.
        # `input` contains the parameters (expected to be) understood by the likelihood,
        #   with its fixed value, its fixing function, or None if their value is given
        #   directly by the sampler.
        self._input = odict()
        self._input_funcs = dict()
        self._input_args = dict()
        self._output = odict()
        self._fixed = odict()
        self._sampled = odict()
        self._derived = odict()
        self._derived_funcs = dict()
        self._derived_args = dict()
        for p, info in info_params.items():
            if is_fixed_param(info):
                self._input[p] = float(info) if isinstance(info, Number) else None
                if self._input[p] is None:
                    self._input_funcs[p] = get_external_function(info)
                    self._input_args[p] = getargspec(self._input_funcs[p]).args
                else:
                    self._fixed[p] = self._input[p]
            if is_sampled_param(info):
                self._sampled[p] = info
                if not info.get(_p_drop, False):
                    self._input[p] = None
            if is_derived_param(info):
                self._derived[p] = info
                if _p_derived in (info or {}):
                    self._derived_funcs[p] = get_external_function(info[_p_derived])
                    self._derived_args[p] = getargspec(self._derived_funcs[p]).args
                else:
                    self._output[p] = None
        # Check that the sampled and derived params are all valid python variable names
        def valid(name):
            try:
                parse("%s=None"%name)
                return True
            except SyntaxError:
                return False
        for p in chain(self.sampled_params(),self.derived_params()):
            if not valid(p):
                is_in = p in self.sampled_params()
                eg_in = "  p_prime:\n    prior: ...\n  %s: 'lambda p_prime: p_prime'\n"%p
                eg_out = "  p_prime: 'lambda %s: %s'\n"%(p,p)
                log.error("Parameter name '%s' is not a valid Python variable name "
                          "(it needs to start with a letter or '_').\n"
                          "If this is an %s parameter of a likelihood or theory, "
                          "whose name you cannot change,%s define an associated "
                          "%s one with a valid name 'p_prime' as: \n\n%s",
                          p, "input" if is_in else "output",
                          "" if is_in else " remove it and",
                          "sampled" if is_in else "derived",
                          eg_in if is_in else eg_out)
                raise HandledException
        # Assume that the *un*known function arguments are likelihood output parameters
        args = (set(chain(*self._input_args.values()))
                .union(chain(*self._derived_args.values())))
        for p in list(self._input.keys()) + list(self._sampled.keys()) + list(self._output.keys()):
            if p in args:
                args.remove(p)
        self._output.update({p:None for p in args})
        # Useful sets: directly-sampled input parameters and directly "output-ed" derived
        self._directly_sampled = [p for p in self._input if p in self._sampled]
        self._directly_output = [p for p in self._derived if p in self._output]
        # Useful mapping: input params that vary if each sampled is varied
        self._sampled_input_dependence = odict(
            [[s,[i for i in self._input if s in self._input_args.get(i, {})]]
             for s in self._sampled])
        # From here on, some error control.
        dropped_but_never_used = (
            set([p for p,v in self._sampled_input_dependence.items() if not v])
            .difference(set(self._directly_sampled)))
        if dropped_but_never_used:
            log.error("Parameters %r are sampled but not passed to the likelihood or "
                      "theory code, neither ever used as arguments for any parameters. "
                      "Check that you are not using the '%s' tag unintentionally.",
                      list(dropped_but_never_used), _p_drop)
            raise HandledException
        # input params depend on input and sampled only, never on output/derived
        bad_input_dependencies = set(chain(*self._input_args.values())).difference(
            set(self.input_params()).union(set(self.sampled_params())))
        if bad_input_dependencies:
            log.error("Input parameters defined as functions can only depend on other "
                      "input parameters that are not defined as functions. "
                      "In particular, an input parameter cannot depend on %r",
                      list(bad_input_dependencies))
            raise HandledException
        # derived depend of input and output, never of sampled which are not input
        bad_derived_dependencies = set(chain(*self._derived_args.values())).difference(
            set(self.input_params()).union(set(self.output_params())))
        if bad_derived_dependencies:
            log.error("Derived parameters can only depend on input and output parameters,"
                      " never on sampled parameters that have been defined as a function."
                      " In particular, a derived parameter cannot depend on %r",
                      list(bad_derived_dependencies))
            raise HandledException

    def input_params(self):
        return self._input

    def output_params(self):
        return self._output

    def fixed_params(self):
        return self._fixed

    def sampled_params(self):
        return self._sampled

    def derived_params(self):
        return self._derived

    def sampled_input_dependence(self):
        return self._sampled_input_dependence

    def to_input(self, sampled_params_values):
        as_odict = odict(zip(self.sampled_params(), sampled_params_values))
        # Fill first directly sampled input parameters
        self.input_params().update({p:as_odict[p] for p in self._directly_sampled})
        # Then evaluate the functions
        for p in self._input_funcs:
            args = {p:self.input_params().get(p, as_odict.get(p, None))
                    for p in self._input_args[p]}
            self.input_params()[p] = self._input_funcs[p](**args)
        return self.input_params()

    def to_derived(self, output_params_values):
        as_odict = odict(zip(self.output_params(), output_params_values))
        # Fill first derived parameters which are direct output parameters
        self.derived_params().update({p:as_odict[p] for p in self._directly_output})
        # Then evaluate the functions
        for p in self._derived_funcs:
            args = {p:self.input_params().get(p, as_odict.get(p, None))
                    for p in self._derived_args[p]}
            self.derived_params()[p] = self._derived_funcs[p](**args)
        return self.derived_params().values()

    def labels(self):
        """
        Returns a dictionary of LaTeX labels of the sampled and derived parameters.

        Uses the parameter name of no label has been given.
        """
        get_label = lambda p,info: (
            ensure_nolatex(getattr(info, "get", lambda x,y: y)(_p_label,p)))
        return odict([[p,get_label(p,info)] for p, info in
                      list(self.sampled_params().items()) +
                      list(self.derived_params().items())])

    # Python magic for the "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        return
