"""
.. module:: parameterization

:Synopsis: Class managing the possibly different parameterizations
           used by sampler and likelihoods
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from six import string_types

# Global
from collections import OrderedDict as odict
from numbers import Number
from itertools import chain
from copy import deepcopy

# Local
from cobaya.conventions import _prior, _p_drop, _p_derived, _p_label, _p_value, _p_renames
from cobaya.tools import get_external_function, ensure_nolatex, is_valid_variable_name, getargspec
from cobaya.log import HandledException

# Logger
import logging

log = logging.getLogger(__name__.split(".")[-1])


def is_fixed_param(info_param):
    """
    Returns True if the parameter has been fixed to a value or through a function.
    """
    return expand_info_param(info_param).get(_p_value, None) is not None


def is_sampled_param(info_param):
    """
    Returns True if the parameter has a prior.
    """
    return _prior in expand_info_param(info_param)


def is_derived_param(info_param):
    """
    Returns True if the parameter is saved as a derived one.
    """
    return expand_info_param(info_param).get(_p_derived, False)


def expand_info_param(info_param):
    """
    Expands the info of a parameter, from the user friendly, shorter format
    to a more unambiguous one.
    """
    info_param = deepcopy(info_param)
    if not hasattr(info_param, "keys"):
        if info_param is None:
            info_param = odict()
        else:
            info_param = odict([[_p_value, info_param]])
    if all([(f not in info_param) for f in [_prior, _p_value, _p_derived]]):
        info_param[_p_derived] = True
    # Dynamical input parameters: save as derived by default
    value = info_param.get(_p_value, None)
    if isinstance(value, string_types) or callable(value):
        info_param[_p_derived] = info_param.get(_p_derived, True)
    return info_param


def reduce_info_param(info_param):
    """
    Compresses the info of a parameter, suppressing default values.
    This is the opposite of :func:`~input.expand_info_param`.
    """
    info_param = deepcopy(info_param)
    if not hasattr(info_param, "keys"):
        return
    # All parameters without a prior are derived parameters unless otherwise specified
    if info_param.get(_p_derived) is True:
        info_param.pop(_p_derived)
    # Fixed parameters with single "value" key
    if list(info_param) == [_p_value]:
        return info_param[_p_value]
    return info_param


class Parameterization(object):
    """
    Class managing parameterization.
    Translates parameter between sampler+prior and likelihood
    """

    def __init__(self, info_params, allow_renames=True):
        self.allow_renames = allow_renames
        # First, we load the parameters,
        # not caring about whether they are understood by any likelihood.
        # `input` contains the parameters (expected to be) understood by the likelihood,
        #   with its fixed value, its fixing function, or None if their value is given
        #   directly by the sampler.
        self._input = odict()
        self._input_funcs = dict()
        self._input_args = dict()
        self._output = odict()
        self._constant = odict()
        self._sampled = odict()
        self._sampled_info = odict()
        self._sampled_renames = odict()
        self._derived = odict()
        self._derived_funcs = dict()
        self._derived_args = dict()
        # Notice here that expand_info_param *always* adds a _p_derived:True tag
        # to infos without _prior or _p_value, and a _p_value field to fixed params
        for p, info in info_params.items():
            if is_fixed_param(info):
                if isinstance(info[_p_value], Number):
                    self._constant[p] = info[_p_value]
                    if not info.get(_p_drop, False):
                        self._input[p] = self._constant[p]
                else:
                    self._input[p] = None
                    self._input_funcs[p] = get_external_function(info[_p_value])
                    self._input_args[p] = getargspec(self._input_funcs[p]).args
            if is_sampled_param(info):
                self._sampled[p] = None
                self._sampled_info[p] = deepcopy(info)
                if not info.get(_p_drop, False):
                    self._input[p] = None
                self._sampled_renames[p] = (
                    (lambda x: [x] if isinstance(x, string_types) else x)
                    (info.get(_p_renames, [])))
            if is_derived_param(info):
                self._derived[p] = deepcopy(info)
                # Dynamical parameters whose value we want to save
                if info[_p_derived] is True and is_fixed_param(info):
                    info[_p_derived] = "lambda %s: %s" % (p, p)
                if info[_p_derived] is True:
                    self._output[p] = None
                else:
                    self._derived_funcs[p] = get_external_function(info[_p_derived])
                    self._derived_args[p] = getargspec(self._derived_funcs[p]).args
        # Check that the sampled and derived params are all valid python variable names
        for p in chain(self.sampled_params(), self.derived_params()):
            if not is_valid_variable_name(p):
                is_in = p in self.sampled_params()
                eg_in = "  p_prime:\n    prior: ...\n  %s: 'lambda p_prime: p_prime'\n" % p
                eg_out = "  p_prime: 'lambda %s: %s'\n" % (p, p)
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
        for p in (list(self._constant) + list(self._input) +
                  list(self._sampled) + list(self._derived)):
            if p in args:
                args.remove(p)
        self._output.update({p: None for p in args})
        # Useful sets: directly-sampled input parameters and directly "output-ed" derived
        self._directly_sampled = [p for p in self._input if p in self._sampled]
        self._directly_output = [p for p in self._derived if p in self._output]
        # Useful mapping: input params that vary if each sampled is varied
        self._sampled_input_dependence = odict(
            [[s, [i for i in self._input if s in self._input_args.get(i, {})]]
             for s in self._sampled])
        # From here on, some error control.
        dropped_but_never_used = (
            set([p for p, v in self._sampled_input_dependence.items() if not v])
                .difference(set(self._directly_sampled)))
        if dropped_but_never_used:
            log.error("Parameters %r are sampled but not passed to the likelihood or "
                      "theory code, neither ever used as arguments for any parameters. "
                      "Check that you are not using the '%s' tag unintentionally.",
                      list(dropped_but_never_used), _p_drop)
            raise HandledException
        # input params depend on input and sampled only, never on output/derived
        all_input_arguments = set(chain(*self._input_args.values()))
        bad_input_dependencies = all_input_arguments.difference(
            set(self.input_params()).union(set(self.sampled_params())).union(set(self.constant_params())))
        if bad_input_dependencies:
            log.error("Input parameters defined as functions can only depend on other "
                      "input parameters that are not defined as functions. "
                      "In particular, an input parameter cannot depend on %r",
                      list(bad_input_dependencies))
            raise HandledException

    def input_params(self):
        return deepcopy(self._input)

    def output_params(self):
        return deepcopy(self._output)

    def constant_params(self):
        return deepcopy(self._constant)

    def sampled_params(self):
        return deepcopy(self._sampled)

    def sampled_params_info(self):
        return deepcopy(self._sampled_info)

    def sampled_params_renames(self):
        return deepcopy(self._sampled_renames)

    def derived_params(self):
        return deepcopy(self._derived)

    def sampled_input_dependence(self):
        return deepcopy(self._sampled_input_dependence)

    def _to_input(self, sampled_params_values):
        # Store sampled params, so that derived can depend on them
        if not hasattr(sampled_params_values, "keys"):
            sampled_params_values = odict(
                zip(self.sampled_params(), sampled_params_values))
        elif not isinstance(sampled_params_values, odict):
            sampled_params_values = odict(
                [(p, sampled_params_values[p]) for p in self.sampled_params()])
        self._sampled = deepcopy(sampled_params_values)
        # Fill first directly sampled input parameters
        self._input.update(
            {p: sampled_params_values[p] for p in self._directly_sampled})
        # Then evaluate the functions
        resolved_old = None
        resolved = []
        while resolved != resolved_old:
            resolved_old = deepcopy(resolved)
            for p in self._input_funcs:
                if p in resolved:
                    continue
                args = {p:
                    self._constant.get(
                        p, self._input.get(
                            p, sampled_params_values.get(p, None)))
                    for p in self._input_args[p]}
                if not all([isinstance(v, Number) for v in args.values()]):
                    continue
                try:
                    self._input[p] = self._input_funcs[p](**args)
                except NameError as exception:
                    unknown = str(exception).split("'")[1]
                    log.error(
                        "Unknown variable '%s' was referenced in the definition of "
                        "the parameter '%s', with arguments %r.",
                        unknown, p, list(args))
                    raise HandledException
                resolved.append(p)
        if set(resolved) != set(self._input_funcs):
            log.error("Could not resolve arguments for input parameters %s. Maybe there "
                      "is a circular dependency between derived parameters?",
                      list(set(self._input_funcs).difference(set(resolved))))
            raise HandledException
        return self.input_params()

    def _to_derived(self, output_params_values):
        if not hasattr(output_params_values, "keys"):
            output_params_values = dict(
                zip(self.output_params(), output_params_values))
        # Fill first derived parameters which are direct output parameters
        self._derived.update(
            {p: output_params_values[p] for p in self._directly_output})
        # Then evaluate the functions
        resolved_old = None
        resolved = []
        while resolved != resolved_old:
            resolved_old = deepcopy(resolved)
            for p in self._derived_funcs:
                if p in resolved:
                    continue
                args = {p: (self.input_params().get(
                    p, self.sampled_params().get(p, output_params_values.get(
                        p, self._derived.get(p, None)))))
                    for p in self._derived_args[p]}
                if not all([isinstance(v, Number) for v in args.values()]):
                    continue
                self._derived[p] = self._derived_funcs[p](**args)
                resolved.append(p)
        if set(resolved) != set(self._derived_funcs):
            log.error("Could not resolve arguments for derived parameters %s. Maybe there"
                      " is a circular dependency between derived parameters?",
                      list(set(self._derived_funcs).difference(set(resolved))))
            raise HandledException
        return list(self._derived.values())

    def _check_sampled(self, **sampled_params):
        """
        Check that the input dictionary contains all the sampled parameters,
        and just them. Is aware of known renamings.

        Returns `OrderedDict` of parameters (model's naming) and their values.
        """
        sampled_output = odict()
        sampled_input = deepcopy(sampled_params)
        for p, renames in self._sampled_renames.items():
            for pprime in sampled_input:
                if pprime == p or (pprime in renames if self.allow_renames else False):
                    sampled_output[p] = sampled_input.pop(pprime)
                    break
        if len(sampled_output) < len(self._sampled):
            not_found = set(self.sampled_params()).difference(set(sampled_output))
            if self.allow_renames:
                log.error("The following expected sampled parameters " +
                          ("(or their aliases) " if self.allow_renames else "") +
                          "where not found : %r",
                          ({p: self._sampled_renames[p] for p in not_found}
                           if self.allow_renames else not_found))
            else:
                log.error("The following expected sampled parameters "
                          "where not found : %r",
                          {p: self._sampled_renames[p] for p in not_found})
            raise HandledException
        if sampled_input:
            not_used = set(sampled_input)
            duplicated = not_used.intersection(set(
                chain(*[list(chain(*[[k], v])) for k, v in self._sampled_renames.items()])))
            not_used = not_used.difference(duplicated)
            derived = not_used.intersection(set(self.derived_params()))
            input_ = not_used.intersection(set(self.input_params()))
            unknown = not_used.difference(derived).difference(input_)
            log.error(
                "Incorrect parameters! " +
                ("\n   Duplicated entries (using their aliases): %r" % list(duplicated)
                 if duplicated else "") +
                ("\n   Not known: %r" % list(unknown) if unknown else "") +
                ("\n   Cannot be fixed: %r " % list(input_) +
                 "--> instead, fix sampled parameters that depend on them!"
                 if input_ else "") +
                ("\n   Cannot be fixed because are derived parameters: %r " % list(derived)
                 if derived else ""))
            raise HandledException
        return sampled_output

    def labels(self):
        """
        Returns a dictionary of LaTeX labels of the sampled and derived parameters.

        Uses the parameter name of no label has been given.
        """
        get_label = lambda p, info: (
            ensure_nolatex(getattr(info, "get", lambda x, y: y)(_p_label, p)))
        return odict([[p, get_label(p, info)] for p, info in
                      list(self.sampled_params().items()) +
                      list(self.derived_params().items())])

    # Python magic for the "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        return
