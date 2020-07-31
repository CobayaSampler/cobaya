"""
.. module:: parameterization

:Synopsis: Class managing the possibly different parameterizations
           used by sampler and likelihoods
:Author: Jesus Torrado

"""
# Global
import numpy as np
from numbers import Number
from itertools import chain
from copy import deepcopy

# Local
from cobaya.conventions import partag
from cobaya.tools import get_external_function, ensure_nolatex, is_valid_variable_name, \
    getfullargspec, deepcopy_where_possible, invert_dict
from cobaya.log import LoggedError, HasLogger


def is_fixed_param(info_param):
    """
    Returns True if the parameter has been fixed to a value or through a function.
    """
    return expand_info_param(info_param).get(partag.value, None) is not None


def is_sampled_param(info_param):
    """
    Returns True if the parameter has a prior.
    """
    return partag.prior in expand_info_param(info_param)


def is_derived_param(info_param):
    """
    Returns True if the parameter is saved as a derived one.
    """
    return expand_info_param(info_param).get(partag.derived, False)


def expand_info_param(info_param, default_derived=True):
    """
    Expands the info of a parameter, from the user friendly, shorter format
    to a more unambiguous one.
    """
    info_param = deepcopy_where_possible(info_param)
    if not isinstance(info_param, dict):
        if info_param is None:
            info_param = {}
        else:
            info_param = {partag.value: info_param}
    if all(f not in info_param for f in [partag.prior, partag.value, partag.derived]):
        info_param[partag.derived] = default_derived
    # Dynamical input parameters: save as derived by default
    value = info_param.get(partag.value, None)
    if isinstance(value, str) or callable(value):
        info_param[partag.derived] = info_param.get(partag.derived, True)
    return info_param


def reduce_info_param(info_param):
    """
    Compresses the info of a parameter, suppressing default values.
    This is the opposite of :func:`~input.expand_info_param`.
    """
    info_param = deepcopy_where_possible(info_param)
    if not isinstance(info_param, dict):
        return
    # All parameters without a prior are derived parameters unless otherwise specified
    if info_param.get(partag.derived) is True:
        info_param.pop(partag.derived)
    # Fixed parameters with single "value" key
    if list(info_param) == [partag.value]:
        return info_param[partag.value]
    return info_param


class Parameterization(HasLogger):
    """
    Class managing parameterization.
    Translates parameter between sampler+prior and likelihood
    """

    def __init__(self, info_params, allow_renames=True, ignore_unused_sampled=False):
        self.set_logger(lowercase=True)
        self.allow_renames = allow_renames
        # First, we load the parameters,
        # not caring about whether they are understood by any likelihood.
        # `input` contains the parameters (expected to be) understood by the likelihood,
        #   with its fixed value, its fixing function, or None if their value is given
        #   directly by the sampler.
        self._infos = {}
        self._input = {}
        self._input_funcs = {}
        self._input_args = {}
        self._output = {}
        self._constant = {}
        self._sampled = {}
        self._sampled_renames = {}
        self._derived = {}
        self._derived_funcs = {}
        self._derived_args = {}
        # Notice here that expand_info_param *always* adds a partag.derived:True tag
        # to infos without _prior or partag.value, and a partag.value field
        # to fixed params
        for p, info in info_params.items():
            self._infos[p] = deepcopy_where_possible(info)
            if is_fixed_param(info):
                if isinstance(info[partag.value], Number):
                    self._constant[p] = info[partag.value]
                    if not info.get(partag.drop, False):
                        self._input[p] = self._constant[p]
                else:
                    self._input[p] = None
                    self._input_funcs[p] = get_external_function(info[partag.value])
                    self._input_args[p] = getfullargspec(self._input_funcs[p]).args
            if is_sampled_param(info):
                self._sampled[p] = None
                if not info.get(partag.drop, False):
                    self._input[p] = None
                self._sampled_renames[p] = (
                    (lambda x: [x] if isinstance(x, str) else x)
                    (info.get(partag.renames, [])))
            if is_derived_param(info):
                self._derived[p] = deepcopy_where_possible(info)
                # Dynamical parameters whose value we want to save
                if info[partag.derived] is True and is_fixed_param(info):
                    info[partag.derived] = "lambda %s: %s" % (p, p)
                if info[partag.derived] is True:
                    self._output[p] = None
                else:
                    self._derived_funcs[p] = get_external_function(info[partag.derived])
                    self._derived_args[p] = getfullargspec(self._derived_funcs[p]).args
        # Check that the sampled and derived params are all valid python variable names
        for p in chain(self._sampled, self._derived):
            if not is_valid_variable_name(p):
                is_in = p in self._sampled
                eg_in = "  p_prime:\n    prior: ...\n  %s: 'lambda p_prime: p_prime'\n" % p
                eg_out = "  p_prime: 'lambda %s: %s'\n" % (p, p)
                raise LoggedError(
                    self.log, "Parameter name '%s' is not a valid Python variable name "
                              "(it needs to start with a letter or '_').\n"
                              "If this is an %s parameter of a likelihood or theory, "
                              "whose name you cannot change,%s define an associated "
                              "%s one with a valid name 'p_prime' as: \n\n%s",
                    p, "input" if is_in else "output",
                    "" if is_in else " remove it and",
                    "sampled" if is_in else "derived",
                    eg_in if is_in else eg_out)
        # Assume that the *un*known function arguments are likelihood/theory
        # output parameters
        for arg in (set(chain(*self._input_args.values()))
                            .union(chain(*self._derived_args.values()))
                    - set(self._constant) - set(self._input)
                    - set(self._sampled) - set(self._derived)):
            self._output[arg] = None

        # Useful sets: directly-sampled input parameters and directly "output-ed" derived
        self._directly_sampled = [p for p in self._input if p in self._sampled]
        self._directly_output = [p for p in self._derived if p in self._output]
        # Useful mapping: input params that vary if each sample is varied
        self._sampled_input_dependence = {s: [i for i in self._input
                                              if s in self._input_args.get(i, {})]
                                          for s in self._sampled}
        # From here on, some error control.
        dropped_but_never_used = (
            set(p for p, v in self._sampled_input_dependence.items() if not v)
                .difference(set(self._directly_sampled)))
        if dropped_but_never_used and not ignore_unused_sampled:
            raise LoggedError(
                self.log,
                "Parameters %r are sampled but not passed to a likelihood or theory "
                "code, and never used as arguments for any parameter functions. "
                "Check that you are not using the '%s' tag unintentionally.",
                list(dropped_but_never_used), partag.drop)
        # input params depend on input and sampled only, never on output/derived
        all_input_arguments = set(chain(*self._input_args.values()))
        bad_input_dependencies = all_input_arguments.difference(
            set(self.input_params()).union(set(self.sampled_params())).union(
                set(self.constant_params())))
        if bad_input_dependencies:
            raise LoggedError(
                self.log,
                "Input parameters defined as functions can only depend on other "
                "input parameters that are not defined as functions. "
                "In particular, an input parameter cannot depend on %r."
                "Use an explicit Theory calculator for more complex dependencies.",
                list(bad_input_dependencies))
        self._wrapped_input_funcs, self._wrapped_derived_funcs = \
            self._get_wrapped_functions_evaluation_order()
        # warn if repeated labels
        labels_inv_repeated = invert_dict(self.labels())
        for k in list(labels_inv_repeated):
            if len(labels_inv_repeated[k]) == 1:
                labels_inv_repeated.pop(k)
        if labels_inv_repeated:
            self.log.warn("There are repeated parameter labels: %r", labels_inv_repeated)

    def input_params(self):
        return self._input.copy()

    def output_params(self):
        return self._output.copy()

    def constant_params(self):
        return self._constant.copy()

    def sampled_params(self):
        return self._sampled.copy()

    def sampled_params_info(self):
        return {p: deepcopy_where_possible(info) for p, info
                in self._infos.items() if p in self._sampled}

    def sampled_params_renames(self):
        return deepcopy(self._sampled_renames)

    def derived_params(self):
        return self._derived.copy()

    def sampled_input_dependence(self):
        return deepcopy(self._sampled_input_dependence)

    def to_input(self, sampled_params_values, copied=True):
        # Store sampled params, so that derived can depend on them
        if not isinstance(sampled_params_values, dict):
            sampled_params_values = dict(zip(self._sampled, sampled_params_values))
        else:
            sampled_params_values = sampled_params_values.copy()

        self._sampled = sampled_params_values
        # Fill first directly sampled input parameters
        for p in self._directly_sampled:
            self._input[p] = sampled_params_values[p]
        if self._wrapped_input_funcs:
            # Then evaluate the functions
            for p, (func, args, to_set) in self._wrapped_input_funcs.items():
                for arg in to_set:
                    args[arg] = self._input.get(arg,
                                                sampled_params_values.get(arg, None))
                self._input[p] = self._call_param_func(p, func, args)
        return self.input_params() if copied else self._input

    def to_derived(self, output_params_values):
        if not isinstance(output_params_values, dict):
            output_params_values = dict(zip(self._output, output_params_values))
        # Fill first derived parameters which are direct output parameters
        for p in self._directly_output:
            self._derived[p] = output_params_values[p]
        # Then evaluate the functions
        if self._wrapped_derived_funcs:
            # Then evaluate the functions
            for p, (func, args, to_set) in self._wrapped_derived_funcs.items():
                for arg in to_set:
                    val = self._input.get(arg)
                    if val is None:
                        val = output_params_values.get(arg)
                        if val is None:
                            val = self._derived.get(arg)
                            if val is None:
                                val = self._sampled.get(arg)
                    args[arg] = val
                self._derived[p] = self._call_param_func(p, func, args)
        return list(self._derived.values())

    def check_sampled(self, **sampled_params):
        """
        Check that the input dictionary contains all the sampled parameters,
        and just them. Is aware of known renamings.

        Returns dict of parameters (model's naming) and their values.
        """
        sampled_output = {}
        sampled_input = sampled_params.copy()
        for p, renames in self._sampled_renames.items():
            for pprime in sampled_input:
                if pprime == p or (pprime in renames if self.allow_renames else False):
                    sampled_output[p] = sampled_input.pop(pprime)
                    break
        if len(sampled_output) < len(self._sampled):
            not_found = set(self.sampled_params()).difference(set(sampled_output))
            if self.allow_renames:
                msg = ("The following expected sampled parameters " +
                       ("(or their aliases) " if self.allow_renames else "") +
                       "where not found : %r",
                       ({p: self._sampled_renames[p] for p in not_found}
                        if self.allow_renames else not_found))
            else:
                msg = ("The following expected sampled parameters "
                       "where not found : %r",
                       {p: self._sampled_renames[p] for p in not_found})
            raise LoggedError(self.log, *msg)
        # Ignore fixed input parameters if they have the correct value
        to_pop = []
        for p, value in sampled_input.items():
            known_value = self.constant_params().get(p, None)
            if known_value is None:
                raise LoggedError(self.log, "Unknown parameter %r.", p)
            elif np.allclose(value, known_value):
                to_pop.append(p)
                self.log.debug("Fixed parameter %r ignored.", p)
            else:
                raise LoggedError(
                    self.log, "Cannot change value of constant parameter: "
                              "%s = %g (new) vs %g (old).", p, value, known_value)
        for p in to_pop:
            sampled_input.pop(p)
        if sampled_input:
            not_used = set(sampled_input)
            duplicated = not_used.intersection(set(
                chain(
                    *[list(chain(*[[k], v])) for k, v in self._sampled_renames.items()])))
            not_used = not_used.difference(duplicated)
            derived = not_used.intersection(set(self.derived_params()))
            input_ = not_used.intersection(set(self.input_params()))
            unknown = not_used.difference(derived).difference(input_)
            msg = ("Incorrect parameters! " +
                   ("\n   Duplicated entries (using their aliases): %r" % list(duplicated)
                    if duplicated else "") +
                   ("\n   Not known: %r" % list(unknown) if unknown else "") +
                   ("\n   Cannot be fixed: %r " % list(input_) +
                    "--> instead, fix sampled parameters that depend on them!"
                    if input_ else "") +
                   ("\n   Cannot be fixed because are derived parameters: %r " % list(
                       derived) if derived else ""))
            for line in msg.split("\n"):
                self.log.error(line)
            raise LoggedError
        return sampled_output

    def labels(self):
        """
        Returns a dictionary of LaTeX labels of the sampled and derived parameters.

        Uses the parameter name if no label has been given.
        """

        def get_label(p, info):
            return ensure_nolatex(getattr(info, "get", lambda x, y: y)
                                  (partag.latex, p.replace("_", r"\ ")))

        return {p: get_label(p, info) for p, info in self._infos.items()}

    def _call_param_func(self, p, func, kwargs):
        try:
            return func(**kwargs)
        except NameError as exception:
            unknown = str(exception).split("'")[1]
            raise LoggedError(
                self.log, "Unknown variable '%s' was referenced in the definition of "
                          "the parameter '%s', with arguments %r.", unknown, p,
                list(kwargs))
        except:
            self.log.error("Function for parameter '%s' failed at evaluation "
                           "and threw the following exception:", p)
            raise

    def _get_wrapped_functions_evaluation_order(self):
        # get evaluation order for input and derived parameter function
        # and pre-prepare argument dicts

        wrapped_funcs = ({}, {})
        known = set(self._constant).union(self._sampled)

        for derived, wrapped_func in zip((False, True), wrapped_funcs):
            if derived:
                inputs = self._derived_funcs.copy()
                input_args = self._derived_args
                known.update(self._output)
                output = self._derived
            else:
                inputs = self._input_funcs.copy()
                input_args = self._input_args
                output = self._input

            while inputs:
                for p, func in inputs.items():
                    args = input_args[p]
                    if set(args).difference(known):
                        continue
                    known.add(p)

                    if not set(args).difference(self._constant):
                        # all inputs are constant, so output is constant and precomputed
                        self._constant[p] = \
                            self._call_param_func(p, func,
                                                  {arg: self._constant[arg] for arg in
                                                   args})
                        output[p] = self._constant[p]
                    else:
                        # Store function, argument dict with constants pre-filled,
                        # and unset args as tuple
                        wrapped_func[p] = \
                            (func, {arg: self._constant.get(arg) for arg in args},
                             [arg for arg in args if arg not in self._constant])

                    del inputs[p]
                    break
                else:
                    raise LoggedError(
                        self.log, "Could not resolve arguments for parameters %s. "
                                  "Maybe there is a circular dependency between derived "
                                  "parameters?", list(inputs))
        return wrapped_funcs

    # Python magic for the "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        return
