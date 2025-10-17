"""
.. module:: parameterization

:Synopsis: Class managing the possibly different parameterizations
           used by sampler and likelihoods
:Author: Jesus Torrado

"""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from itertools import chain
from numbers import Real
from typing import Any

import numpy as np

from cobaya.log import HasLogger, LoggedError
from cobaya.tools import (
    deepcopy_where_possible,
    ensure_nolatex,
    get_external_function,
    get_scipy_1d_pdf,
    getfullargspec,
    invert_dict,
    is_valid_variable_name,
    str_to_list,
)
from cobaya.typing import (
    ExpandedParamsDict,
    ParamDict,
    ParamInput,
    ParamsDict,
    ParamValuesDict,
    partags,
)


def is_fixed_or_function_param(info_param: ParamInput) -> bool:
    """
    Returns True if the parameter has been fixed to a value or through a function.
    """
    return expand_info_param(info_param).get("value") is not None


def is_sampled_param(info_param: ParamInput) -> bool:
    """
    Returns True if the parameter has a prior.
    """
    return "prior" in expand_info_param(info_param)


def is_derived_param(info_param: ParamInput) -> bool:
    """
    Returns True if the parameter is saved as a derived one.
    """
    return expand_info_param(info_param).get("derived", False) is not False


def expand_info_param(info_param: ParamInput, default_derived=True) -> ParamDict:
    """
    Expands the info of a parameter, from the user-friendly, shorter format
    to a more unambiguous one.
    """
    info_param = deepcopy_where_possible(info_param)
    if not isinstance(info_param, Mapping):
        if info_param is None:
            info_param = {}
        elif isinstance(info_param, Sequence) and not isinstance(info_param, str):
            values = list(info_param)
            allowed_lengths = [2, 4, 5]
            if len(values) not in allowed_lengths:
                raise LoggedError(
                    __name__,
                    "Parameter info length not valid: %d. "
                    "The allowed lengths are %r. See documentation.",
                    len(values),
                    allowed_lengths,
                )
            info_param = {"prior": [values[0], values[1]]}
            if len(values) >= 4:
                info_param["ref"] = [values[2], values[3]]
                if len(values) == 5:
                    info_param["proposal"] = values[4]
        else:
            info_param = {"value": info_param}
    if all(f not in info_param for f in ["prior", "value", "derived"]):
        info_param["derived"] = default_derived
    # Dynamical input parameters: save as derived by default
    value = info_param.get("value")
    if isinstance(value, str) or callable(value):
        info_param["derived"] = info_param.get("derived", True)
    return info_param


def reduce_info_param(info_param: ParamDict) -> ParamInput:
    """
    Compresses the info of a parameter, suppressing default values.
    This is the opposite of :func:`~input.expand_info_param`.
    """
    info_param = deepcopy_where_possible(info_param)
    if not isinstance(info_param, dict):
        return None
    # All parameters without a prior are derived parameters unless otherwise specified
    if info_param.get("derived") is True:
        info_param.pop("derived")
    # Fixed parameters with single "value" key
    if list(info_param) == ["value"] and not callable(info_param["value"]):
        return info_param["value"]
    return info_param


_WrappedFunc = tuple[Callable, dict[str, Any], list[str]]


class Parameterization(HasLogger):
    """
    Class managing parameterization.
    Translates parameter between sampler+prior and likelihood
    """

    def __init__(
        self,
        info_params: ParamsDict | ExpandedParamsDict,
        allow_renames=True,
        ignore_unused_sampled=False,
    ):
        self.set_logger()
        self.allow_renames = allow_renames
        # First, we load the parameters,
        # not caring about whether they are understood by any likelihood.
        # `input` contains the parameters (expected to be) understood by the likelihood,
        #   with its fixed value, its fixing function, or None if their value is given
        #   directly by the sampler.
        self._infos = {}
        self._input: ParamValuesDict = {}
        self._input_funcs = {}
        self._input_args = {}
        self._input_dependencies: dict[str, set[str]] = {}
        self._dropped: set[str] = set()
        self._output: ParamValuesDict = {}
        self._constant: ParamValuesDict = {}
        self._sampled: ParamValuesDict = {}
        self._sampled_renames: dict[str, list[str]] = {}
        self._derived: ParamValuesDict = {}
        self._derived_inputs = []
        self._derived_funcs = {}
        self._derived_args = {}
        self._derived_dependencies: dict[str, set[str]] = {}
        # Notice here that expand_info_param *always* adds a "derived":True tag
        # to infos without "prior" or "value", and a "value" field
        # to fixed params
        for p, info in info_params.items():
            if isinstance(info, Mapping) and not set(info).issubset(partags):
                raise LoggedError(
                    self.log,
                    "Parameter '%s' has unknown options %s",
                    p,
                    set(info).difference(partags),
                )
            info = expand_info_param(info)
            self._infos[p] = info
            if is_fixed_or_function_param(info):
                if isinstance(info["value"], Real):
                    self._constant[p] = float(info["value"])
                    self._input[p] = self._constant[p]
                    if info.get("drop"):
                        self._dropped.add(p)
                else:
                    self._input[p] = np.nan
                    self._input_funcs[p] = get_external_function(info["value"])
                    self._input_args[p] = getfullargspec(self._input_funcs[p]).args
            if is_sampled_param(info):
                self._sampled[p] = np.nan
                self._input[p] = np.nan
                if info.get("drop"):
                    self._dropped.add(p)
                self._sampled_renames[p] = str_to_list(info.get("renames") or [])
            if is_derived_param(info):
                self._derived[p] = np.nan
                # Check for consistency for periodic parameters:
                if info.get("periodic", False) and None in (
                    info.get("min", None),
                    info.get("max", None),
                ):
                    raise LoggedError(
                        self.log,
                        f"Derived parameter '{p}' defined as periodic, but no range "
                        "specified with 'min' and 'max'.",
                    )
                # Dynamical parameters whose value we want to save
                if info["derived"] is True and is_fixed_or_function_param(info):
                    # parameters that are already known or computed by input funcs
                    self._derived_inputs.append(p)
                elif info["derived"] is True:
                    self._output[p] = np.nan
                else:
                    self._derived_funcs[p] = get_external_function(info["derived"])
                    self._derived_args[p] = getfullargspec(self._derived_funcs[p]).args
        # Check that the sampled and derived params are all valid python variable names
        for p in chain(self._sampled, self._derived):
            if not is_valid_variable_name(p):
                is_in = p in self._sampled
                eg_in = (
                    "  p_prime:\n    prior: ...\n  %s: 'lambda p_prime: p_prime'\n" % p
                )
                eg_out = f"  p_prime: 'lambda {p}: {p}'\n"
                raise LoggedError(
                    self.log,
                    "Parameter name '%s' is not a valid Python variable name "
                    "(it needs to start with a letter or '_').\n"
                    "If this is an %s parameter of a likelihood or theory, "
                    "whose name you cannot change,%s define an associated "
                    "%s one with a valid name 'p_prime' as: \n\n%s",
                    p,
                    "input" if is_in else "output",
                    "" if is_in else " remove it and",
                    "sampled" if is_in else "derived",
                    eg_in if is_in else eg_out,
                )

        # input params depend on input and sampled only,
        # never on output/derived unless constant
        known_input = set(self._input)
        all_input_arguments = set(chain(*self._input_args.values()))
        if bad_input_dependencies := all_input_arguments - known_input:
            raise LoggedError(
                self.log,
                "Input parameters defined as functions can only depend on other "
                "input parameters. In particular, an input parameter cannot depend on %r."
                " Use an explicit Theory calculator for more complex dependencies.\n"
                "If you intended to define a derived output parameter use derived: "
                "instead of value:",
                list(bad_input_dependencies),
            )

        # Assume that the *un*known function arguments are likelihood/theory
        # output parameters
        for arg in (
            all_input_arguments.union(*self._derived_args.values())
            .difference(known_input)
            .difference(self._derived)
        ):
            self._output[arg] = np.nan

        # Useful set: directly "output-ed" derived
        self._directly_output = [p for p in self._derived if p in self._output]

        self._wrapped_input_funcs, self._wrapped_derived_funcs = (
            self._get_wrapped_functions_evaluation_order()
        )

        # Useful mapping: input params that vary if each sample is varied
        self._sampled_input_dependence = {
            s: [i for i in self._input if s in self._input_dependencies.get(i, {})]
            for s in self._sampled
        }
        # From here on, some error control.
        # Only actually raise error after checking if used by prior.
        if not ignore_unused_sampled:
            self._dropped_not_directly_used = self._dropped.intersection(
                p for p, v in self._sampled_input_dependence.items() if not v
            )
        else:
            self._dropped_not_directly_used = set()

        # warn if repeated labels
        labels_inv_repeated = invert_dict(self.labels())
        labels_inv_repeated = {k: v for k, v in labels_inv_repeated.items() if len(v) > 1}
        if labels_inv_repeated:
            self.mpi_warning(
                "There are repeated parameter labels: %r", labels_inv_repeated
            )

    def dropped_param_set(self) -> set[str]:
        return self._dropped.copy()

    def input_params(self) -> ParamValuesDict:
        return self._input.copy()

    def output_params(self) -> ParamValuesDict:
        return self._output.copy()

    def constant_params(self) -> ParamValuesDict:
        return self._constant.copy()

    def sampled_params(self) -> ParamValuesDict:
        return self._sampled.copy()

    def sampled_params_info(self) -> ExpandedParamsDict:
        return {
            p: deepcopy_where_possible(info)
            for p, info in self._infos.items()
            if p in self._sampled
        }

    def sampled_params_renames(self) -> dict[str, list[str]]:
        return deepcopy(self._sampled_renames)

    def derived_params(self) -> ParamValuesDict:
        return self._derived.copy()

    def derived_params_info(self) -> ExpandedParamsDict:
        return {
            p: deepcopy_where_possible(info)
            for p, info in self._infos.items()
            if p in self._derived
        }

    def get_sampled_params_proposals(self) -> dict[str, float | None]:
        """
        Returns a dictionary of proposal values for sampled parameters.
        Returns None for parameters without a proposal value defined.
        """
        return {
            p: self._infos[p].get("proposal")
            for p in self._sampled
        }

    def sampled_input_dependence(self) -> dict[str, list[str]]:
        return deepcopy(self._sampled_input_dependence)

    def get_input_func(self, p, **params_values):
        func = self._input_funcs[p]
        args = self._input_args[p]
        return func(*[params_values.get(arg) for arg in args])

    @property
    def input_dependencies(self) -> dict[str, set[str]]:
        return self._input_dependencies

    def to_input(self, sampled_params_values) -> ParamValuesDict:
        # Gets all current sampled and input derived parameters as a dictionary,
        # including dropped parameters. Result is not a copy and must not be modified.

        # Store sampled params, so that derived can depend on them
        if not isinstance(sampled_params_values, dict):
            sampled_params_values = dict(zip(self._sampled, sampled_params_values))
        else:
            sampled_params_values = sampled_params_values.copy()
        self._sampled = sampled_params_values

        # First include all sampled input parameters,
        self._input.update(sampled_params_values)

        if self._wrapped_input_funcs:
            # Then evaluate the functions
            for p, (func, args, to_set) in self._wrapped_input_funcs.items():
                for arg in to_set:
                    args[arg] = self._input.get(arg, sampled_params_values.get(arg))
                self._input[p] = self._call_param_func(p, func, args)
        return self._input

    def to_derived(self, output_params_values) -> ParamValuesDict:
        if not isinstance(output_params_values, dict):
            output_params_values = dict(zip(self._output, output_params_values))
        # Fill first derived parameters which are direct output parameters
        for p in self._directly_output:
            self._derived[p] = output_params_values[p]
        for p in self._derived_inputs:
            self._derived[p] = self._input[p]
        # Then evaluate the functions
        if self._wrapped_derived_funcs:
            for p, (func, args, to_set) in self._wrapped_derived_funcs.items():
                for arg in to_set:
                    if (val := self._input.get(arg)) is None:
                        if (val := output_params_values.get(arg)) is None:
                            val = self._derived.get(arg)
                    args[arg] = val
                self._derived[p] = self._call_param_func(p, func, args)
        return self._derived

    def check_sampled(
        self, sampled_params: Sequence[float] | dict[str, float]
    ) -> Sequence[float] | dict[str, float]:
        """
        Performs some checks on the given sampled params.

        If an array is passed, the only test performed is for the right amount of
        parameters, and the same array is returned if successful.

        If a dictionary is passed, it checks that it contains all the sampled parameters,
        and just them. This function is aware of known renamings. Returns dict of
        parameters (model's naming, not renames) and their values.
        """
        if sampled_params is None:  # only works if there are no sampled params
            sampled_params = []
        if hasattr(sampled_params, "keys"):
            return self.check_sampled_dict(**sampled_params)
        else:
            if len(sampled_params) != len(self._sampled):
                raise LoggedError(
                    self.log,
                    "Wrong number of sampled parameters passed: %d given vs %d expected",
                    len(sampled_params),
                    len(self._sampled),
                )
            return sampled_params

    def check_sampled_dict(self, **sampled_params) -> ParamValuesDict:
        """
        Check that the input dictionary contains all the sampled parameters,
        and just them.

        This function is aware of known renamings.

        Returns dict of parameters (model's naming, not renames) and their values.
        """
        sampled_output: ParamValuesDict = {}
        for p, renames in self._sampled_renames.items():
            for pprime in sampled_params:
                if pprime == p or (pprime in renames if self.allow_renames else False):
                    sampled_output[p] = sampled_params.pop(pprime)
                    break
        if len(sampled_output) < len(self._sampled):
            not_found = set(self._sampled).difference(sampled_output)
            if self.allow_renames:
                msg = (
                    "The following expected sampled parameters "
                    + ("(or their aliases) " if self.allow_renames else "")
                    + "were not found : %r",
                    (
                        {p: self._sampled_renames[p] for p in not_found}
                        if self.allow_renames
                        else not_found
                    ),
                )
            else:
                msg = (
                    "The following expected sampled parameters were not found : %r",
                    {p: self._sampled_renames[p] for p in not_found},
                )
            raise LoggedError(self.log, *msg)
        # Ignore fixed input parameters if they have the correct value
        not_used = set(sampled_params)
        for p, value in sampled_params.items():
            known_value = self._constant.get(p)
            if known_value is None:
                raise LoggedError(self.log, "Unknown parameter %r.", p)
            elif np.allclose(value, known_value):
                not_used.remove(p)
                self.log.debug("Fixed parameter %r ignored.", p)
            else:
                raise LoggedError(
                    self.log,
                    "Cannot change value of constant parameter: "
                    "%s = %g (new) vs %g (old).",
                    p,
                    value,
                    known_value,
                )
        if not_used:
            duplicated = not_used.intersection(
                chain(*[list(chain(*[[k], v])) for k, v in self._sampled_renames.items()])
            )
            not_used = not_used.difference(duplicated)
            derived = not_used.intersection(self._derived)
            input_ = not_used.intersection(self._input)
            unknown = not_used.difference(derived).difference(input_)
            msg_text = (
                "Incorrect parameters! "
                + (
                    "\n   Duplicated entries (using their aliases): %r" % list(duplicated)
                    if duplicated
                    else ""
                )
                + ("\n   Not known: %r" % list(unknown) if unknown else "")
                + (
                    "\n   Cannot be fixed: %r " % list(input_)
                    + "--> instead, fix sampled parameters that depend on them!"
                    if input_
                    else ""
                )
                + (
                    "\n   Cannot be fixed because are derived parameters: %r "
                    % list(derived)
                    if derived
                    else ""
                )
            )
            for line in msg_text.split("\n"):
                self.log.error(line)
            raise LoggedError
        return sampled_output

    def check_dropped(self, external_dependence):
        # some error control, given external_dependence from prior
        # only raise error after checking not used by prior
        if self._dropped_not_directly_used.difference(external_dependence):
            raise LoggedError(
                self.log,
                "Parameters %r are sampled but not passed to a likelihood or theory "
                "code, and never used as arguments for any prior or parameter "
                "functions. Check that you are not using "
                "the '%s' tag unintentionally.",
                list(self._dropped_not_directly_used),
                "drop",
            )

    def labels(self) -> dict[str, str]:
        """
        Returns a dictionary of LaTeX labels of the sampled and derived parameters.

        Uses the parameter name if no label has been given.
        """

        def get_label(p, info):
            return ensure_nolatex(
                getattr(info, "get", lambda x, y: y)("latex", p.replace("_", r"\ "))
            )

        return {p: get_label(p, info) for p, info in self._infos.items()}

    def _call_param_func(self, p, func, kwargs):
        try:
            return func(**kwargs)
        except NameError as exception:
            unknown = str(exception).split("'")[1]
            raise LoggedError(
                self.log,
                "Unknown variable '%s' was referenced in the definition of "
                "the parameter '%s', with arguments %r.",
                unknown,
                p,
                list(kwargs),
            )
        except:
            self.log.error(
                "Function for parameter '%s' failed at evaluation "
                "and threw the following exception:",
                p,
            )
            raise

    def _get_wrapped_functions_evaluation_order(self):
        # get evaluation order for input and derived parameter function
        # and pre-prepare argument dicts

        wrapped_funcs: tuple[dict[str, _WrappedFunc], dict[str, _WrappedFunc]] = ({}, {})
        known = set(chain(self._constant, self._sampled))

        for derived, wrapped_func in zip((False, True), wrapped_funcs):
            if derived:
                inputs = self._derived_funcs.copy()
                input_args = self._derived_args
                known.update(self._output)
                output = self._derived
                dependencies = self._derived_dependencies
            else:
                inputs = self._input_funcs.copy()
                input_args = self._input_args
                output = self._input
                dependencies = self._input_dependencies

            while inputs:
                for p, func in inputs.items():
                    args = input_args[p]
                    if not known.issuperset(args):
                        continue
                    known.add(p)
                    dependencies[p] = set(
                        chain(args, *(dependencies.get(arg, []) for arg in args))
                    )

                    if set(args).issubset(self._constant):
                        # all inputs are constant, so output is constant and precomputed
                        self._constant[p] = self._call_param_func(
                            p, func, {arg: self._constant[arg] for arg in args}
                        )
                        output[p] = self._constant[p]
                    else:
                        # Store function, argument dict with constants pre-filled,
                        # and unset args as tuple
                        wrapped_func[p] = (
                            func,
                            {arg: self._constant.get(arg) for arg in args},
                            [arg for arg in args if arg not in self._constant],
                        )
                    del inputs[p]
                    break
                else:
                    raise LoggedError(
                        self.log,
                        "Could not resolve arguments for parameters %s. "
                        "Maybe there is a circular dependency between derived "
                        "parameters?",
                        list(inputs),
                    )
        return wrapped_funcs


def get_literal_param_range(param_info, confidence_for_unbounded=1):
    """
    Extracts parameter bounds from a parameter input dict, if present.
    """

    def get_bounds_from_dict(i):
        return [i.get("min", -np.inf), i.get("max", np.inf)]

    # Sampled
    if is_sampled_param(param_info):
        pdf_dist = get_scipy_1d_pdf(param_info.get("prior", {}))  # may raise ValueError
        lims = pdf_dist.interval(confidence_for_unbounded)
    # Derived
    elif is_derived_param(param_info):
        lims = get_bounds_from_dict(param_info or {})
    # Fixed
    else:
        value = expand_info_param(param_info).get("value", None)
        try:
            value = float(value)
        except (ValueError, TypeError):
            # e.g. lambda function values
            lims = get_bounds_from_dict(param_info or {})
        else:
            lims = (value, value)
    return lims[0] if lims[0] != -np.inf else None, lims[1] if lims[1] != np.inf else None


def get_literal_param_ranges(params_info, confidence_for_unbounded=1):
    """
    Extracts parameters bounds from a parameter input dict, or a
    :class:`~parameterization.Parameterization` instance.

    Notes
    -----
    Only use this if you know what you are doing. In general, you should get parameter
    bounds from a :class:`~model.Model` instance as :func:`model.Model.prior.bounds`.
    """
    if isinstance(params_info, Parameterization):
        params_info = params_info._infos
    return {
        p: get_literal_param_range(info, confidence_for_unbounded)
        for p, info in params_info.items()
    }
