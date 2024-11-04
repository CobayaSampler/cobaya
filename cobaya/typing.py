from typing import Dict, Any, Optional, Union, Type, TypedDict, Literal, Mapping, \
    Callable, Sequence, Iterable
from types import MappingProxyType
import typing
import numbers
import numpy as np
import sys

InfoDict = Dict[str, Any]
InfoDictIn = Mapping[str, Any]

# an immutable empty dict (e.g. for argument defaults)
empty_dict: InfoDictIn = MappingProxyType({})
# empty iterator-compatible object that can be used like None to test if assigned
unset_params: Sequence[str] = ()

LikeDict = InfoDict
TheoryDict = InfoDict
SamplerDict = InfoDict

# Read-only versions for input
LikeDictIn = InfoDictIn
TheoryDictIn = InfoDictIn
SamplerDictIn = InfoDictIn

ParamValuesDict = Dict[str, float]
# Do not yet explicitly support passing instances here
TheoriesDict = Dict[str, Union[None, TheoryDict, Type]]
LikesDict = Dict[str, Union[None, LikeDict, Type, Callable]]
SamplersDict = Dict[str, Optional[SamplerDict]]
PriorsDict = Dict[str, Union[str, Callable]]

partags = {"prior", "ref", "proposal", "value", "drop",
           "derived", "latex", "renames", "min", "max"}

LiteralFalse = Literal[False]
ModelBlock = Literal["theory", "likelihood", "prior", "params"]
Kind = Literal["theory", "likelihood", "sampler"]


class SciPyDistDict(TypedDict):
    dist: str
    loc: float
    scale: float


class SciPyMinMaxDict(TypedDict, total=False):
    dist: str  # default uniform
    min: float
    max: float


class ParamDict(TypedDict, total=False):
    value: Union[float, Callable, str]
    derived: Union[bool, str, Callable]
    prior: Union[None, Sequence[float], SciPyDistDict, SciPyMinMaxDict]
    ref: Union[None, Sequence[float], SciPyDistDict, SciPyMinMaxDict]
    proposal: Optional[float]
    renames: Union[str, Sequence[str]]
    latex: str
    drop: bool  # true if parameter should not be available for assignment to theories
    min: float  # hard bounds (does not affect prior)
    max: float


# parameters in a params list can be specified on input by
# 1. a ParamDict dictionary
# 2. constant value
# 3. a string giving lambda function of other parameters
# 4. None - must be a computed output parameter
# 5. Sequence specifying uniform prior range [min, max] and optionally
#    'ref' mean and standard deviation for starting positions, and optionally
#    proposal width. Allowed lengths, 2, 4, 5
ParamInput = Union[ParamDict, None, str, float, Sequence[float]]
ParamsDict = Dict[str, ParamInput]
ExpandedParamsDict = Dict[str, ParamDict]


class ModelDict(TypedDict, total=False):
    theory: TheoriesDict
    likelihood: LikesDict
    prior: PriorsDict
    params: ParamsDict
    auto_params: ParamsDict


class PostDict(TypedDict, total=False):
    add: Optional[ModelDict]
    remove: Union[None, ModelDict, Dict[str, Union[str, Sequence[str]]]]
    output: Optional[str]
    suffix: Optional[str]
    skip: Union[None, float, int]
    thin: Optional[int]
    packages_path: Optional[str]


class InputDict(ModelDict, total=False):
    sampler: SamplersDict
    post: PostDict
    force: bool
    debug: Union[bool, int, str]
    resume: bool
    stop_at_error: bool
    test: bool
    timing: bool
    packages_path: Optional[str]
    output: Optional[str]
    version: Optional[Union[str, InfoDict]]


enforce_type_checking = None


def validate_type(expected_type: type, value: Any, path: str = ''):
    """
    Checks for soft compatibility of a value with a type.
    Raises TypeError with descriptive messages when validation fails.

    :param expected_type: from annotation
    :param value: value to validate
    :param path: string tracking the nested path for error messages
    :raises TypeError: with descriptive message when validation fails
    """
    curr_path = f"'{path}'" if path else 'value'

    if value is None or expected_type is Any:
        return

    if expected_type is int:
        if not (value in (np.inf, -np.inf) or isinstance(value, numbers.Integral)):
            raise TypeError(
                f"{curr_path} must be an integer or infinity, got {type(value).__name__}"
            )
        return

    if expected_type is float:
        if not (isinstance(value, numbers.Real) or
                (isinstance(value, np.ndarray) and value.shape == ())):
            raise TypeError(f"{curr_path} must be a float, got {type(value).__name__}")
        return

    if expected_type is bool:
        if not hasattr(value, '__bool__') and not isinstance(value, (str, np.ndarray)):
            raise TypeError(
                f"{curr_path} must be boolean, got {type(value).__name__}"
            )
        return

        # special case for Cobaya

    if sys.version_info < (3, 10):
        from typing_extensions import is_typeddict
    else:
        from typing import is_typeddict

    if is_typeddict(expected_type):
        type_hints = typing.get_type_hints(expected_type)
        if not isinstance(value, Mapping):
            raise TypeError(f"{curr_path} must be a mapping for TypedDict "
                            f"'{expected_type.__name__}', got {type(value).__name__}")
        if invalid_keys := set(value) - set(type_hints):
            raise TypeError(f"{curr_path} contains invalid keys for TypedDict "
                            f"'{expected_type.__name__}': {invalid_keys}")
        for key, val in value.items():
            validate_type(type_hints[key], val, f"{path}.{key}" if path else str(key))
        return True

    if origin := typing.get_origin(expected_type):
        args = typing.get_args(expected_type)

        if origin is Union:
            errors = []
            structural_errors = []

            for t in args:
                try:
                    return validate_type(t, value, path)
                except TypeError as e:
                    error_msg = str(e)
                    error_path = error_msg.split(' ')[0].strip("'")

                    # If error is about the current path, it's a structural error
                    if error_path == path:
                        # Skip uninformative "must be of type NoneType" errors
                        if "must be of type NoneType" not in error_msg:
                            structural_errors.append(error_msg)
                        else:
                            errors.append((error_path, error_msg))
                    else:
                        errors.append((error_path, error_msg))

            # If we have structural errors, show those first
            if structural_errors:
                if len(structural_errors) == 1:
                    raise TypeError(structural_errors[0])
                raise TypeError(
                    f"{curr_path} failed to match any Union type:\n" +
                    "\n".join(f"- {e}" for e in set(structural_errors))
                )

            # Otherwise, show the deepest validation errors
            longest_path = max((p for p, _ in errors), key=len)
            path_errors = list(set(e for p, e in errors if p == longest_path))
            raise TypeError(
                f"{longest_path} failed to match any Union type:\n" +
                "\n".join(f"- {e}" for e in path_errors)
            )

        if origin is typing.ClassVar:
            return validate_type(args[0], value, path)

        if origin in (dict, Mapping):
            if not isinstance(value, Mapping):
                raise TypeError(f"{curr_path} must be a mapping, "
                                f"got {type(value).__name__}")
            for k, v in value.items():
                key_path = f"{path}[{k!r}]" if path else f"[{k!r}]"
                validate_type(args[0], k, f"{key_path} (key)")
                validate_type(args[1], v, key_path)
            return

        if issubclass(origin, Iterable):
            if isinstance(value, np.ndarray):
                if not value.shape:
                    raise TypeError(f"{curr_path} numpy array zero rank")
                if len(args) == 1 and not np.issubdtype(value.dtype, args[0]):
                    raise TypeError(
                        f"{curr_path} numpy array has wrong dtype: "
                        f"expected {args[0]}, got {value.dtype}"
                    )
                return

            if not isinstance(value, Iterable):
                raise TypeError(
                    f"{curr_path} must be iterable, got {type(value).__name__}"
                )

            if len(args) == 1:
                for i, item in enumerate(value):
                    validate_type(args[0], item, f"{path}[{i}]" if path else f"[{i}]")
            else:
                if not isinstance(value, Sequence):
                    raise TypeError(f"{curr_path} must be a sequence for "
                                    f"tuple types, got {type(value).__name__}")
                if len(args) != len(value):
                    raise TypeError(f"{curr_path} has wrong length: "
                                    f"expected {len(args)}, got {len(value)}")
                for i, (t, v) in enumerate(zip(args, value)):
                    validate_type(t, v, f"{path}[{i}]" if path else f"[{i}]")
            return

    if not (isinstance(value, expected_type) or
            expected_type is Sequence and isinstance(value, np.ndarray)):

        # special case for Cobaya's NumberWithUnits, if not instance yet
        if getattr(expected_type, "__name__", "") == 'NumberWithUnits':
            if not isinstance(value, (numbers.Real, str)):
                raise TypeError(
                    f"{curr_path} must be a number or string for NumberWithUnits,"
                    f" got {type(value).__name__}")
            return

        raise TypeError(f"{curr_path} must be of type {expected_type.__name__}, "
                        f"got {type(value).__name__}")
