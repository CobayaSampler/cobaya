import contextlib
import numbers
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType, UnionType
from typing import Any, Literal, TypedDict

import numpy as np

InfoDict = dict[str, Any]
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

ParamValuesDict = dict[str, float]
# Do not yet explicitly support passing instances here
TheoriesDict = dict[str, None | TheoryDict | type]
LikesDict = dict[str, None | str | LikeDict | type | Callable]
SamplersDict = dict[str, SamplerDict | None]
PriorsDict = dict[str, str | Callable]

partags = {
    "prior",
    "ref",
    "proposal",
    "value",
    "drop",
    "derived",
    "latex",
    "renames",
    "min",
    "max",
    "periodic",
}

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
    value: float | Callable | str
    derived: bool | str | Callable
    prior: None | Sequence[float] | SciPyDistDict | SciPyMinMaxDict
    ref: None | float | Sequence[float] | SciPyDistDict | SciPyMinMaxDict
    proposal: float | None
    renames: str | Sequence[str]
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
ParamInput = ParamDict | None | str | float | Sequence[float]
ParamsDict = dict[str, ParamInput]
ExpandedParamsDict = dict[str, ParamDict]


class ModelDict(TypedDict, total=False):
    theory: TheoriesDict
    likelihood: LikesDict
    prior: PriorsDict
    params: ParamsDict
    auto_params: ParamsDict


class PostDict(TypedDict, total=False):
    add: ModelDict | None
    remove: None | ModelDict | dict[str, str | Sequence[str]]
    output: str | None
    suffix: str | None
    skip: None | float | int
    thin: int | None
    packages_path: str | None


class InputDict(ModelDict, total=False):
    sampler: SamplersDict
    post: PostDict
    force: bool
    debug: bool | int | str
    resume: bool
    minimize: bool
    stop_at_error: bool
    test: bool
    timing: bool
    packages_path: str | None
    output: str | None
    version: str | InfoDict | None


enforce_type_checking = None


@contextlib.contextmanager
def type_checking(value: bool):
    """
    Context manager to temporarily set typing.enforce_type_checking to a specific value.
    Restores the original value when exiting the context.
    """
    global enforce_type_checking
    original_value = enforce_type_checking
    enforce_type_checking = value
    try:
        yield
    finally:
        enforce_type_checking = original_value


def validate_type(expected_type: type, value: Any, path: str = ""):
    """
    Checks for soft compatibility of a value with a type.
    Raises TypeError with descriptive messages when validation fails.

    :param expected_type: from annotation
    :param value: value to validate
    :param path: string tracking the nested path for error messages
    :raises TypeError: with descriptive message when validation fails
    """
    if value is None or expected_type is Any:
        return

    curr_path = f"'{path}'" if path else "value"

    if expected_type is int:
        if not (value in (np.inf, -np.inf) or isinstance(value, numbers.Integral)):
            raise TypeError(f"{curr_path} must be an integer, got {type(value).__name__}")
        return

    if expected_type is float:
        if not (
            isinstance(value, numbers.Real)
            or (isinstance(value, np.ndarray) and value.ndim == 0)
        ):
            raise TypeError(f"{curr_path} must be a float, got {type(value).__name__}")
        return

    if expected_type is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{curr_path} must be boolean, got {type(value).__name__}")
        return

    from typing import is_typeddict

    if is_typeddict(expected_type):
        type_hints = typing.get_type_hints(expected_type)
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{curr_path} must be a mapping for TypedDict "
                f"'{expected_type.__name__}', got {type(value).__name__}"
            )
        if invalid_keys := set(value) - set(type_hints):
            raise TypeError(
                f"{curr_path} contains invalid keys for TypedDict "
                f"'{expected_type.__name__}': {invalid_keys}"
            )
        for key, val in value.items():
            validate_type(type_hints[key], val, f"{path}.{key}" if path else str(key))
        return

    if (origin := typing.get_origin(expected_type)) and (
        args := typing.get_args(expected_type)
    ):
        # complex types like dict[str, float] etc.

        if origin is typing.Union or origin is UnionType:
            errors = []
            for t in args:
                try:
                    return validate_type(t, value, path)
                except TypeError as e:
                    error_msg = str(e)
                    error_path = error_msg.split(" ")[0].strip("'")
                    errors.append((error_path, error_msg))

            longest_path = max((p for p, _ in errors), key=len)
            path_errors = {e for p, e in errors if p == longest_path}
            raise TypeError(
                f"{longest_path} failed to match any Union type:\n"
                + "\n".join(f"- {e}" for e in path_errors)
            )

        if not isinstance(origin, type):
            return validate_type(args[0], value, path)

        if isinstance(value, Mapping) != issubclass(origin, Mapping):
            raise TypeError(
                f"{curr_path} must be {origin.__name__}, got {type(value).__name__}"
            )

        if issubclass(origin, Mapping):
            for k, v in value.items():
                key_path = f"{path}[{k!r}]" if path else f"[{k!r}]"
                validate_type(args[0], k, f"{key_path} (key)")
                validate_type(args[1], v, key_path)
            return

        if issubclass(origin, Iterable):
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    raise TypeError(f"{curr_path} numpy array zero rank")
                if len(args) == 1 and not np.issubdtype(value.dtype, args[0]):
                    raise TypeError(
                        f"{curr_path} numpy array has wrong dtype: "
                        f"expected {args[0]}, got {value.dtype}"
                    )
                return

            if len(args) == 1:
                if not isinstance(value, Iterable):
                    raise TypeError(
                        f"{curr_path} must be iterable, got {type(value).__name__}"
                    )
                for i, item in enumerate(value):
                    validate_type(args[0], item, f"{path}[{i}]" if path else f"[{i}]")
            else:
                if not isinstance(value, Sequence):
                    raise TypeError(
                        f"{curr_path} must be a sequence for "
                        f"tuple types, got {type(value).__name__}"
                    )
                if len(args) != len(value):
                    raise TypeError(
                        f"{curr_path} has wrong length: "
                        f"expected {len(args)}, got {len(value)}"
                    )
                for i, (t, v) in enumerate(zip(args, value)):
                    validate_type(t, v, f"{path}[{i}]" if path else f"[{i}]")
            return

    if (
        not isinstance(expected_type, type)
        or isinstance(value, expected_type)
        or expected_type is Sequence
        and isinstance(value, np.ndarray)
    ):
        return

    type_name = getattr(expected_type, "__name__", repr(expected_type))

    # special case for Cobaya's NumberWithUnits, if not instance yet
    if type_name == "NumberWithUnits":
        if not isinstance(value, (numbers.Real, str)):
            raise TypeError(
                f"{curr_path} must be a number or string for NumberWithUnits,"
                f" got {type(value).__name__}"
            )
        return

    raise TypeError(
        f"{curr_path} must be of type {type_name}, got {type(value).__name__}"
    )
