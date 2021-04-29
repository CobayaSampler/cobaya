from typing import Dict, Any, Optional, Union, Sequence, Type, Tuple, List
import numpy as np
from types import MappingProxyType

# an immutable empty dict (e.g. for argument defaults)
empty_dict = MappingProxyType({})

InfoDict = Dict[str, Any]
LikeDict = InfoDict
TheoryDict = InfoDict
SamplerDict = InfoDict

ParamValuesDict = Dict[str, Optional[float]]
TheoriesDict = Dict[str, Union[None, TheoryDict, Type]]
LikesDict = Dict[str, Union[None, LikeDict, Type]]
SamplersDict = Dict[str, Optional[SamplerDict]]
PriorsDict = Dict[str, Union[str, callable]]

partags = {"prior", "ref", "proposal", "value", "drop",
           "derived", "latex", "renames", "min", "max"}

try:
    # noinspection PyUnresolvedReferences
    from typing import TypedDict
except ImportError:
    InputDict = InfoDict
    ParamDict = InfoDict
    ModelDict = InfoDict
    PostDict = InfoDict
    ParamsDict = Dict[str, Union[ParamDict, None, str, float, List[float]]]
else:

    class SciPyDistDict(TypedDict):
        dist: str
        loc: float
        scale: float


    class SciPyMinMaxDict(TypedDict, total=False):
        dist: str  # default uniform
        min: float
        max: float


    class ParamDict(TypedDict, total=False):
        value: Union[float, callable, str]
        derived: Union[bool, str, callable]
        prior: Union[None, Tuple[float, float], SciPyDistDict, SciPyMinMaxDict]
        ref: Union[None, Tuple[float, float], SciPyDistDict, SciPyMinMaxDict]
        proposal: Optional[float]
        renames: Union[str, Sequence[str]]
        latex: str
        drop: bool  # true if parameter should not be available for assignment to theories
        min: float  # hard bounds (does not affect prior)
        max: float


    # partags = set(ParamDict.__annotations__)

    ParamsDict = Dict[str, Union[ParamDict, None, str, float, List[float]]]


    class ModelDict(TypedDict, total=False):
        theory: TheoriesDict
        likelihood: LikesDict
        prior: PriorsDict
        params: ParamsDict


    class PostDict(TypedDict, total=False):
        add: Optional[ModelDict]
        remove: Optional[ModelDict]
        output: Optional[str]
        suffix: Optional[str]
        skip: Union[None, float, int]
        thin: Optional[int]
        packages_path: Optional[str]


    class InputDict(ModelDict, total=False):
        sampler: SamplersDict
        post: Optional[PostDict]
        force: bool
        debug: bool
        debug_file: Optional[str]
        resume: bool
        stop_at_error: bool
        test: bool
        timing: bool
        packages_path: Optional[str]
        output: Optional[str]
        version: Optional[Union[str, InfoDict]]

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Union[Sequence, np.ndarray]

OptionalArrayLike = Optional[ArrayLike]
ArrayOrFloat = Union[float, ArrayLike]
