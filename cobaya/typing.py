from typing import Dict, Any, Optional, Union, Sequence, Type, Callable
import numpy as np
from types import MappingProxyType

# an immutable empty dict (e.g. for argument defaults)
empty_dict = MappingProxyType({})
# empty list-type object that can be used like None to test if assigned
unset_params: Sequence[str] = ()

InfoDict = Dict[str, Any]
LikeDict = InfoDict
TheoryDict = InfoDict
SamplerDict = InfoDict

ParamValuesDict = Dict[str, float]
TheoriesDict = Dict[str, Union[None, TheoryDict, Type]]
LikesDict = Dict[str, Union[None, LikeDict, Type]]
SamplersDict = Dict[str, Optional[SamplerDict]]
PriorsDict = Dict[str, Union[str, Callable]]
# parameters in a params list can be specified on input by
# 1. a ParamDict dictionary
# 2. constant value
# 3. a string giving lambda function of other parameters
# 4. None - must be a computed output parameter
# 5. Sequence specifying uniform prior range [min, max] and optionally
#    'ref' mean and standard deviation for starting positions, and optionally
#    proposal width. Allowed lengths, 2, 4, 5
ParamSpec = Union['ParamDict', None, str, float, Sequence[float]]
ParamsDict = Dict[str, ParamSpec]
ExpandedParamsDict = Dict[str, 'ParamDict']

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


    # partags = set(ParamDict.__annotations__)

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
