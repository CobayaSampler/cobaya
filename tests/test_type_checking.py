"""General test for types of components."""

from typing import Any, ClassVar, Dict, List, Optional, Tuple
import numpy as np
import pytest

from cobaya.component import CobayaComponent
from cobaya.likelihood import Likelihood
from cobaya.tools import NumberWithUnits
from cobaya.typing import InputDict, ParamDict, Sequence
from cobaya.run import run


class GenericLike(Likelihood):
    any: Any
    classvar: ClassVar[int] = 1
    infinity: int = float("inf")
    mean: NumberWithUnits = 1
    noise: float = 0
    none: int = None
    numpy_int: int = np.int64(1)
    optional: Optional[int] = None
    paramdict_params: ParamDict = {"prior": [0.0, 1.0]}
    params: Dict[str, List[float]] = {"a": [0.0, 1.0], "b": [0, 1]}
    tuple_params: Tuple[float, float] = (0.0, 1.0)

    _enforce_types = True

    def logp(self, **params_values):
        return 1


def test_sampler_types():
    original_info: InputDict = {
        "likelihood": {"like": GenericLike},
        "sampler": {"mcmc": {"max_samples": 1}},
    }
    _ = run(original_info)

    info = original_info.copy()
    info["sampler"]["mcmc"]["max_samples"] = "not_an_int"
    with pytest.raises(TypeError):
        run(info)


class GenericComponent(CobayaComponent):
    any: Any
    classvar: ClassVar[int] = 1
    infinity: int = float("inf")
    mean: NumberWithUnits = 1
    noise: float = 0
    none: int = None
    numpy_int: int = np.int64(1)
    optional: Optional[int] = None
    paramdict_params: ParamDict
    params: Dict[str, List[float]]
    tuple_params: Tuple[float, float] = (0.0, 1.0)
    array: Sequence[float]
    array2: Sequence[float]

    _enforce_types = True


def test_component_types():
    correct_kwargs = {
        "any": 1,
        "classvar": 1,
        "infinity": float("inf"),
        "mean": 1,
        "noise": 0,
        "none": None,
        "numpy_int": 1,
        "optional": 3,
        "paramdict_params": {"prior": [0.0, 1.0]},
        "params": {"a": [0.0, 1.0], "b": [0, 1]},
        "tuple_params": (0.0, 1.0),
        "array": np.arange(2, dtype=np.float64),
        "array2": [1, 2]
    }
    GenericComponent(correct_kwargs)

    wrong_cases = [
        {"classvar": "not_an_int"},
        {"infinity": "not_an_int"},
        {"mean": {}},
        {"noise": "not_a_float"},
        {"none": "not_a_none"},
        {"numpy_int": "not_an_int"},
        {"paramdict_params": {"prior": {"c": 1}}},
        {"params": "not_a_dict"},
        {"params": {1: [0.0, 1.0]}},
        {"params": {"a": "not_a_list"}},
        {"params": {"a": [0.0, "not_a_float"]}},
        {"optional": "not_an_int"},
        {"tuple_params": "not_a_tuple"},
        {"tuple_params": (0.0, "not_a_float")},
        {"array": 2},
    ]
    for case in wrong_cases:
        with pytest.raises(TypeError):
            GenericComponent({**correct_kwargs, **case})
