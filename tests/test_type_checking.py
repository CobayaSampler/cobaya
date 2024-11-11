"""General test for types of components."""

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Mapping
import numpy as np
import pytest

from cobaya.component import CobayaComponent
from cobaya.tools import NumberWithUnits
from cobaya.typing import ParamDict, Sequence


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
    map: Mapping[float, str]
    deferred: 'ParamDict'
    unset = 1
    install_options: ClassVar

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
        "array2": [1, 2],
        "map": {1.0: "a", 2.0: "b"},
        "deferred": {'value': lambda x: x},
        "install_options": {}
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
        {"map": {"a": 2.0}}
    ]
    for case in wrong_cases:
        with pytest.raises(TypeError):
            GenericComponent({**correct_kwargs, **case})


class NextComponent(CobayaComponent):
    pass
