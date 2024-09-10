"""General test for types of components."""

from typing import Any, ClassVar, Dict, List, Optional, Tuple
import numpy as np
import pytest

from cobaya.component import CobayaComponent
from cobaya.likelihood import Likelihood
from cobaya.tools import NumberWithUnits
from cobaya.typing import InputDict, ParamDict
from cobaya.run import run


class GenericLike(Likelihood):
    any: Any
    classvar: ClassVar[int] = 1
    forwardref_params: "ParamDict" = {"d": [0.0, 1.0]}
    infinity: int = float("inf")
    mean: NumberWithUnits = 1
    noise: float = 0
    none: int = None
    numpy_int: int = np.int64(1)
    optional: Optional[int] = None
    paramdict_params: ParamDict = {"c": [0.0, 1.0]}
    params: Dict[str, List[float]] = {"a": [0.0, 1.0], "b": [0, 1]}
    tuple_params: Tuple[float, float] = (0.0, 1.0)

    enforce_types = True

    def logp(self, **params_values):
        return 1


def test_sampler_types():
    original_info: InputDict = {
        "likelihood": {"like": GenericLike},
        "sampler": {"mcmc": {"max_samples": 1, "enforce_types": True}},
    }
    _ = run(original_info)

    info = original_info.copy()
    info["sampler"]["mcmc"]["max_samples"] = "not_an_int"
    with pytest.raises(TypeError):
        run(info)


class GenericComponent(CobayaComponent):
    any: Any
    classvar: ClassVar[int] = 1
    forwardref_params: "ParamDict" = {"d": [0.0, 1.0]}
    infinity: int = float("inf")
    mean: NumberWithUnits = 1
    noise: float = 0
    none: int = None
    numpy_int: int = np.int64(1)
    optional: Optional[int] = None
    paramdict_params: ParamDict = {"c": [0.0, 1.0]}
    params: Dict[str, List[float]] = {"a": [0.0, 1.0], "b": [0, 1]}
    tuple_params: Tuple[float, float] = (0.0, 1.0)

    enforce_types = True

    def __init__(
        self,
        any,
        classvar,
        forwardref_params,
        infinity,
        mean,
        noise,
        none,
        numpy_int,
        optional,
        paramdict_params,
        params,
        tuple_params,
    ):
        if self.enforce_types:
            super().validate_attributes()


def test_component_types():
    correct_kwargs = {
        "any": 1,
        "classvar": 1,
        "forwardref_params": {"d": [0.0, 1.0]},
        "infinity": float("inf"),
        "mean": 1,
        "noise": 0,
        "none": None,
        "numpy_int": 1,
        "optional": 3,
        "paramdict_params": {"c": [0.0, 1.0]},
        "params": {"a": [0.0, 1.0], "b": [0, 1]},
        "tuple_params": (0.0, 1.0),
    }
    _ = GenericComponent(**correct_kwargs)

    wrong_cases = [
        {"any": "not_an_int"},
        {"classvar": "not_an_int"},
        {"forwardref_params": "not_a_paramdict"},
        {"infinity": "not_an_int"},
        {"mean": "not_a_numberwithunits"},
        {"noise": "not_a_float"},
        {"none": "not_a_none"},
        {"numpy_int": "not_an_int"},
        {"paramdict_params": "not_a_paramdict"},
        {"params": "not_a_dict"},
        {"params": {1: [0.0, 1.0]}},
        {"params": {"a": "not_a_list"}},
        {"params": {"a": [0.0, "not_a_float"]}},
        {"optional": "not_an_int"},
        {"tuple_params": "not_a_tuple"},
        {"tuple_params": (0.0, "not_a_float")},
    ]
    for case in wrong_cases:
        with pytest.raises(TypeError):
            _ = GenericComponent({**correct_kwargs, **case})
