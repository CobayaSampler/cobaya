"""
Tests the setting and updating of the reference pdf, including re-checking if point-like.
"""
import pytest
import numpy as np

from cobaya.model import get_model
from cobaya.sampler import get_sampler
from cobaya import mpi


def test_ref():
    val = 1
    mean, std = 0.5, 0.1
    info = {
        "params": {
            "a": {"prior": [0, 1]},
            "b": {"prior": [0, 1], "ref": None},
            "c": {"prior": [0, 1], "ref": val},
            "d": {"prior": [0, 1], "ref": [mean, std]},
            "e": {"prior": [0, 1], "ref": {"dist": "norm", "loc": mean, "scale": std}}},
        "likelihood": {"one": None}}
    model = get_model(info)
    for i in [3, 4]:
        assert model.prior.ref_pdf[i].dist.name == "norm"
        assert model.prior.ref_pdf[i].mean() == mean
        assert model.prior.ref_pdf[i].std() == std
    # Not point-like: 2 nan's (use prior) and 2 norms
    assert not model.prior.reference_is_pointlike
    # Let's update it -- remove norms
    upd1_ref_info = {
        "d": val + 2,
        "e": val + 3}
    model.prior.set_reference(upd1_ref_info)
    # Updated:
    assert model.prior.ref_pdf[3] == upd1_ref_info["d"]
    assert model.prior.ref_pdf[4] == upd1_ref_info["e"]
    # Unchanged:
    assert model.prior.ref_pdf[0] is np.nan
    assert model.prior.ref_pdf[1] is np.nan
    assert model.prior.ref_pdf[2] == val
    # Still not pointlike: there are nan's --> use prior
    assert not model.prior.reference_is_pointlike
    # Let's update it -- remove nan's
    upd2_ref_info = {
        "a": val - 2,
        "b": val - 1}
    model.prior.set_reference(upd2_ref_info)
    # Updated:
    assert model.prior.ref_pdf[0] == upd2_ref_info["a"]
    assert model.prior.ref_pdf[1] == upd2_ref_info["b"]
    # Unchanged:
    assert model.prior.ref_pdf[2] == val
    assert model.prior.ref_pdf[3] == upd1_ref_info["d"]
    assert model.prior.ref_pdf[4] == upd1_ref_info["e"]
    # Should be point-like now!
    assert model.prior.reference_is_pointlike
    # Let's update it -- back to one norm
    upd3_ref_info = {
        "a": [mean, std]}
    model.prior.set_reference(upd3_ref_info)
    # Updated:
    assert model.prior.ref_pdf[0].dist.name == "norm"
    assert model.prior.ref_pdf[0].mean() == mean
    assert model.prior.ref_pdf[0].std() == std
    # Unchanged:
    assert model.prior.ref_pdf[1] == upd2_ref_info["b"]
    assert model.prior.ref_pdf[2] == val
    assert model.prior.ref_pdf[3] == upd1_ref_info["d"]
    assert model.prior.ref_pdf[4] == upd1_ref_info["e"]
    # Should be non-point-like again
    assert not model.prior.reference_is_pointlike


# Tests MPI-awareness of the reference, when using it to get the initial point of mcmc
@pytest.mark.mpi
@mpi.sync_errors
def test_ref_mcmc_initial_point():
    val = 0.5
    info = {
        "params": {"a": {"prior": [0, 1 + mpi.size()], "ref": val + mpi.rank()}},
        "likelihood": {"one": None}}
    model = get_model(info)
    mcmc_sampler = get_sampler({"mcmc": None}, model)
    initial_point = mcmc_sampler.current_point.values[0]
    assert initial_point == val + mpi.rank()
