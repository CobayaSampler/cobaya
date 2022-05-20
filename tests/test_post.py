import os
from copy import deepcopy
from scipy.stats import multivariate_normal
from getdist.mcsamples import loadMCSamples, MCSamplesFromCobaya
import numpy as np
import pytest

from cobaya.run import run
from cobaya.post import post, OutputOptions
from cobaya.typing import ParamsDict, InputDict
from cobaya.conventions import separator_files
from cobaya import mpi

pytestmark = pytest.mark.mpi

_post_ = separator_files + "post" + separator_files

mean = np.array([0, 0])
sigma = 0.5
cov = np.array([[sigma ** 2, 0], [0, sigma ** 2]])
sampled = {"mean": mean, "cov": cov}
target = {"mean": mean + np.array([sigma / 2, 0]), "cov": cov}


def allclose(a, b):
    return np.allclose(a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64))


def sampled_pdf(a, b):
    return multivariate_normal.logpdf([a, b], mean=sampled["mean"], cov=sampled["cov"])


def target_pdf(a, b, c=0):
    logp = multivariate_normal.logpdf([a, b], mean=target["mean"], cov=target["cov"])
    derived = {"cprime": c}
    return logp, derived


def target_pdf_prior(a, b):
    return target_pdf(a, b, c=0)[0]


_range = {"min": -2, "max": 2}
ref_pdf = {"dist": "norm", "loc": 0, "scale": 0.1}
info_params: ParamsDict = dict([
    ("a", {"prior": _range, "ref": ref_pdf, "proposal": sigma}),
    ("b", {"prior": _range, "ref": ref_pdf, "proposal": sigma}),
    ("a_plus_b", {"derived": lambda a, b: a + b})])

info_sampler = {"mcmc": {"Rminus1_stop": 0.5, "Rminus1_cl_stop": 0.5, "seed": 1}}
info_sampler_dummy = {"evaluate": {"N": 10}}


def _get_targets(mcsamples_in):
    loglikes = []
    for a, b in zip(mcsamples_in['a'], mcsamples_in['b']):
        loglikes.append(-target_pdf_prior(a, b) + sampled_pdf(a, b))
    mcsamples_in.reweightAddingLogLikes(loglikes)
    target_mean = mcsamples_in.mean(["a", "b"])
    target_cov = mcsamples_in.getCovMat().matrix
    return target_mean, target_cov


@mpi.sync_errors
def test_post_prior(tmpdir):
    # Generate original chain
    info: InputDict = {
        "output": os.path.join(tmpdir, "gaussian"), "force": True,
        "params": info_params, "sampler": info_sampler,
        "likelihood": {"one": None}, "prior": {"gaussian": sampled_pdf}}
    info_post: InputDict = {
        "output": info["output"], "force": True,
        "post": {"suffix": "foo", 'skip': 0.1,
                 "remove": {"prior": {"gaussian": None}},
                 "add": {"prior": {"target": target_pdf_prior}}}}
    _, sampler = run(info)
    if mpi.is_main_process():
        mcsamples_in = loadMCSamples(info["output"], settings={'ignore_rows': 0.1})
        target_mean, target_cov = mpi.share(_get_targets(mcsamples_in))
    else:
        target_mean, target_cov = mpi.share()

    for mem in [False, True]:
        post(info_post, sample=sampler.products()["sample"] if mem else None)
        # Load with GetDist and compare
        if mpi.is_main_process():
            mcsamples = loadMCSamples(
                info_post["output"] + _post_ + info_post["post"]["suffix"])
            new_mean = mcsamples.mean(["a", "b"])
            new_cov = mcsamples.getCovMat().matrix
            mpi.share((new_mean, new_cov))
        else:
            new_mean, new_cov = mpi.share()
        assert np.allclose(new_mean, target_mean)
        assert np.allclose(new_cov, target_cov)


def test_post_likelihood():
    """
    Swaps likelihood "gaussian" for "target".

    It also tests aggregated chi2's by removing and adding a likelihood to an existing
    type.
    """
    # Generate original chain
    orig_interval = OutputOptions.output_inteveral_s
    try:
        OutputOptions.output_inteveral_s = 0
        info_params_local = deepcopy(info_params)
        info_params_local["dummy"] = 0
        dummy_loglike_add = 0.1
        dummy_loglike_remove = 0.01
        info = {
            "output": None, "force": True,
            "params": info_params_local, "sampler": info_sampler,
            "likelihood": {
                "gaussian": {"external": sampled_pdf, "type": "A"},
                "dummy": {"external": lambda dummy: 1, "type": "BB"},
                "dummy_remove": {"external": lambda dummy: dummy_loglike_add,
                                 "type": "BB"}}}
        info_out, sampler = run(info)
        samples_in = mpi.gather(sampler.products()["sample"])
        if mpi.is_main_process():
            mcsamples_in = MCSamplesFromCobaya(info_out, samples_in)
        else:
            mcsamples_in = None

        info_out.update({
            "post": {"suffix": "foo",
                     "remove": {"likelihood": {
                         "gaussian": None, "dummy_remove": None}},
                     "add": {"likelihood": {
                         "target": {
                             "external": target_pdf, "type": "A",
                             "output_params": ["cprime"]},
                         "dummy_add": {
                             "external": lambda dummy: dummy_loglike_remove,
                             "type": "BB"}}}}})
        info_post_out, products_post = post(info_out, sampler.products()["sample"])
        samples = mpi.gather(products_post["sample"])

        # Load with GetDist and compare
        if mcsamples_in:
            target_mean, target_cov = mpi.share(_get_targets(mcsamples_in))

            mcsamples = MCSamplesFromCobaya(info_post_out, samples, name_tag="sample")
            new_mean = mcsamples.mean(["a", "b"])
            new_cov = mcsamples.getCovMat().matrix
            mpi.share((new_mean, new_cov))
        else:
            target_mean, target_cov = mpi.share()
            new_mean, new_cov = mpi.share()
        assert np.allclose(new_mean, target_mean)
        assert np.allclose(new_cov, target_cov)
        assert allclose(products_post["sample"]["chi2__A"],
                        products_post["sample"]["chi2__target"])
        assert allclose(products_post["sample"]["chi2__BB"],
                        products_post["sample"]["chi2__dummy"] +
                        products_post["sample"]["chi2__dummy_add"])
    finally:
        OutputOptions.output_inteveral_s = orig_interval


def test_post_params():
    # Tests:
    # - added simple dynamical derived parameter "a+b"
    # - added dynamical derived parameter that depends on *new* chi2__target
    # - added new fixed input "c" + new derived-from-external-function "cprime"
    # Generate original chain
    info = {
        "params": info_params, "sampler": info_sampler_dummy,
        "likelihood": {"gaussian": sampled_pdf}}
    updated_info_gaussian, sampler_gaussian = run(info)
    products_gaussian = sampler_gaussian.products()
    info_post = {
        "post": {"suffix": "foo",
                 "remove": {"params": {"a_plus_b": None}},
                 "add": {
                     "likelihood": {
                         "target": {"external": target_pdf, "output_params": ["cprime"]}},
                     "params": {
                         "c": 1.234,
                         "a_minus_b": {"derived": "lambda a,b: a-b"},
                         "my_chi2__target": {
                             "derived": "lambda chi2__target: chi2__target"},
                         "cprime": None}}}}
    info_post.update(updated_info_gaussian)
    _, products = post(info_post, products_gaussian["sample"])
    # Compare parameters
    assert allclose(products["sample"]["a"] - products["sample"]["b"],
                    products["sample"]["a_minus_b"])
    assert np.allclose(products["sample"]["cprime"].to_numpy(dtype=np.float64),
                       info_post["post"]["add"]["params"]["c"])
    assert allclose(products["sample"]["my_chi2__target"],
                    products["sample"]["chi2__target"])
