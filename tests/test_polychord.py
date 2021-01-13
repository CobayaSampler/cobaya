from flaky import flaky
import numpy as np

from cobaya.run import run
from .common_sampler import body_of_test, body_of_test_speeds
from cobaya.conventions import kinds, _params, partag, _output_prefix
from .conftest import install_test_wrapper


@flaky(max_runs=3, min_passes=1)
def test_polychord(packages_path, skip_not_installed, tmpdir):
    dimension = 3
    n_modes = 1
    info_sampler = {"polychord": {"nlive": 25 * dimension * n_modes}}
    body_of_test(dimension=dimension, n_modes=n_modes,
                 info_sampler=info_sampler, tmpdir=str(tmpdir),
                 packages_path=packages_path, skip_not_installed=skip_not_installed)


def test_polychord_resume(packages_path, skip_not_installed, tmpdir):
    """
    Tests correct resuming of a run, especially conserving the original blocking.

    To test preservation of the oversampling+blocking, we try to confuse the sampler by
    requesting speed measuring at resuming, and providing speeds very different from the
    real ones.
    """
    nlive = 10
    max_ndead = 2 * nlive
    def callback(sampler):
        global dead_points
        dead_points = sampler.dead[["a", "b"]].values.copy()
    info = {
        kinds.likelihood: {
            "A": {"external": "lambda a: stats.norm.logpdf(a)", "speed": 1},
            "B": {"external": "lambda b: stats.norm.logpdf(b)", "speed": 0.01}},
        _params: {
            "a": {partag.prior: {"min": 0, "max": 1}},
            "b": {partag.prior: {"min": 0, "max": 1}}},
        kinds.sampler: {
            "polychord": {
                "measure_speeds": True,
                "nlive": nlive,
                "max_ndead": max_ndead,
                "callback_function": callback,
            }},
        _output_prefix: str(tmpdir)}
    upd_info, sampler = install_test_wrapper(skip_not_installed, run, info)
    old_dead_points = dead_points.copy()
    info["resume"] = True
    upd_info, sampler = run(info)
    assert np.allclose(old_dead_points, dead_points)


@flaky(max_runs=5, min_passes=1)
def test_polychord_multimodal(packages_path, skip_not_installed, tmpdir):
    dimension = 2
    n_modes = 2
    info_sampler = {"polychord": {"nlive": 40 * dimension * n_modes}}
    body_of_test(dimension=dimension, n_modes=n_modes,
                 info_sampler=info_sampler, tmpdir=str(tmpdir),
                 packages_path=packages_path, skip_not_installed=skip_not_installed)


@flaky(max_runs=3, min_passes=1)
def test_polychord_speeds(packages_path, skip_not_installed):
    info_polychord = {"polychord": {"oversample_power": 1}}
    body_of_test_speeds(info_polychord, packages_path=packages_path,
                        skip_not_installed=skip_not_installed)


@flaky(max_runs=3, min_passes=1)
def test_polychord_speeds_manual(packages_path, skip_not_installed):
    info_polychord = {"polychord": {"oversample_power": 1}}
    body_of_test_speeds(info_polychord, manual_blocking=True,
                        packages_path=packages_path,
                        skip_not_installed=skip_not_installed)


@flaky(max_runs=3, min_passes=1)
def test_polychord_unphysical(packages_path, skip_not_installed):
    """
    Tests that the effect of unphysical regions is subtracted correctly.

    To do that, it integrates a normalised 2D Guassian likelihood over a uniform prior in
    the region (-bounds, +bound) x (-bound, bound), with a x>y cut.

    The correct evidence is int(pi*L) = 1/V int(L), which for a normalised prior, that is
    1/piVol * int(L) = 1/(2*bound)**2 / 2) * 0.5.

    To get that result, we need a run with likelihood off to get the correct prior
    normalisation, which is of course the prior *density* (1/(2*bound))**2 over the result
    of the run, which is 0.5.

    We then run with the full likelihood and, as usual, divide (subtract in log) by the
    normalisation factor of the prior: the result of the prior-only run.
    """
    bound = 10
    info = {
        "likelihood": {
            "gaussian":
            "lambda a_0, a_1: stats.multivariate_normal.logpdf([a_0, a_1], mean=[0,0])"},
        "prior": {"prior0": "lambda a_0, a_1: np.log(a_0 > a_1)"},
        "params": {
            "a_0": {"prior": {"min": -bound, "max": bound}},
            "a_1": {"prior": {"min": -bound, "max": bound}}},
        "sampler": {
            "polychord": {"nprior": "100nlive", "measure_speeds": False}}}
    # NB: we increase nprior wrt the default (25d=nlive) to get an accurate estimation
    #     of the unphysical region.
    info_like = info.pop("likelihood")
    info["likelihood"] = {"one": None}
    _, sampler_prior_only = install_test_wrapper(skip_not_installed, run, info)
    logZpi = sampler_prior_only.products()["logZ"]
    logZpistd = sampler_prior_only.products()["logZstd"]
    info["likelihood"] = info_like
    _, sampler_with_like = run(info)
    logZlike = sampler_with_like.products()["logZ"]
    logZlikestd = sampler_with_like.products()["logZstd"]
    logZ = logZlike - logZpi
    sigma = logZlikestd + logZpistd
    truth = 1/((2*bound)**2 / 2) * 0.5
    assert logZ - 2*sigma < np.log(truth) < logZ + 2*sigma
