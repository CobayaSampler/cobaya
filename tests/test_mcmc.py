from flaky import flaky
import numpy as np
import pytest

from cobaya.likelihoods.gaussian_mixture import random_cov
from cobaya.tools import KL_norm
from cobaya.likelihood import Likelihood
from cobaya.run import run
from cobaya import mpi
from .common_sampler import body_of_test, body_of_test_speeds

pytestmark = pytest.mark.mpi

# Max number of tries per test
max_runs = 3


@flaky(max_runs=max_runs, min_passes=1)
def test_mcmc(tmpdir, packages_path=None):
    dimension = 3
    # Random initial proposal
    S0 = mpi.share(
        random_cov(dimension * [[0, 1]], n_modes=1, O_std_min=0.01, O_std_max=0.5)
        if mpi.is_main_process() else None)
    info_sampler = {"mcmc": {
        # Bad guess for covmat, so big burn in and max_tries
        # TODO: why * dimension here?
        "max_tries": 1000 * dimension, "burn_in": 100 * dimension,
        # Learn proposal
        # "learn_proposal": True,  # default now!
        # Proposal
        "covmat": S0}}

    def check_gaussian(sampler_instance):
        KL_proposer = KL_norm(
            S1=sampler_instance.model.likelihood["gaussian_mixture"].covs[0],
            S2=sampler_instance.proposer.get_covariance())
        KL_sample = KL_norm(
            m1=sampler_instance.model.likelihood["gaussian_mixture"].means[0],
            S1=sampler_instance.model.likelihood["gaussian_mixture"].covs[0],
            m2=sampler_instance.collection.mean(
                first=int(sampler_instance.n() / 2)),
            S2=sampler_instance.collection.cov(
                first=int(sampler_instance.n() / 2)))
        print("KL proposer: %g ; KL sample: %g" % (KL_proposer, KL_sample))

    if mpi.rank() == 0:
        info_sampler["mcmc"].update({
            # Callback to check KL divergence -- disabled in the automatic test
            "callback_function": check_gaussian, "callback_every": 100})
    body_of_test(dimension=dimension, info_sampler=info_sampler, tmpdir=tmpdir)


@flaky(max_runs=max_runs, min_passes=1)
def test_mcmc_blocking():
    info_mcmc = {"mcmc": {"burn_in": 0, "learn_proposal": False}}
    body_of_test_speeds(info_mcmc)


@flaky(max_runs=max_runs, min_passes=1)
def test_mcmc_oversampling():
    info_mcmc = {"mcmc": {"burn_in": 0, "learn_proposal": False, "oversample_power": 1}}
    body_of_test_speeds(info_mcmc)


@flaky(max_runs=max_runs, min_passes=1)
def test_mcmc_oversampling_manual():
    # TODO - update ('oversample')
    info_mcmc = {"mcmc": {"burn_in": 0, "learn_proposal": False, "oversample": True}}
    body_of_test_speeds(info_mcmc, manual_blocking=True)


# The flaky test fails if likes or derived at chain points are not reproduced directly
# (dragging is somewhat delicate)
@flaky(max_runs=max_runs, min_passes=1,
       rerun_filter=(lambda err, *args: issubclass(err[0], AssertionError)))
def test_mcmc_dragging():
    info_mcmc = {"mcmc": {"burn_in": 0, "learn_proposal": False,
                          "drag": True, "oversample_power": 1}}
    body_of_test_speeds(info_mcmc)


def _make_gaussian_like(nparam):
    class LikeTest(Likelihood):
        params = {'x' + str(name): {'prior': {'min': -5, 'max': 5}, 'proposal': 1}
                  for name in range(nparam)}

        def calculate(self, state, want_derived=True, **params_values_dict):
            state["logp"] = -np.sum(np.array(list(params_values_dict.values())) ** 2 / 2)

    return LikeTest


def _test_overhead_timing(dim=15):
    # prints timing for simple Gaussian vanilla mcmc
    import pstats
    from cProfile import Profile
    from io import StringIO
    # noinspection PyUnresolvedReferences
    from cobaya.samplers.mcmc import proposal  # one-time numba compile out of profiling

    LikeTest = _make_gaussian_like(dim)
    info = {'likelihood': {'like': LikeTest}, 'debug': False, 'sampler': {
        'mcmc': {'max_samples': 1000, 'burn_in': 0, "learn_proposal": False,
                 "Rminus1_stop": 0.0001}}}
    prof = Profile()
    prof.enable()
    run(info)
    prof.disable()
    # prof.dump_stats("out.prof")  # to visualize with e.g. snakeviz
    s = StringIO()
    ps = pstats.Stats(prof, stream=s)
    print_n_calls = 10
    ps.strip_dirs()
    ps.sort_stats('time')
    ps.print_stats(print_n_calls)
    ps.sort_stats('cumtime')
    ps.print_stats(print_n_calls)
    print(s.getvalue())
