from __future__ import division, print_function

from mpi4py import MPI
from flaky import flaky

from cobaya.likelihoods.gaussian import random_cov
from cobaya.tools import KL_norm

from common_sampler import body_of_test, body_of_test_speeds


### import pytest
### @pytest.mark.mpi


@flaky(max_runs=3, min_passes=1)
def test_mcmc(tmpdir, modules=None):
    dimension = 3
    # Random initial proposal
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        S0 = random_cov(dimension * [[0, 1]], n_modes=1, O_std_min=0.01, O_std_max=0.5)
    else:
        S0 = None
    S0 = comm.bcast(S0, root=0)
    info_sampler = {"mcmc": {
        # Bad guess for covmat, so big burn in and max_tries
        "max_tries": 1000 * dimension, "burn_in": 100 * dimension,
        # Learn proposal
        # "learn_proposal": True,  # default now!
        # Proposal
        "covmat": S0, }}

    def check_gaussian(sampler_instance):
        KL_proposer = KL_norm(S1=sampler_instance.model.likelihood["gaussian"].cov[0],
                              S2=sampler_instance.proposer.get_covariance())
        KL_sample = KL_norm(m1=sampler_instance.model.likelihood["gaussian"].mean[0],
                            S1=sampler_instance.model.likelihood["gaussian"].cov[0],
                            m2=sampler_instance.collection.mean(
                                first=int(sampler_instance.n() / 2)),
                            S2=sampler_instance.collection.cov(
                                first=int(sampler_instance.n() / 2)))
        print("KL proposer: %g ; KL sample: %g" % (KL_proposer, KL_sample))

    if rank == 0:
        info_sampler["mcmc"].update({
            # Callback to check KL divergence -- disabled in the automatic test
            "callback_function": check_gaussian, "callback_every": 100})
    body_of_test(
        dimension=dimension, n_modes=1, info_sampler=info_sampler, tmpdir=str(tmpdir))


@flaky(max_runs=3, min_passes=1)
def test_mcmc_blocking():
    info_mcmc = {"mcmc": {"burn_in": 0, "learn_proposal": False}}
    body_of_test_speeds(info_mcmc)


@flaky(max_runs=3, min_passes=1)
def test_mcmc_oversampling():
    info_mcmc = {"mcmc": {"burn_in": 0, "learn_proposal": False, "oversample": True}}
    body_of_test_speeds(info_mcmc)


# The flaky test fails if likes or derived at chain points are not reproduced directly
# (dragging is somewhat delicate)
@flaky(max_runs=3, min_passes=1,
       rerun_filter=(lambda err, *args: issubclass(err[0], AssertionError)))
def test_mcmc_dragging():
    info_mcmc = {"mcmc": {"burn_in": 0, "learn_proposal": False, "drag": True,
                          # For tests
                          "drag_limits": [None, None]}}
    body_of_test_speeds(info_mcmc)
