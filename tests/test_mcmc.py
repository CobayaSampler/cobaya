import logging
import os
import time

import numpy as np
import pytest
from flaky import flaky

from cobaya import InputDict, Likelihood, mpi, run
from cobaya.collection import SampleCollection
from cobaya.log import LoggedError, NoLogging
from cobaya.tools import KL_norm
from cobaya.typing import type_checking
from cobaya.yaml import yaml_dump_file, yaml_load, yaml_load_file

from .common_sampler import body_of_sampler_test, body_of_test_speeds

pytestmark = pytest.mark.mpi

# Max number of tries per test
max_runs = 3


@flaky(max_runs=max_runs, min_passes=1)
@pytest.mark.parametrize("temperature", (1, 2))
def test_mcmc(tmpdir, temperature, do_plots, packages_path=None):
    dimension = 3
    # Random initial proposal
    # cov = mpi.share(
    #    random_cov(dimension * [[0, 1]], O_std_min=0.01, O_std_max=0.5)
    #    if mpi.is_main_process() else None)
    cov = np.array(
        [
            [0.01853538, -0.02990048, 0.00046138],
            [-0.02990048, 0.14312571, -0.00441829],
            [0.00046138, -0.00441829, 0.00019141],
        ]
    )
    info_sampler = {
        "mcmc": {
            # Bad guess for covmat, so big burn in and max_tries
            "max_tries": 3000,
            "burn_in": 100 * dimension,
            # Proposal, relearn quickly as poor guess
            "covmat": cov,
            "learn_proposal_Rminus1_max": 30,
            # Sampling temperature, 1 and >1 to be tested
            "temperature": temperature,
        }
    }

    def check_gaussian(sampler_instance):
        if not len(sampler_instance.collection) or not len(
            sampler_instance.collection[int(sampler_instance.n() / 2) :]
        ):
            return
        proposer = KL_norm(
            S1=sampler_instance.model.likelihood["gaussian_mixture"].covs[0],
            S2=sampler_instance.proposer.get_covariance(),
        )
        sample = KL_norm(
            m1=sampler_instance.model.likelihood["gaussian_mixture"].means[0],
            S1=sampler_instance.model.likelihood["gaussian_mixture"].covs[0],
            m2=sampler_instance.collection.mean(first=int(sampler_instance.n() / 2)),
            S2=sampler_instance.collection.cov(first=int(sampler_instance.n() / 2)),
        )
        print(f"KL proposer: {proposer:g} ; KL sample: {sample:g}")

    if mpi.rank() == 0:
        info_sampler["mcmc"].update(
            {
                # Callback to check KL divergence -- disabled in the automatic test
                "callback_function": check_gaussian,
                "callback_every": 100,
            }
        )
    body_of_sampler_test(
        info_sampler,
        dimension=dimension,
        fixed=True,
        tmpdir=tmpdir,
        do_plots=do_plots,
        random_state=np.random.default_rng(1),
    )


yaml_drag = r"""
params:
  a:
    prior:
      min: -0.5
      max: 3
    proposal: 0.6
  b:
    prior:
      dist: norm
      loc: 0
      scale: 1
    ref: 0
    proposal: 0.6
sampler:
  mcmc:
   drag: True
   measure_speeds: False
   Rminus1_stop: 0.001
   Rminus1_cl_stop: 0.04
   seed: 1
"""


class GaussLike(Likelihood):
    speed = 100
    params = {"a": None}

    def calculate(self, state, want_derived=True, **params_values_dict):
        state["logp"] = -((params_values_dict["a"] - 0.2) ** 2) / 0.15 / 2


class GaussLike2(Likelihood):
    speed = 600
    params = {"a": None, "b": None}

    def logp(self, **params_values_dict):
        return (
            -((params_values_dict["a"] - 0.2) ** 2 + params_values_dict["b"] ** 2)
            / 0.2
            / 2
        )


@flaky(max_runs=max_runs, min_passes=1)
@mpi.sync_errors
@pytest.mark.parametrize("temperature", (1, 2))
def test_mcmc_drag_results(temperature):
    info: InputDict = yaml_load(yaml_drag)  # type: ignore
    info["likelihood"] = {"g1": {"external": GaussLike}, "g2": {"external": GaussLike2}}
    info["sampler"]["mcmc"]["temperature"] = temperature
    _updated_info, sampler = run(info)
    gdsample = sampler.samples(combined=True, skip_samples=0.2, to_getdist=True)
    if temperature != 1:
        gdsample.cool(temperature)
    assert abs(gdsample.mean("a") - 0.2) < 0.03
    assert abs(gdsample.mean("b")) < 0.03
    assert abs(gdsample.std("a") - 0.293) < 0.03
    assert abs(gdsample.std("b") - 0.4) < 0.03


yaml = r"""
likelihood:
  gaussian_mixture:
    means: [0.2, 0]
    covs: [[0.1, 0.05], [0.05,0.2]]

params:
  a:
    prior:
      min: -0.5
      max: 3
    latex: \alpha
    proposal: 0.3
  b:
    prior:
      dist: norm
      loc: 0
      scale: 1
    ref: 0
    latex: \beta
sampler:
  mcmc:
"""

logger = logging.getLogger("test")


@pytest.mark.mpionly
def test_mcmc_sync():
    info: InputDict = yaml_load(yaml)  # type: ignore
    logger.info("Test end synchronization")

    if mpi.rank() == 1:
        max_samples = 200
    else:
        max_samples = 600
    # simulate asynchronous ending sampling loop
    info["sampler"]["mcmc"] = {"max_samples": max_samples, "seed": 1}
    _updated_info, sampler = run(info, stop_at_error=True)
    assert len(sampler.products()["sample"]) == max_samples

    logger.info("Test error synchronization")
    if mpi.rank() == 0:
        info["sampler"]["mcmc"] = {"max_samples": "bad_val"}
        with NoLogging(logging.ERROR), pytest.raises(TypeError), type_checking(False):
            run(info)
    else:
        with pytest.raises(mpi.OtherProcessError):
            run(info)

    logger.info("Test one-process hang abort")

    aborted = False

    def test_abort():
        nonlocal aborted
        aborted = True

    # test error converted into MPI_ABORT after timeout
    with pytest.raises((LoggedError, mpi.OtherProcessError)), NoLogging(logging.CRITICAL):
        with mpi.ProcessState(
            "test", time_out_seconds=0.5, timeout_abort_proc=test_abort
        ):
            if mpi.rank() != 1:
                time.sleep(0.6)  # fake hang
            else:
                raise LoggedError(logger, "Expected test error")
    if mpi.rank() == 1:
        assert aborted


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
    info_mcmc = {"mcmc": {"burn_in": 0, "learn_proposal": False, "oversample_power": 0.4}}
    body_of_test_speeds(info_mcmc, manual_blocking=True)


# The flaky test fails if likes or derived at chain points are not reproduced directly
# (dragging is somewhat delicate)
@flaky(
    max_runs=max_runs,
    min_passes=1,
    rerun_filter=(lambda err, *args: issubclass(err[0], AssertionError)),
)
def test_mcmc_dragging():
    info_mcmc = {
        "mcmc": {
            "burn_in": 0,
            "learn_proposal": False,
            "drag": True,
            "oversample_power": 1,
        }
    }
    body_of_test_speeds(info_mcmc)


def _make_gaussian_like(nparam) -> type[Likelihood]:
    class LikeTest(Likelihood):
        params = {
            "x" + str(name): {"prior": {"min": -5, "max": 5}, "proposal": 1}
            for name in range(nparam)
        }

        def calculate(self, state, want_derived=True, **params_values_dict):
            state["logp"] = -np.sum(np.array(list(params_values_dict.values())) ** 2 / 2)

    # to make dynamically generated class picklable: (can't be loaded from yaml anyway)
    LikeTest.__module__ = "__main__"
    return LikeTest


def _test_overhead_timing(dim=15):
    # prints timing for simple Gaussian vanilla mcmc
    import pstats
    from cProfile import Profile
    from io import StringIO

    # one-time numba compile out of profiling
    from cobaya.functions import random_SO_N  # noqa: F401

    like_test = _make_gaussian_like(dim)
    info: InputDict = {
        "likelihood": {"like": like_test},
        "debug": False,
        "sampler": {
            "mcmc": {
                "max_samples": 1000,
                "burn_in": 0,
                "learn_proposal": False,
                "Rminus1_stop": 0.0001,
            }
        },
    }
    prof = Profile()
    prof.enable()
    run(info)
    prof.disable()
    # prof.dump_stats("out.prof")  # to visualize with e.g. snakeviz
    s = StringIO()
    ps = pstats.Stats(prof, stream=s)
    print_n_calls = 10
    ps.strip_dirs()
    ps.sort_stats("time")
    ps.print_stats(print_n_calls)
    ps.sort_stats("cumtime")
    ps.print_stats(print_n_calls)
    print(s.getvalue())


@pytest.mark.mpionly
def test_mcmc_sync_learn_at_end(monkeypatch):
    # A chain that stops (max_samples) just after becoming READY for a
    # learn+convergence check must not leave the others hanging in the (collective)
    # check when they catch up before it signals END.
    info: InputDict = yaml_load(yaml)  # type: ignore
    learn_every = 50
    max_samples = 2 * learn_every if mpi.rank() == 1 else 4 * learn_every + 10

    def like(a, b):
        if mpi.rank() == 0:
            time.sleep(0.002)  # make sure rank 1 gets to the last learn point first
        return -0.5 * (a**2 + b**2)

    info["likelihood"] = {"slow_on_rank_0": like}
    info["sampler"]["mcmc"] = {
        "max_samples": max_samples,
        "learn_every": learn_every,
        "burn_in": 0,
        "Rminus1_stop": 1e-6,
        "Rminus1_cl_stop": 1e-6,
        "seed": 1,
    }
    if mpi.rank() == 1:
        # Delay the END signal of rank 1, so that rank 0 catches up while rank 1 READY
        out_update = SampleCollection.out_update

        def delayed_out_update(self):
            if len(self) >= max_samples:
                time.sleep(3)
            out_update(self)

        monkeypatch.setattr(SampleCollection, "out_update", delayed_out_update)
    _updated_info, sampler = run(info, stop_at_error=True)
    assert len(sampler.products()["sample"]) == max_samples


@pytest.mark.mpionly
def test_mcmc_resume_learn_ready(tmpdir):
    # Chains that were READY (waiting for the rest at a learning point) when the run
    # was interrupted should be READY again on resume, so that the check happens
    # when the rest reach that learning point.
    info: InputDict = yaml_load(yaml)  # type: ignore
    learn_every = 50
    info["sampler"]["mcmc"] = {
        "max_samples": 3 * learn_every - 10,
        "learn_every": learn_every,
        "burn_in": 0,
        "Rminus1_stop": 1e-6,
        "Rminus1_cl_stop": 1e-6,
        "seed": 1,
    }
    info["output"] = os.path.join(tmpdir, "chain")
    _updated_info, sampler = run(info, force=True)
    # Simulate interruption with chain 1 READY past 2 * learn_every, chain 2 not yet
    if mpi.is_main_process():
        for i, n in enumerate([2 * learn_every + 7, 2 * learn_every - 3]):
            chain_file = os.path.join(tmpdir, f"chain.{i + 1}.txt")
            with open(chain_file, encoding="utf-8") as f:
                lines = f.readlines()
            with open(chain_file, "w", encoding="utf-8") as f:
                f.writelines(lines[: 1 + n])  # header
        checkpoint_file = sampler.checkpoint_filename()
        checkpoint = yaml_load_file(checkpoint_file)
        checkpoint["sampler"]["mcmc"].update(
            {"converged": False, "n_last_learn": [learn_every + 2, learn_every + 1]}
        )
        yaml_dump_file(checkpoint_file, checkpoint, error_if_exists=False)
    mpi.sync_processes()
    _updated_info, sampler = run(info, resume=True)
    if mpi.is_main_process():
        n_last_learn = yaml_load_file(sampler.checkpoint_filename())["sampler"]["mcmc"][
            "n_last_learn"
        ]
        # check done when chain 2 reached 2 * learn_every (without READY at resume,
        # chain 1 would not be ready again until 3 * learn_every > max_samples)
        assert n_last_learn[1] == 2 * learn_every
        assert n_last_learn[0] >= 2 * learn_every + 7
