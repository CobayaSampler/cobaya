# Minimization of a random Gaussian likelihood using the minimize sampler.

import numpy as np
from mpi4py import MPI
from scipy.stats import multivariate_normal
from flaky import flaky

from cobaya.conventions import kinds
from cobaya.likelihoods.gaussian_mixture import info_random_gaussian_mixture


@flaky(max_runs=3, min_passes=1)
def test_minimize_gaussian():
    # parameters
    dimension = 3
    n_modes = 1
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Info of likelihood and prior
    ranges = np.array([[0, 1] for _ in range(dimension)])
    prefix = "a_"
    info = info_random_gaussian_mixture(
        ranges=ranges, n_modes=n_modes, input_params_prefix=prefix, derived=True)
    mean = info[kinds.likelihood]["gaussian_mixture"]["means"][0]
    cov = info[kinds.likelihood]["gaussian_mixture"]["covs"][0]
    maxloglik = multivariate_normal.logpdf(mean, mean=mean, cov=cov)
    if rank == 0:
        print("Maximum of the gaussian mode to be found:")
        print(mean)
    info[kinds.sampler] = {"minimize": {"ignore_prior": True}}
    info["debug"] = False
    info["debug_file"] = None
    #    info["output_prefix"] = "./minigauss/"
    from cobaya.run import run
    updated_info, sampler = run(info)
    products = sampler.products()
    # Done! --> Tests
    if rank == 0:
        rel_error = abs(maxloglik - -products["minimum"]["minuslogpost"]) / abs(maxloglik)
        assert rel_error < 0.001
