# Minimization of a random Gaussian likelihood using the minimize sampler.

from __future__ import print_function
from __future__ import division

import numpy as np
from mpi4py import MPI
from scipy.stats import multivariate_normal
from flaky import flaky

from cobaya.conventions import _likelihood, _sampler
from cobaya.likelihoods.gaussian import info_random_gaussian


@flaky(max_runs=3, min_passes=1)
def test_minimize_gaussian():
    # parameters
    dimension = 3
    n_modes = 1
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Info of likelihood and prior
    ranges = np.array([[0, 1] for i in range(dimension)])
    prefix = "a_"
    info = info_random_gaussian(ranges=ranges, n_modes=n_modes, prefix=prefix)
    mean = info[_likelihood]["gaussian"]["mean"][0]
    cov = info[_likelihood]["gaussian"]["cov"][0]
    maxloglik = multivariate_normal.logpdf(mean, mean=mean, cov=cov)
    if rank == 0:
        print("Maximim of the gaussian mode to be found:")
        print(mean)
    info[_sampler] = {"minimize": {"ignore_prior": True}}
    info["debug"] = False
    info["debug_file"] = None
    #    info["output_prefix"] = "./minigauss/"
    from cobaya.run import run
    updated_info, products = run(info)
    # Done! --> Tests
    if rank == 0:
        rel_error = abs(maxloglik - -products["maximum"]["minuslogpost"]) / abs(maxloglik)
        assert rel_error < 10 * updated_info[_sampler]["minimize"]["tol"]
