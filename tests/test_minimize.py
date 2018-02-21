# Minimization of a random Gaussian likelihood using the minimize sampler.

from __future__ import print_function
from __future__ import division

import numpy as np
from mpi4py import MPI

from cobaya.conventions import _likelihood, _sampler
from cobaya.likelihoods.gaussian import info_random_gaussian


def test_minimize_gaussian():
    # parameters
    dimension = 3
    n_modes = 1
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Info of likelihood and prior
    ranges = np.array([[0,1] for i in range(dimension)])
    info = info_random_gaussian(ranges=ranges, n_modes=n_modes, prefix="a_")
    if rank == 0:
        print("Maximim of the gaussian mode to be found:")
        print(info["likelihood"]["gaussian"]["mean"])
    info[_sampler] = {"minimize": {"tol": 1e-4, "maxiter":1e4, "ignore_prior": True}}
    info["debug"] = False
    info["debug_file"] = None
    info["output_prefix"] = "./minigauss/"
    from cobaya.run import run
    updated_info, products = run(info)
    print(products["maximum"])
