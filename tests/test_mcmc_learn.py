# Samples from a random Gaussian likelihood using the MCMC sampler.

from __future__ import division
import pytest
import numpy as np
from collections import OrderedDict as odict
from mpi4py import MPI

from cobaya.conventions import _likelihood, _params, _sampler
from cobaya.likelihoods.gaussian import random_mean, random_cov

# Kullback-Leibler divergence between 2 gaussians
def KL_norm(m1=None, S1=np.array([]), m2=None, S2=np.array([])):
    assert S1.shape[0], "Must give at least S1"
    dim = S1.shape[0]
    if m1 is None:
        m1 = np.zeros(dim)
    if not S2.shape[0]:
        S2 = np.identity(dim)
    if m2 is None:
        m2 = np.zeros(dim)
    S2inv = np.linalg.inv(S2)
    KL = 0.5*(np.trace(S2inv.dot(S1)) + (m1-m2).dot(S2inv).dot(m1-m2)
              -dim+np.log(np.linalg.det(S2)/np.linalg.det(S1)))
    return KL


# Prepares the likelihood and prior parts of the info
def info_gaussian(ranges, n_modes=1, mock_prefix=""):
    """MPI-aware: only draws the random stuff once!"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        cov = random_cov(ranges, n_modes=n_modes, O_std_min=0.05, O_std_max=0.1)
        # Make sure it stays away from the edges
        std = np.sqrt(cov.diagonal())
        factor = 3
        ranges_mean = [[l[0]+factor*s,l[1]-+factor*s] for l,s in zip(ranges,std)]
        # If this implies min>max, take the centre
        ranges_mean = [(l if l[0] <= l[1] else 2*[(l[0]+l[1])/2]) for l in ranges_mean]
        mean = random_mean(ranges_mean, n_modes=n_modes)
    elif rank != 0:
        mean, cov = None, None
    mean = comm.bcast(mean, root=0)
    cov  = comm.bcast(cov, root=0)
    dimension = len(ranges)
    info = {_likelihood: {"gaussian": {
        "mean": mean, "cov": cov, "mock_prefix": mock_prefix}}}
    info[_params] = odict(
        # sampled
        [[mock_prefix+"%d"%i,
          {"prior":{"min": ranges[i][0], "max": ranges[i][1]},
           "latex": r"\alpha_{%i}"%i}]
         for i in range(dimension)] +
        # derived
        [[mock_prefix+"derived_%d"%i,
          {"min": -3,"max": 3,"latex": r"\beta_{%i}"%i}] for i in range(dimension*n_modes)])
    return info


@pytest.mark.mpi
def test_gaussian_mcmc():
    # parameters
    dimension = 3
    n_modes = 1
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Info of likelihood and prior
    ranges = np.array([[0,1] for i in range(dimension)])
    info = info_gaussian(ranges=ranges, n_modes=n_modes, mock_prefix="a_")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print "Original mean of the gaussian mode:"
        print info["likelihood"]["gaussian"]["mean"]
        print "Original covmat of the gaussian mode:"
        print info["likelihood"]["gaussian"]["cov"]
    # Sample info (random covariance matrices for the proposal)
    S_proposal = []
    def check(sampler_instance):
        KL_proposer = KL_norm(S1=info["likelihood"]["gaussian"]["cov"],
                              S2=sampler_instance.proposer.get_covariance())
        KL_sample   = KL_norm(m1=info["likelihood"]["gaussian"]["mean"],
                              S1=info["likelihood"]["gaussian"]["cov"],
                              m2=sampler_instance.collection.mean(first=int(sampler_instance.n()/2)),
                              S2=sampler_instance.collection.cov (first=int(sampler_instance.n()/2)))
        print KL_proposer, KL_sample

    # Mcmc info
    if rank == 0:
        S0 = random_cov(ranges, n_modes=1, O_std_min=0.01, O_std_max=0.5)
    else:
        S0 = None
    S0 = comm.bcast(S0, root=0)
    # First KL distance
    print "*** 1st KL: ", KL_norm(S1=info["likelihood"]["gaussian"]["cov"], S2=S0)
    info[_sampler] = {"mcmc": {
        # Bad guess for covmat, so big burn in and max_tries
        "max_tries": 1000, "burn_in": 100,
        # Learn proposal
        "learn_proposal": True,
        # Callback to check KL divergence -- disabled in the automatic test
        "callback_function": check, "callback_every": 100,
        "covmat": S0, "covmat_params": info["params"].keys()[:dimension]
        }}
    # Run!!!
    info["debug"] = False
    info["debug_file"] = None
    # Delay to one chain to check that the MPI communication of the sampler is non-blocking
    #    if rank == 1:
    #        info["likelihood"]["gaussian"]["delay"] = 0.1
    from cobaya.run import run
    updated_info, products = run(info)
    # Done! --> Tests
    if rank == 0:
        import getdist as gd
        gdsamples = products["sample"].as_getdist_mcsamples(first=int(products["sample"].n()/2))
        cov_sample, mean_sample = gdsamples.getCov(), gdsamples.getMeans()
        print mean_sample
        KL_final = KL_norm(m1=info[_likelihood]["gaussian"]["mean"],
                           S1=info[_likelihood]["gaussian"]["cov"],
                           m2 = mean_sample[:dimension],
                           S2=cov_sample[:dimension,:dimension])
        np.set_printoptions(linewidth=np.inf)
        print "Likelihood covmat:  \n", info[_likelihood]["gaussian"]["cov"]
        print "Sample covmat:      \n", cov_sample[:dimension,:dimension]
        print "Sample covmat (std):\n", cov_sample[dimension:-dimension,dimension:-dimension]
        print "Final KL: ", KL_norm(S1=info["likelihood"]["gaussian"]["cov"], S2=cov_sample[:dimension,:dimension])
        assert KL_final <= 0.05, "KL not small enough. Got %g."%KL_final

if __name__ == "__main__":
    test_gaussian_mcmc()
