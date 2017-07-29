# Samples from a random Guassian likelihood using the MCMC and the PolyChord samplers.

import numpy as np
from collections import OrderedDict as odict

from cobaya.conventions import input_likelihood, input_params, input_sampler
from cobaya.likelihoods.gaussian import random_mean, random_cov

def info_gaussian(dimension, n_modes=1, mock_prefix=""):
    ranges = np.array([[0,1] for i in range(dimension)])
    info = {
        input_likelihood: {"gaussian": {
            "mean": random_mean(ranges, n_modes=n_modes),
            "cov":  random_cov(ranges, n_modes=n_modes, O_std_min=0.05, O_std_max=0.1),
            "mock_prefix": mock_prefix
        }}}
    info[input_params] = odict(
        # sampled
        [[mock_prefix+"%d"%i,
          {"prior": {"min":ranges[i][0], "max":ranges[i][1]},
           "proposal": np.sqrt(info[input_likelihood]["gaussian"]["cov"][i,i]),
           "ref": info[input_likelihood]["gaussian"]["mean"][i],
           "latex": r"\alpha_{%i}"%i}]
         for i in range(dimension)] +
        # derived
        [[mock_prefix+"derived_%d"%i,
          {"min":-3,"max":3,"latex":r"\beta_{%i}"%i}] for i in range(dimension*n_modes)])
    return info

def test_gaussian_mcmc():
    dimension = 3
    n_modes = 1
    info = info_gaussian(dimension=dimension, n_modes=n_modes, mock_prefix="a_")
    # Mcmc info
    info[input_sampler] = {"mcmc":{"max_samples": 10000, "max_tries": 1000}}
    # Save covariance matrix to help sampler
    import os
    import tempfile
    covmat_file_name = os.path.join(tempfile.gettempdir(),"covmat.dat")
    if n_modes == 1:
        np.savetxt(covmat_file_name, info["likelihood"]["gaussian"]["cov"],
                   header=" ".join(info["params"].keys()[:dimension]))
        info["sampler"]["mcmc"]["covmat"] = covmat_file_name
    # Run
    from cobaya.run import run
    updated_info, products = run(info)
    # Tests
    import getdist as gd
    gdsamples = products["sample"].as_getdist_mcsamples()
    np.set_printoptions(linewidth=np.inf)
    cov_sample = gdsamples.getCov()
    cov_likelihood = info[input_likelihood]["gaussian"]["cov"]
    print "Likelihood covmat:  \n", cov_likelihood
    print "Sample covmat:      \n", cov_sample[:dimension,:dimension]
    print "Sample covmat (std):\n", cov_sample[dimension:-dimension,dimension:-dimension]

    
def test_gaussian_polychord(polychord_path):
    dimension = 3
    n_modes = 1
    info = info_gaussian(dimension=dimension, n_modes=n_modes, mock_prefix="a_")
    # mcmc
    info[input_sampler] = {"mcmc":{"max_samples": 1000, "max_tries": 1000}}
    
    info[input_sampler] = {"polychord":{"path":polychord_path, "nlive":200}}
#    info[input_sampler] = {"evaluate":{}}
#    print "INFO: ", info
#    exit()
    from cobaya.run import run
    updated_info, products = run(info)
    import getdist as gd
    gdsamples = products["sample"].as_getdist_mcsamples()
    np.set_printoptions(linewidth=np.inf)
    cov_sample = gdsamples.getCov()
    cov_likelihood = info[input_likelihood]["gaussian"]["cov"]
    print "Likelihood covmat:  \n", cov_likelihood
    print "Sample covmat:      \n", cov_sample[:dimension,:dimension]
    print "Sample covmat (std):\n", cov_sample[dimension:-3,dimension:-3]

    
# DELETE ME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if __name__ == "__main__":
    test_gaussian_mcmc()
#    test_gaussian_polychord("/home/jesus/scratch/sampler/samplers/PolyChord")
