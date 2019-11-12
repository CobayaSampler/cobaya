"""General test for samplers. Checks convergence, cluster detection, evidence."""

from __future__ import division, print_function, absolute_import
import numpy as np
from mpi4py import MPI
from random import shuffle, choice
from scipy.stats import multivariate_normal
from collections import OrderedDict as odict
from getdist.mcsamples import MCSamplesFromCobaya
from time import sleep
from itertools import chain

from cobaya.conventions import _likelihood, _sampler, _params, _output_prefix
from cobaya.conventions import _debug, _debug_file, _path_install
from cobaya.likelihoods.gaussian_mixture import info_random_gaussian_mixture
from cobaya.tools import KL_norm
from cobaya.run import run
from .common import process_modules_path

KL_tolerance = 0.05
logZ_nsigmas = 2
O_std_min = 0.01
O_std_max = 0.05
distance_factor = 4


def body_of_test(dimension=1, n_modes=1, info_sampler={}, tmpdir="", modules=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Info of likelihood and prior
    ranges = np.array([[-1, 1] for _ in range(dimension)])
    while True:
        info = info_random_gaussian_mixture(
            ranges=ranges, n_modes=n_modes, input_params_prefix="a_",
            O_std_min=O_std_min, O_std_max=O_std_max, derived=True)
        if n_modes == 1:
            break
        means = info["likelihood"]["gaussian_mixture"]["means"]
        distances = chain(*[[np.linalg.norm(m1 - m2) for m2 in means[i + 1:]]
                            for i, m1 in enumerate(means)])
        if min(distances) >= distance_factor * O_std_max:
            break
    if rank == 0:
        print("Original mean of the gaussian mode:")
        print(info["likelihood"]["gaussian_mixture"]["means"])
        print("Original covmat of the gaussian mode:")
        print(info["likelihood"]["gaussian_mixture"]["covs"])
    info[_sampler] = info_sampler
    if list(info_sampler.keys())[0] == "mcmc":
        if "covmat" in info_sampler["mcmc"]:
            info[_sampler]["mcmc"]["covmat_params"] = (
                list(info["params"].keys())[:dimension])
    info[_debug] = False
    info[_debug_file] = None
    info[_output_prefix] = getattr(tmpdir, "realpath()", lambda: tmpdir)()
    if modules:
        info[_path_install] = process_modules_path(modules)
    # Delay to one chain to check that MPI communication of the sampler is non-blocking
    #    if rank == 1:
    #        info["likelihood"]["gaussian_mixture"]["delay"] = 0.1
    updated_info, products = run(info)
    # Done! --> Tests
    if rank == 0:
        if list(info_sampler.keys())[0] == "mcmc":
            ignore_rows = 0.5
        else:
            ignore_rows = 0
        results = MCSamplesFromCobaya(updated_info, products["sample"],
                                      ignore_rows=ignore_rows, name_tag="sample")
        clusters = None
        if "clusters" in products:
            clusters = [MCSamplesFromCobaya(
                updated_info, products["clusters"][i]["sample"],
                name_tag="cluster %d" % (i + 1))
                for i in products["clusters"]]
        # Plots!
        try:
            import getdist.plots as gdplots
            from getdist.gaussian_mixtures import MixtureND
            mixture = MixtureND(
                info[_likelihood]["gaussian_mixture"]["means"],
                info[_likelihood]["gaussian_mixture"]["covs"],
                names=[p for p in info[_params] if "deriv" not in p], label="truth")
            g = gdplots.getSubplotPlotter()
            to_plot = [mixture, results]
            if clusters:
                to_plot = to_plot + clusters
            g.triangle_plot(to_plot, )
            g.export("test.png")
        except:
            print("Plotting failed!")
        # 1st test: KL divergence
        if n_modes == 1:
            cov_sample, mean_sample = results.getCov(), results.getMeans()
            KL_final = KL_norm(m1=info[_likelihood]["gaussian_mixture"]["means"][0],
                               S1=info[_likelihood]["gaussian_mixture"]["covs"][0],
                               m2=mean_sample[:dimension],
                               S2=cov_sample[:dimension, :dimension])
            print("Final KL: ", KL_final)
            assert KL_final <= KL_tolerance
        # 2nd test: clusters
        else:
            if "clusters" in products:
                assert len(products["clusters"].keys()) >= n_modes, (
                    "Not all clusters detected!")
                for i, c2 in enumerate(clusters):
                    cov_c2, mean_c2 = c2.getCov(), c2.getMeans()
                    KLs = [KL_norm(m1=info[_likelihood]["gaussian_mixture"]["means"][i_c1],
                                   S1=info[_likelihood]["gaussian_mixture"]["covs"][i_c1],
                                   m2=mean_c2[:dimension],
                                   S2=cov_c2[:dimension, :dimension])
                           for i_c1 in range(n_modes)]
                    extra_tol = 4 * n_modes if n_modes > 1 else 1
                    print("Final KL for cluster %d: %g", i, min(KLs))
                    assert min(KLs) <= KL_tolerance * extra_tol
            else:
                assert 0, "Could not check sample convergence: multimodal but no clusters"
        # 3rd test: Evidence
        if "logZ" in products:
            logZprior = sum(np.log(ranges[:, 1] - ranges[:, 0]))
            assert (products["logZ"] - logZ_nsigmas * products["logZstd"] < -logZprior <
                    products["logZ"] + logZ_nsigmas * products["logZstd"])


def body_of_test_speeds(info_sampler={}, manual_blocking=False, modules=None):
    # Generate 2 3-dimensional gaussians
    dim = 3
    speed1, speed2 = 5, 20
    ranges = [[i, i + 1] for i in range(2 * dim)]
    prefix = "a_"
    mean1, cov1 = [info_random_gaussian_mixture(
        ranges=[ranges[i] for i in range(dim)], n_modes=1, input_params_prefix=prefix,
        O_std_min=0.01, O_std_max=0.2, derived=True)
                   [_likelihood]["gaussian_mixture"][p][0] for p in ["means", "covs"]]
    mean2, cov2 = [info_random_gaussian_mixture(
        ranges=[ranges[i] for i in range(dim, 2 * dim)], n_modes=1,
        input_params_prefix=prefix, O_std_min=0.01, O_std_max=0.2, derived=True)
                   [_likelihood]["gaussian_mixture"][p][0] for p in ["means", "covs"]]
    global n1, n2
    n1, n2 = 0, 0
    # PolyChord measures its own speeds, so we need to "sleep"
    sleep_unit = 1 / 50
    sampler = list(info_sampler.keys())[0]

    def like1(a_0, a_1, a_2, _derived=["sum_like1"]):
        if sampler == "polychord":
            sleep(1 / speed1 * sleep_unit)
        global n1
        n1 += 1
        if _derived is not None:
            _derived["sum_like1"] = a_0 + a_1 + a_2
        return multivariate_normal.logpdf([a_0, a_1, a_2], mean=mean1, cov=cov1)

    def like2(a_3, a_4, a_5, _derived=["sum_like2"]):
        if sampler == "polychord":
            sleep(1 / speed2 * sleep_unit)
        global n2
        n2 += 1
        if _derived is not None:
            _derived["sum_like2"] = a_3 + a_4 + a_5
        return multivariate_normal.logpdf([a_3, a_4, a_5], mean=mean2, cov=cov2)

    # Rearrange parameter in arbitrary order
    perm = list(range(2 * dim))
    shuffle(perm)
    # Create info
    info = {"params":
                odict([ [prefix + "%d" % i, {"prior": dict(zip(["min", "max"], ranges[i]))}]
                          for i in perm] + [["sum_like1", None], ["sum_like2", None]]),
            "likelihood": {"like1": {"external": like1, "speed": speed1},
                           "like2": {"external": like2, "speed": speed2}}}
    info["sampler"] = info_sampler
    if manual_blocking:
        info["sampler"][sampler]["blocking"] = [
            [speed1, ["a_0", "a_1", "a_2"]],
            [speed2, ["a_3", "a_4", "a_5"]]]
    print("Parameter order:", list(info["params"]))
    # info["debug"] = True
    info["modules"] = modules
    # Adjust number of samples
    n_cycles_all_params = 10
    if sampler == "mcmc":
        info["sampler"][sampler]["burn_in"] = 0
        info["sampler"][sampler]["max_samples"] = n_cycles_all_params * 10 * dim
        # Force mixing of blocks:
        info["sampler"][sampler]["covmat_params"] = list(info["params"])
        info["sampler"][sampler]["covmat"] = 1 / 10000 * np.eye(len(info["params"]))
        i_1st, i_2nd = map(lambda x: info["sampler"][sampler]["covmat_params"].index(x),
                           [prefix + "0", prefix + "%d" % dim])
        info["sampler"][sampler]["covmat"][i_1st, i_2nd] = 1 / 100000
        info["sampler"][sampler]["covmat"][i_2nd, i_1st] = 1 / 100000
    elif sampler == "polychord":
        info["sampler"][sampler]["nlive"] = 2 * dim
        info["sampler"][sampler]["max_ndead"] = n_cycles_all_params * dim
    else:
        assert 0, "Unknown sampler for this test."
    updated_info, products = run(info)
    # Done! --> Tests
    if sampler == "polychord":
        tolerance = 0.2
        assert abs((n2 - n1) / n1 / (speed2 / speed1) - 1) < tolerance, (
                "#evaluations off: %g > %g" % (
            abs((n2 - n1) / n1 / (speed2 / speed1) - 1), tolerance))
    # For MCMC tests, notice that there is a certain tolerance to be allowed for,
    # since for every proposed step the BlockedProposer cycles once, but the likelihood
    # may is not evaluated if the proposed point falls outside the prior bounds
    elif sampler == "mcmc" and info["sampler"][sampler].get("drag"):
        assert abs((n2 - n1) / n1 / (speed2 / speed1) - 1) < 0.1
    elif sampler == "mcmc" and info["sampler"][sampler].get("oversample"):
        # Testing oversampling: number of evaluations per param * oversampling factor
        assert abs((n2 - n1) * dim / (n1 * dim) / (speed2 / speed1) - 1) < 0.1
    elif sampler == "mcmc":
        # Testing just correct blocking: same number of evaluations per param
        assert abs((n2 - n1) * dim / (n1 * dim) - 1) < 0.1
    # Finally, test some points of the chain to reproduce the correct likes and derived
    # These are not AssertionError's to override the flakyness of the test
    for _ in range(10):
        i = choice(list(range(products["sample"].n())))
        chi2_1_chain = -0.5 * products["sample"]["chi2__like1"][i]
        chi2_1_good = like1(
            _derived=None, **{p: products["sample"][p][i] for p in ["a_0", "a_1", "a_2"]})
        chi2_2_chain = -0.5 * products["sample"]["chi2__like2"][i]
        chi2_2_good = like2(
            _derived=None, **{p: products["sample"][p][i] for p in ["a_3", "a_4", "a_5"]})
        if not np.allclose([chi2_1_chain, chi2_2_chain], [chi2_1_good, chi2_2_good]):
            raise ValueError(
                "Likelihoods not reproduced correctly. "
                "Chain has %r but should be %r. " % (
                    [chi2_1_chain, chi2_2_chain], [chi2_1_good, chi2_2_good]) +
                "Full chain point: %r" % products["sample"][i])
        derived_chain = products["sample"][["sum_like1", "sum_like2"]].values[i]
        derived_good = np.array([
            sum(products["sample"][["a_0", "a_1", "a_2"]].values[i]),
            sum(products["sample"][["a_3", "a_4", "a_5"]].values[i])])
        if not np.allclose(derived_chain, derived_good):
            raise ValueError(
                "Derived params not reproduced correctly. "
                "Chain has %r but should be %r. " % (derived_chain, derived_good) +
                "Full chain point:\n%r" % products["sample"][i])
    print(products["sample"])
