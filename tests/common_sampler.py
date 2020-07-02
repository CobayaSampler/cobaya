"""General test for samplers. Checks convergence, cluster detection, evidence."""

import numpy as np
from mpi4py import MPI
from random import shuffle, choice
from scipy.stats import multivariate_normal
from getdist.mcsamples import MCSamplesFromCobaya
from itertools import chain

from cobaya.conventions import kinds, _output_prefix, empty_dict
from cobaya.conventions import _debug, _debug_file, _packages_path, partag
from cobaya.likelihoods.gaussian_mixture import info_random_gaussian_mixture
from cobaya.tools import KL_norm
from cobaya.run import run
from .common import process_packages_path, is_travis
from .conftest import install_test_wrapper

KL_tolerance = 0.05
logZ_nsigmas = 2
O_std_min = 0.01
O_std_max = 0.05
distance_factor = 4


def body_of_test(dimension=1, n_modes=1, info_sampler=empty_dict, tmpdir="",
                 packages_path=None, skip_not_installed=False):
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
    info[kinds.sampler] = info_sampler
    sampler_name = list(info_sampler)[0]
    if sampler_name == "mcmc":
        if "covmat" in info_sampler["mcmc"]:
            info[kinds.sampler]["mcmc"]["covmat_params"] = (
                list(info["params"])[:dimension])
    info[_debug] = False
    info[_debug_file] = None
    # TODO: this looks weird/bug:?
    info[_output_prefix] = getattr(tmpdir, "realpath()", lambda: tmpdir)()
    if packages_path:
        info[_packages_path] = process_packages_path(packages_path)
    # Delay to one chain to check that MPI communication of the sampler is non-blocking
    #    if rank == 1:
    #        info["likelihood"]["gaussian_mixture"]["delay"] = 0.1
    updated_info, sampler = install_test_wrapper(skip_not_installed, run, info)
    products = sampler.products()
    # Done! --> Tests
    if rank == 0:
        if sampler_name == "mcmc":
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
            if is_travis():
                raise ValueError
            import getdist.plots as gdplots
            from getdist.gaussian_mixtures import MixtureND
            sampled_params = [
                p for p, v in info["params"].items() if partag.prior not in v]
            mixture = MixtureND(
                info[kinds.likelihood]["gaussian_mixture"]["means"],
                info[kinds.likelihood]["gaussian_mixture"]["covs"],
                names=sampled_params, label="truth")
            g = gdplots.getSubplotPlotter()
            to_plot = [mixture, results]
            if clusters:
                to_plot = to_plot + clusters
            g.triangle_plot(to_plot, params=sampled_params)
            g.export("test.png")
        except:
            print("Plotting failed!")
        # 1st test: KL divergence
        if n_modes == 1:
            cov_sample, mean_sample = results.getCov(), results.getMeans()
            KL_final = KL_norm(m1=info[kinds.likelihood]["gaussian_mixture"]["means"][0],
                               S1=info[kinds.likelihood]["gaussian_mixture"]["covs"][0],
                               m2=mean_sample[:dimension],
                               S2=cov_sample[:dimension, :dimension])
            print("Final KL: ", KL_final)
            assert KL_final <= KL_tolerance
        # 2nd test: clusters
        else:
            if "clusters" in products:
                assert len(products["clusters"]) >= n_modes, (
                    "Not all clusters detected!")
                for i, c2 in enumerate(clusters):
                    cov_c2, mean_c2 = c2.getCov(), c2.getMeans()
                    KLs = [
                        KL_norm(
                            m1=info[kinds.likelihood]["gaussian_mixture"]["means"][i_c1],
                            S1=info[kinds.likelihood]["gaussian_mixture"]["covs"][i_c1],
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


def body_of_test_speeds(info_sampler=empty_dict, manual_blocking=False,
                        packages_path=None, skip_not_installed=False):
    # #dimensions and speed ratio mutually prime (e.g. 2,3,5)
    dim0, dim1 = 5, 2
    speed0, speed1 = 1, 10
    ranges = [[i, i + 1] for i in range(dim0 + dim1)]
    prefix = "a_"
    params0, params1 = (lambda x: (x[:dim0], x[dim0:]))(
        [prefix + str(d) for d in range(dim0 + dim1)])
    derived0, derived1 = "sum_like0", "sum_like1"
    mean0, cov0 = [info_random_gaussian_mixture(
        ranges=[ranges[i] for i in range(dim0)], n_modes=1, input_params_prefix=prefix,
        O_std_min=0.01, O_std_max=0.2, derived=True)
                   [kinds.likelihood]["gaussian_mixture"][p][0] for p in
                   ["means", "covs"]]
    mean1, cov1 = [info_random_gaussian_mixture(
        ranges=[ranges[i] for i in range(dim0, dim0 + dim1)], n_modes=1,
        input_params_prefix=prefix, O_std_min=0.01, O_std_max=0.2, derived=True)
                   [kinds.likelihood]["gaussian_mixture"][p][0] for p in
                   ["means", "covs"]]
    n_evals = [0, 0]

    def like0(**kwargs):
        n_evals[0] += 1
        input_params = [kwargs[p] for p in params0]
        derived = {derived0: sum(input_params)}
        return multivariate_normal.logpdf(input_params, mean=mean0, cov=cov0), derived

    def like1(**kwargs):
        n_evals[1] += 1
        input_params = [kwargs[p] for p in params1]
        derived = {derived1: sum(input_params)}
        return multivariate_normal.logpdf(input_params, mean=mean1, cov=cov1), derived

    # Rearrange parameter in arbitrary order
    perm = list(range(dim0 + dim1))
    shuffle(perm)
    # Create info
    info = {"params": dict(
        {prefix + "%d" % i: {"prior": dict(zip(["min", "max"], ranges[i]))}
         for i in perm}, sum_like0=None, sum_like1=None),
        "likelihood": {"like0": {"external": like0, "speed": speed0,
                                 "input_params": params0, "output_params": derived0},
                       "like1": {"external": like1, "speed": speed1,
                                 "input_params": params1, "output_params": derived1}},
        "sampler": info_sampler}
    sampler_name = list(info_sampler)[0]
    if manual_blocking:
        over0, over1 = speed0, speed1
        info["sampler"][sampler_name]["blocking"] = [
            [over0, params0],
            [over1, params1]]
    print("Parameter order:", list(info["params"]))
    # info["debug"] = True
    info["packages_path"] = packages_path
    # Adjust number of samples
    n_cycles_all_params = 10
    if sampler_name == "mcmc":
        info["sampler"][sampler_name]["measure_speeds"] = False
        info["sampler"][sampler_name]["burn_in"] = 0
        info["sampler"][sampler_name]["max_samples"] = n_cycles_all_params * 10 * (
                dim0 + dim1)
        # Force mixing of blocks:
        info["sampler"][sampler_name]["covmat_params"] = list(info["params"])
        info["sampler"][sampler_name]["covmat"] = 1 / 10000 * np.eye(len(info["params"]))
        i_0th, i_1st = map(
            lambda x: info["sampler"][sampler_name]["covmat_params"].index(x),
            [prefix + "0", prefix + "%d" % dim0])
        info["sampler"][sampler_name]["covmat"][i_0th, i_1st] = 1 / 100000
        info["sampler"][sampler_name]["covmat"][i_1st, i_0th] = 1 / 100000
        info["sampler"][sampler_name]["learn_proposal"] = False
    elif sampler_name == "polychord":
        info["sampler"][sampler_name]["nlive"] = dim0 + dim1
        info["sampler"][sampler_name]["max_ndead"] = n_cycles_all_params * (dim0 + dim1)
    else:
        assert False, "Unknown sampler for this test."
    updated_info, sampler = install_test_wrapper(skip_not_installed, run, info)
    products = sampler.products()
    # TEST: same (steps block i / speed i / dim i) (steps block 1 = evals[1] - evals[0])
    test_func = lambda n_evals, dim0, speed0, dim1, speed1: (
            abs((n_evals[1] - n_evals[0]) / speed1 / dim1 / (
                    n_evals[0] / speed0 / dim0)) - 1)
    # Tolerance accounting for random starts of the proposers (PolyChord and MCMC) and for
    # steps outside prior bounds, where likelihoods are not evaluated (MCMC only)
    tolerance = 0.1
    if sampler_name == "polychord":
        assert test_func(n_evals, dim0, speed0, dim1, speed1) <= tolerance, (
            ("%g > %g" % (test_func(n_evals, dim0, speed0, dim1, speed1), tolerance)))
    elif sampler_name == "mcmc" and info["sampler"][sampler_name].get("drag"):
        assert test_func(n_evals, dim0, speed0, dim1, 2 * speed1) <= tolerance, (
            ("%g > %g" % (test_func(n_evals, dim0, speed0, dim1, speed1), tolerance)))
    elif sampler_name == "mcmc" and info["sampler"][sampler_name].get("oversample"):
        assert test_func(n_evals, dim0, speed0, dim1, speed1) <= tolerance, (
            ("%g > %g" % (test_func(n_evals, dim0, speed0, dim1, speed1), tolerance)))
    elif sampler_name == "mcmc":  # just blocking
        assert test_func(n_evals, dim0, speed0, dim1, speed1) <= tolerance, (
            ("%g > %g" % (test_func(n_evals, dim0, speed0, dim1, speed1), tolerance)))
    else:
        raise ValueError("This should not happen!")
    # Finally, test some points of the chain to reproduce the correct likes and derived
    # These are not AssertionError's to override the flakyness of the test
    for _ in range(10):
        i = choice(list(range(len(products["sample"]))))
        chi2_0_chain = -0.5 * products["sample"]["chi2__like0"][i]
        chi2_0_good = like0(
            **{p: products["sample"][p][i] for p in params0})[0]
        chi2_1_chain = -0.5 * products["sample"]["chi2__like1"][i]
        chi2_1_good = like1(
            **{p: products["sample"][p][i] for p in params1})[0]
        if not np.allclose([chi2_0_chain, chi2_1_chain], [chi2_0_good, chi2_1_good]):
            raise ValueError(
                "Likelihoods not reproduced correctly. "
                "Chain has %r but should be %r. " % (
                    [chi2_0_chain, chi2_1_chain], [chi2_0_good, chi2_1_good]) +
                "Full chain point: %r" % products["sample"][i])
        derived_chain = products["sample"][["sum_like0", "sum_like1"]].values[i]
        derived_good = np.array([
            sum(products["sample"][params0].values[i]),
            sum(products["sample"][params1].values[i])])
        if not np.allclose(derived_chain, derived_good):
            raise ValueError(
                "Derived params not reproduced correctly. "
                "Chain has %r but should be %r. " % (derived_chain, derived_good) +
                "Full chain point:\n%r" % products["sample"][i])
    print(products["sample"])
