"""General test for samplers. Checks convergence, cluster detection, evidence."""

import numpy as np
from itertools import chain
import os
from scipy.stats import multivariate_normal
from getdist.mcsamples import MCSamplesFromCobaya

from cobaya.likelihoods.gaussian_mixture import info_random_gaussian_mixture
from cobaya.typing import InputDict, SamplersDict
from cobaya.tools import KL_norm
from cobaya.run import run
from .common import process_packages_path, is_travis
from .conftest import install_test_wrapper
from cobaya import mpi

KL_tolerance = 0.07
logZ_nsigmas = 2
O_std_min = 0.01
O_std_max = 0.05
distance_factor = 4

fixed_info: InputDict = {'likelihood': {
    'gaussian_mixture': {'means': [np.array([-0.48591462, 0.10064559, 0.64406749])],
                         'covs': [np.array([[0.00078333, 0.00033134, -0.0002923],
                                            [0.00033134, 0.00218118, -0.00170728],
                                            [-0.0002923, -0.00170728, 0.00676922]])],
                         'input_params_prefix': 'a_', 'output_params_prefix': '',
                         'derived': True}},
    'params': {'a__0': {'prior': {'min': -1, 'max': 1}, 'latex': '\\alpha_{0}'},
               'a__1': {'prior': {'min': -1, 'max': 1}, 'latex': '\\alpha_{1}'},
               'a__2': {'prior': {'min': -1, 'max': 1}, 'latex': '\\alpha_{2}'},
               '_0': {'latex': '\\beta_{0}'},
               '_1': {'latex': '\\beta_{1}'},
               '_2': {'latex': '\\beta_{2}'}}}


def generate_random_info(n_modes, ranges, random_state):
    while True:
        inf = info_random_gaussian_mixture(ranges=ranges,
                                           n_modes=n_modes, input_params_prefix="a_",
                                           O_std_min=O_std_min, O_std_max=O_std_max,
                                           derived=True, random_state=random_state)
        if n_modes == 1:
            break
        means = inf["likelihood"]["gaussian_mixture"]["means"]
        distances = chain(*[[np.linalg.norm(m1 - m2) for m2 in means[i + 1:]]
                            for i, m1 in enumerate(means)])
        if min(distances) >= distance_factor * O_std_max:
            break
    return inf


@mpi.sync_errors
def body_of_sampler_test(info_sampler: SamplersDict, dimension=1, n_modes=1, tmpdir="",
                         packages_path=None, skip_not_installed=False, fixed=False,
                         random_state=None):
    # Info of likelihood and prior
    ranges = np.array([[-1, 1] for _ in range(dimension)])
    if fixed:
        info = fixed_info.copy()
    else:
        info = generate_random_info(n_modes, ranges, random_state=random_state)

    if mpi.is_main_process():
        print("Original mean of the gaussian mode:")
        print(info["likelihood"]["gaussian_mixture"]["means"])
        print("Original covmat of the gaussian mode:")
        print(info["likelihood"]["gaussian_mixture"]["covs"])
    info["sampler"] = info_sampler
    sampler_name = list(info_sampler)[0]
    if random_state is not None:
        info_sampler[sampler_name]["seed"] = random_state.integers(0, 2 ** 31)
    if sampler_name == "mcmc":
        if "covmat" in info_sampler["mcmc"]:
            info["sampler"]["mcmc"]["covmat_params"] = (
                list(info["params"])[:dimension])
    info["debug"] = False
    info["output"] = os.path.join(tmpdir, 'out_chain')
    if packages_path:
        info["packages_path"] = process_packages_path(packages_path)

    updated_info, sampler = install_test_wrapper(skip_not_installed, run, info)
    products = sampler.products()
    products["sample"] = mpi.gather(products["sample"])
    # Done! --> Tests
    if mpi.is_main_process():
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
        if not is_travis():
            try:
                import getdist.plots as gdplots
                from getdist.gaussian_mixtures import MixtureND
                sampled_params = [
                    p for p, v in info["params"].items() if "prior" not in v]
                mixture = MixtureND(
                    info["likelihood"]["gaussian_mixture"]["means"],
                    info["likelihood"]["gaussian_mixture"]["covs"],
                    names=sampled_params, label="truth")
                g = gdplots.getSubplotPlotter()
                to_plot = [mixture, results]
                if clusters:
                    to_plot += clusters
                g.triangle_plot(to_plot, params=sampled_params)
                g.export("test.png")
            except:
                print("Plotting failed!")
        # 1st test: KL divergence
        if n_modes == 1:
            cov_sample, mean_sample = results.getCov(dimension), results.getMeans()
            KL_final = KL_norm(m1=info["likelihood"]["gaussian_mixture"]["means"][0],
                               S1=info["likelihood"]["gaussian_mixture"]["covs"][0],
                               m2=mean_sample[:dimension],
                               S2=cov_sample)
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
                            m1=info["likelihood"]["gaussian_mixture"]["means"][i_c1],
                            S1=info["likelihood"]["gaussian_mixture"]["covs"][i_c1],
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


@mpi.sync_errors
def body_of_test_speeds(info_sampler, manual_blocking=False,
                        packages_path=None, skip_not_installed=False, random_state=None):
    # #dimensions and speed ratio mutually prime (e.g. 2,3,5)
    dim0, dim1 = 5, 2
    speed0, speed1 = 1, 10
    ranges = [[i, i + 1] for i in range(dim0 + dim1)]
    prefix = "a_"
    params0, params1 = (lambda x: (x[:dim0], x[dim0:]))(
        [prefix + str(d) for d in range(dim0 + dim1)])
    derived0, derived1 = "sum_like0", "sum_like1"
    random_state = np.random.default_rng(random_state)
    mean0, cov0 = [info_random_gaussian_mixture(
        ranges=[ranges[i] for i in range(dim0)], n_modes=1, input_params_prefix=prefix,
        O_std_min=0.01, O_std_max=0.2, derived=True, random_state=random_state)
                   ["likelihood"]["gaussian_mixture"][p][0] for p in
                   ["means", "covs"]]
    mean1, cov1 = [info_random_gaussian_mixture(
        ranges=[ranges[i] for i in range(dim0, dim0 + dim1)], n_modes=1,
        input_params_prefix=prefix, O_std_min=0.01, O_std_max=0.2, derived=True,
        random_state=random_state)
                   ["likelihood"]["gaussian_mixture"][p][0] for p in
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
    random_state.shuffle(perm)
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
    info_sampler[sampler_name]["seed"] = random_state.integers(0, 2 ** 31)
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
    info_sampler = info["sampler"][sampler_name]
    if sampler_name == "mcmc":
        info_sampler["measure_speeds"] = False
        info_sampler["burn_in"] = 0
        info_sampler["max_samples"] = \
            info_sampler.get("max_samples", n_cycles_all_params * 10 * (dim0 + dim1))
        # Force mixing of blocks:
        info_sampler["covmat_params"] = list(info["params"])
        info_sampler["covmat"] = 1 / 10000 * np.eye(len(info["params"]))
        i_0th, i_1st = map(
            lambda x: info_sampler["covmat_params"].index(x),
            [prefix + "0", prefix + "%d" % dim0])
        info_sampler["covmat"][i_0th, i_1st] = 1 / 100000
        info_sampler["covmat"][i_1st, i_0th] = 1 / 100000
        # info_sampler["learn_proposal"] = False
    elif sampler_name == "polychord":
        info_sampler["nlive"] = dim0 + dim1
        info_sampler["max_ndead"] = n_cycles_all_params * (dim0 + dim1)
    else:
        assert False, "Unknown sampler for this test."
    updated_info, sampler = install_test_wrapper(skip_not_installed, run, info)
    products = sampler.products()

    # TEST: same (steps block i / speed i / dim i) (steps block 1 = evals[1] - evals[0])
    def test_func(_n_evals, _dim0, _speed0, _dim1, _speed1):
        return (abs((_n_evals[1] - _n_evals[0]) / _speed1 / _dim1 /
                    (_n_evals[0] / _speed0 / _dim0)) - 1)

    # Tolerance accounting for random starts of the proposers (PolyChord and MCMC) and for
    # steps outside prior bounds, where likelihoods are not evaluated (MCMC only)
    tolerance = 0.1
    if sampler_name == "polychord":
        assert test_func(n_evals, dim0, speed0, dim1, speed1) <= tolerance, (
            ("%g > %g" % (test_func(n_evals, dim0, speed0, dim1, speed1), tolerance)))
    elif sampler_name == "mcmc" and info["sampler"][sampler_name].get("drag"):
        assert test_func(n_evals, dim0, speed0, dim1, 2 * speed1) <= tolerance, (
            ("%g > %g" % (test_func(n_evals, dim0, speed0, dim1, speed1), tolerance)))
    elif sampler_name == "mcmc" and \
         info["sampler"][sampler_name].get("oversample_power", 0) > 0:
        assert test_func(n_evals, dim0, speed0, dim1, speed1) <= tolerance, (
            ("%g > %g" % (test_func(n_evals, dim0, speed0, dim1, speed1), tolerance)))
    elif sampler_name == "mcmc":  # just blocking
        assert test_func(n_evals, dim0, speed0, dim1, speed1) <= tolerance, (
            ("%g > %g" % (test_func(n_evals, dim0, speed0, dim1, speed1), tolerance)))
    else:
        raise ValueError("This should not happen!")
    # Finally, test some points of the chain to reproduce the correct likes and derived
    # These are not AssertionError's to override the flakiness of the test
    for _ in range(10):
        i = random_state.choice(list(range(len(products["sample"]))))
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
        derived_chain = \
            products["sample"][["sum_like0", "sum_like1"]].to_numpy(dtype=np.float64)[i]
        derived_good = np.array([
            sum(products["sample"][params0].to_numpy(dtype=np.float64)[i]),
            sum(products["sample"][params1].to_numpy(dtype=np.float64)[i])])
        if not np.allclose(derived_chain, derived_good):
            raise ValueError(
                "Derived params not reproduced correctly. "
                "Chain has %r but should be %r. " % (derived_chain, derived_good) +
                "Full chain point:\n%r" % products["sample"][i])
    print(products["sample"])
