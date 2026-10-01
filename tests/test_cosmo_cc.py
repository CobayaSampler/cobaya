"""
Tests the cosmic chronometers likelihood against the published fit of the covariance
recipe it implements, and checks that it is consistent across Boltzmann codes.
"""

import os

import numpy as np
import pytest

from cobaya.cosmo_input import create_input, planck_base_model
from cobaya.log import LoggedError
from cobaya.model import get_model
from cobaya.theory import Theory

from .common import process_packages_path
from .conftest import install_test_wrapper


class FlatLCDMHubble(Theory):
    """Flat LCDM H(z) without radiation, as used in the reference fit."""

    params = {"H0": None, "Omega_m": None}

    def get_can_provide(self):
        return ["Hubble"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        state["H0"] = params_values_dict["H0"]
        state["Omega_m"] = params_values_dict["Omega_m"]

    def get_Hubble(self, z, units="km/s/Mpc"):
        Omega_m = self.current_state["Omega_m"]
        z = np.asarray(z)
        return self.current_state["H0"] * np.sqrt(Omega_m * (1 + z) ** 3 + 1 - Omega_m)


def _H0_quantiles(systematics, packages_path, skip_not_installed):
    """
    Integrates the posterior of the reference fit (flat priors 50 < H0 < 100,
    0.01 < Omega_m < 0.99) on a grid, returning the median of H0 and the distances to
    the 16% and 84% quantiles.
    """
    model = install_test_wrapper(
        skip_not_installed,
        get_model,
        {
            "likelihood": {"cc.moresco2020": {"systematics": systematics}},
            "theory": {"hubble": FlatLCDMHubble},
            "params": {
                "H0": {"prior": {"min": 50, "max": 100}},
                "Omega_m": {"prior": {"min": 0.01, "max": 0.99}},
            },
            "packages_path": process_packages_path(packages_path),
        },
    )
    H0s = np.linspace(50, 100, 201)
    Omega_ms = np.linspace(0.01, 0.99, 99)
    loglikes = np.array(
        [[model.loglike([H0, Om], return_derived=False) for Om in Omega_ms] for H0 in H0s]
    )
    marginal = np.exp(loglikes - loglikes.max()).sum(axis=1)
    # cumulative mass up to each grid point (half of its own bin)
    cdf = (np.cumsum(marginal) - marginal / 2) / marginal.sum()
    low, median, high = np.interp([0.16, 0.5, 0.84], cdf, H0s)
    return median, high - median, median - low


def test_cc_moresco2020_reproduces_published_fit(packages_path, skip_not_installed):
    # Flat LCDM fit to the same data, with and without the systematics, from the
    # CC_fit.ipynb notebook of https://gitlab.com/mmoresco/CCcovariance, where the
    # data and systematics files come from. That is an emcee run, so it carries
    # sampling noise of a few hundredths of km/s/Mpc.
    assert np.allclose(
        _H0_quantiles(["IMF", "mod_ooo"], packages_path, skip_not_installed),
        [65.995, 5.545, 5.591],
        atol=0.1,
    )
    assert np.allclose(
        _H0_quantiles([], packages_path, skip_not_installed),
        [66.171, 3.770, 3.956],
        atol=0.1,
    )


def test_cc_moresco2020_boltzmann(packages_path, skip_not_installed):
    # Same chi2 from CAMB and CLASS at a Planck-like cosmology, and equal to the one
    # computed directly from their H(z) with the covariance recipe
    install_test_wrapper(
        skip_not_installed,
        get_model,
        {
            "likelihood": {"cc.moresco2020": None},
            "theory": {"hubble": FlatLCDMHubble},
            "params": {"H0": 70.0, "Omega_m": 0.3},
            "packages_path": process_packages_path(packages_path),
        },
    )
    data = np.genfromtxt(
        cc_data_path("HzTable_MM_BC03.dat", packages_path),
        delimiter=",",
        usecols=(0, 1, 2),
    )
    z, H, sigma = data.T
    systematics = np.loadtxt(cc_data_path("data_MM20.dat", packages_path))
    cov = np.diag(sigma**2)
    for column in (1, 4):  # IMF, mod_ooo
        error = H * np.interp(z, systematics[:, 0], systematics[:, column]) / 100
        cov += np.outer(error, error)
    params = {
        "camb": {"H0": 67.4, "ombh2": 0.0224, "omch2": 0.12, "As": 2.1e-9, "ns": 0.965},
        "classy": {
            "H0": 67.4,
            "omega_b": 0.0224,
            "omega_cdm": 0.12,
            "A_s": 2.1e-9,
            "n_s": 0.965,
            # same neutrino content as CAMB's default: one massive, 0.06 eV
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
        },
    }
    chi2s = []
    for theory in ["camb", "classy"]:
        info = {
            "likelihood": {"cc.moresco2020": None},
            "theory": {theory: None},
            "params": params[theory],
            "packages_path": process_packages_path(packages_path),
        }
        model = install_test_wrapper(skip_not_installed, get_model, info)
        chi2 = -2 * model.loglike({}, return_derived=False)
        residual = H - model.provider.get_Hubble(z)
        assert chi2 == pytest.approx(residual @ np.linalg.solve(cov, residual))
        chi2s.append(chi2)
    assert chi2s[0] == pytest.approx(chi2s[1], rel=1e-4)
    assert chi2s[0] < 2 * len(z)


def cc_data_path(filename, packages_path):
    from cobaya.likelihoods.cc.moresco2020 import _commit, moresco2020

    return os.path.join(
        moresco2020.get_path(process_packages_path(packages_path)),
        f"CCcovariance-{_commit}",
        "data",
        filename,
    )


def _cc_loglike(options, packages_path, point=(70.0, 0.3)):
    model = get_model(
        {
            "likelihood": {"cc.moresco2020": options},
            "theory": {"hubble": FlatLCDMHubble},
            "params": {
                "H0": {"prior": {"min": 50, "max": 100}},
                "Omega_m": {"prior": {"min": 0.01, "max": 0.99}},
            },
            "packages_path": process_packages_path(packages_path),
        }
    )
    return model.loglike(list(point), return_derived=False)


def test_cc_own_data_and_covmat(tmp_path, monkeypatch, packages_path, skip_not_installed):
    packages_path = process_packages_path(packages_path)

    def loglike(options):
        return _cc_loglike(options, packages_path)

    install_test_wrapper(skip_not_installed, loglike, None)
    # User files, with paths relative to the working directory
    monkeypatch.chdir(tmp_path)
    z, H, sigma = np.genfromtxt(
        cc_data_path("HzTable_MM_BC03.dat", packages_path),
        delimiter=",",
        usecols=(0, 1, 2),
    ).T
    np.savetxt("my_data.txt", np.array([z, H, sigma]).T, header="z H sigma")
    np.savetxt("my_data_no_errors.txt", np.array([z, H]).T)
    np.savetxt("my_covmat.txt", np.diag(sigma**2))
    reference = loglike(None)
    reference_no_sys = loglike({"systematics": []})
    assert loglike({"data_file": "my_data.txt"}) == pytest.approx(reference)
    assert loglike(
        {"data_file": "my_data_no_errors.txt", "covmat_file": "my_covmat.txt"}
    ) == pytest.approx(reference)
    assert loglike(
        {"data_file": "my_data.txt", "covmat_file": "my_covmat.txt", "systematics": []}
    ) == pytest.approx(reference_no_sys)
    # A full covariance matrix is used as given
    cov = np.diag(sigma**2) + 0.3 * np.outer(sigma, sigma) * (1 - np.eye(len(z)))
    np.savetxt("my_full_covmat.txt", cov)
    H0, Omega_m = 70.0, 0.3
    residual = H - H0 * np.sqrt(Omega_m * (1 + z) ** 3 + 1 - Omega_m)
    assert loglike(
        {
            "data_file": "my_data.txt",
            "covmat_file": "my_full_covmat.txt",
            "systematics": [],
        }
    ) == pytest.approx(-0.5 * residual @ np.linalg.solve(cov, residual))
    # Covariance matrix not matching the data
    np.savetxt("wrong_covmat.txt", np.diag(sigma[:-1] ** 2))
    with pytest.raises(LoggedError, match="shape"):
        loglike({"data_file": "my_data.txt", "covmat_file": "wrong_covmat.txt"})


def test_cc_cosmo_generator():
    info = create_input(theory="camb", like_cc="CC_moresco2020", **planck_base_model)
    assert "cc.moresco2020" in info["likelihood"]


def test_cc_minimal_subclass(tmp_path):
    # A new set of measurements only needs a data file: no systematics or covariance
    # matrix by default
    from cobaya.likelihoods.base_classes import CC

    data_file = tmp_path / "data.txt"
    np.savetxt(data_file, [[0.5, 90.0, 10.0], [1.5, 170.0, 20.0]])

    class MinimalCC(CC):
        pass

    MinimalCC.data_file = str(data_file)
    model = get_model(
        {
            "likelihood": {"minimal": MinimalCC},
            "theory": {"hubble": FlatLCDMHubble},
            "params": {"H0": 70.0, "Omega_m": 0.3},
        }
    )
    H = 70.0 * np.sqrt(0.3 * (1 + np.array([0.5, 1.5])) ** 3 + 0.7)
    expected = -0.5 * np.sum(((np.array([90.0, 170.0]) - H) / [10.0, 20.0]) ** 2)
    assert model.loglike({}, return_derived=False) == pytest.approx(expected)
