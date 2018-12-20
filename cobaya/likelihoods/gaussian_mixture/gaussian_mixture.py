"""
.. module:: likelihoods.gaussian

:Synopsis: Gaussian mixture likelihood
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import division

# Global
import numpy as np
from scipy.stats import multivariate_normal, uniform, random_correlation
from scipy.special import logsumexp
from collections import OrderedDict as odict

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import HandledException
from cobaya.mpi import get_mpi_size, get_mpi_comm, am_single_or_primary_process
from cobaya.conventions import _likelihood, _params

derived_suffix = "_derived"


class gaussian_mixture(Likelihood):
    """
    Gaussian likelihood.
    """

    def initialize(self):
        """
        Initializes the gaussian distributions.
        """
        self.log.debug("Initializing")
        # Load mean and cov, and check consistency of n_modes and dimensionality
        if self.means is not None and self.covs is not None:
            # Wrap them in the right arrays (n_mode, param) and check consistency
            self.means = np.atleast_1d(self.means)
            while len(np.array(self.means).shape) < 2:
                self.means = np.array([self.means])
            mean_n_modes, mean_dim = self.means.shape
            self.covs = np.atleast_1d(self.covs)
            while len(np.array(self.covs).shape) < 3:
                self.covs = np.array([self.covs])
            cov_n_modes, cov_dim, cov_dim_2 = self.covs.shape
            if cov_dim != cov_dim_2:
                self.log.error("The covariance matrix(/ces) do not appear to be square!\n"
                               "Got %r", self.covs)
                raise HandledException
            if mean_dim != cov_dim:
                self.log.error(
                    "The dimensionalities guessed from mean(s) and cov(s) do not match!")
                raise HandledException
            if mean_n_modes != cov_n_modes:
                self.log.error(
                    "The numbers of modes guessed from mean(s) and cov(s) do not match!")
                raise HandledException
            if mean_dim != self.d():
                self.log.error(
                    "The dimensionality is %d (guessed from given means and covmats) "
                    "but was passed %d parameters instead. " +
                    ("Maybe you forgot to specify the prefix by which to identify them?"
                     if self.prefix else ""), mean_dim, len(self.input_params))
                raise HandledException
            self.n_modes = mean_n_modes
            if self.derived and len(self.output_params) != self.d() * self.n_modes:
                self.log.error(
                    "The number of derived parameters must be equal to the dimensionality"
                    " times the number of modes, i.e. %d x %d = %d, but was given %d "
                    "derived parameters.", self.d(), self.n_modes, self.d() * self.n_modes,
                    len(self.output_params))
                raise HandledException
            elif not self.derived and self.output_params:
                self.log.error(
                    "Derived parameters were requested, but 'derived' option is False. "
                    "Set to True and define as many derived parameters as the "
                    "dimensionality times the number of modes, i.e. %d x %d = %d.",
                    self.d(), self.n_modes, self.d() * self.n_modes)
                raise HandledException
        else:
            self.log.error("You must specify both a mean (or a list of them) and a "
                           "covariance matrix, or a list of them.")
            raise HandledException
        self.gaussians = [multivariate_normal(mean=mean, cov=cov)
                          for mean, cov in zip(self.means, self.covs)]
        if self.weights:
            if not len(self.weights) == len(self.gaussians):
                self.log.error("There must be as many weights as components.")
                raise HandledException
            if not np.isclose(sum(self.weights), 1):
                self.weights = self.weights/sum(self.weights)
                self.log.warn(
                    "Weights of components renormalized to %r", list(self.weights))
        else:
            self.weights = 1/len(self.gaussians)                
        # Prepare the transformation(s) for the derived parameters
        self.choleskyL = [np.linalg.cholesky(cov) for cov in self.covs]

    def logp(self, **params_values):
        """
        Computes the log-likelihood for a given set of parameters.
        """
        self.wait()
        # Prepare the vector of sampled parameter values
        x = np.array([params_values[p] for p in self.input_params])
        # Fill the derived parameters
        derived = params_values.get("_derived")
        if derived is not None:
            for i in range(self.n_modes):
                standard = np.linalg.inv(self.choleskyL[i]).dot((x - self.means[i]))
                derived.update(dict(
                    [(p, v) for p, v in
                     zip(list(self.output_params)[i * self.d():(i + 1) * self.d()], standard)]))
        # Compute the likelihood and return
        return logsumexp([gauss.logpdf(x) for gauss in self.gaussians], b=self.weights)


# Scripts to generate random means and covariances #######################################

def random_mean(ranges, n_modes=1, mpi_warn=True):
    """
    Returns a uniformly sampled point (as an array) within a list of bounds ``ranges``.

    The output of this function can be used directly as the value of the option ``mean``
    of the :class:`likelihoods.gaussian`.

    If ``n_modes>1``, returns an array of such points.
    """
    if get_mpi_size() and mpi_warn:
        print ("WARNING! "
               "Using with MPI: different process will produce different random results.")
    mean = np.array([uniform.rvs(loc=r[0], scale=r[1] - r[0], size=n_modes)
                     for r in ranges])
    mean = mean.T
    if n_modes == 1:
        mean = mean[0]
    return mean


def random_cov(ranges, O_std_min=1e-2, O_std_max=1, n_modes=1, mpi_warn=True):
    """
    Returns a random covariance matrix, with standard deviations sampled log-uniformly
    from the length of the parameter ranges times ``O_std_min`` and ``O_std_max``, and
    uniformly sampled correlation coefficients between ``rho_min`` and ``rho_max``.

    The output of this function can be used directly as the value of the option ``cov`` of
    the :class:`likelihoods.gaussian`.

    If ``n_modes>1``, returns a list of such matrices.
    """
    if get_mpi_size() and mpi_warn:
        print ("WARNING! "
               "Using with MPI: different process will produce different random results.")
    dim = len(ranges)
    scales = np.array([r[1] - r[0] for r in ranges])
    cov = []
    for i in range(n_modes):
        stds = scales * 10 ** (uniform.rvs(size=dim, loc=np.log10(O_std_min),
                                           scale=np.log10(O_std_max / O_std_min)))
        this_cov = np.diag(stds).dot(
            (random_correlation.rvs(dim * stds / sum(stds)) if dim > 1 else np.eye(1))
                .dot(np.diag(stds)))
        # Symmetrize (numerical noise is usually introduced in the last step)
        cov += [(this_cov + this_cov.T) / 2]
    if n_modes == 1:
        cov = cov[0]
    return cov


def info_random_gaussian_mixture(
        ranges, n_modes=1, prefix="", O_std_min=1e-2, O_std_max=1, derived=False,
        mpi_aware=True):
    """
    Wrapper around ``random_mean`` and ``random_cov`` to generate the likelihood and
    parameter info for a random Gaussian.

    If ``mpi_aware=True``, it draws the random stuff only once, and communicates it to
    the rest of the MPI processes.
    """
    if am_single_or_primary_process() or not mpi_aware:
        cov = random_cov(ranges, n_modes=n_modes,
                         O_std_min=O_std_min, O_std_max=O_std_max, mpi_warn=False)
        if n_modes == 1:
            cov = [cov]
        # Make sure it stays away from the edges
        mean = [[]] * n_modes
        for i in range(n_modes):
            std = np.sqrt(cov[i].diagonal())
            factor = 3
            ranges_mean = [[l[0] + factor * s, l[1] - +factor * s] for l, s in zip(ranges, std)]
            # If this implies min>max, take the centre
            ranges_mean = [
                (l if l[0] <= l[1] else 2 * [(l[0] + l[1]) / 2]) for l in ranges_mean]
            mean[i] = random_mean(ranges_mean, n_modes=1, mpi_warn=False)
    elif not am_single_or_primary_process() and mpi_aware:
        mean, cov = None, None
    if mpi_aware:
        mean, cov = get_mpi_comm().bcast(mean, root=0), get_mpi_comm().bcast(cov, root=0)
    dimension = len(ranges)
    info = {_likelihood: {"gaussian_mixture": {
        "means": mean, "covs": cov, "prefix": prefix, "derived": derived}}}
    info[_params] = odict(
        # sampled
        [[prefix + "%d" % i,
          {"prior": {"min": ranges[i][0], "max": ranges[i][1]},
           "latex": r"\alpha_{%i}" % i}]
         for i in range(dimension)] +
        # derived
        ([[prefix + "derived_%d" % i, {"min": -3, "max": 3, "latex": r"\beta_{%i}" % i}]
         for i in range(dimension * n_modes)] if derived else []))
    return info
