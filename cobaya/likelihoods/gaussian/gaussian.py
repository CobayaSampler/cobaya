"""
.. module:: likelihoods.gaussian

:Synopsis: Simple Gaussian likelihood
:Author: Antony Lewis

"""

import numpy as np

from cobaya.functions import chi_squared
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from cobaya.typing import Sequence


class Gaussian(Likelihood):
    """
    Simple Gaussian likelihood.
    """

    file_base_name = "gaussian"

    # yaml variables
    mean: float | Sequence | np.ndarray
    cov: float | Sequence | np.ndarray
    normalized: bool
    input_params_prefix: str

    def initialize_with_params(self):
        """
        Initializes the gaussian distribution.
        """
        self.log.debug("Initializing")

        # Load mean and cov, and check consistency
        if self.mean is not None and self.cov is not None:
            # Convert to numpy arrays and ensure proper dimensionality
            self.mean = np.atleast_1d(self.mean)
            self.cov = np.atleast_2d(self.cov)

            # Check dimensions
            mean_dim = len(self.mean)
            cov_dim1, cov_dim2 = self.cov.shape

            if cov_dim1 != cov_dim2:
                raise LoggedError(
                    self.log,
                    "The covariance matrix does not appear to be square! Got shape %r",
                    self.cov.shape,
                )

            if mean_dim != cov_dim1:
                raise LoggedError(
                    self.log,
                    "The dimensionalities of mean (%d) and covariance (%d) do not match!",
                    mean_dim,
                    cov_dim1,
                )

            if mean_dim != len(self.input_params):
                raise LoggedError(
                    self.log,
                    "The dimensionality is %d (from mean and cov) "
                    "but was passed %d parameters instead.",
                    mean_dim,
                    len(self.input_params),
                )
        else:
            raise LoggedError(
                self.log,
                "You must specify both a mean and a covariance matrix.",
            )

        # Precompute inverse covariance matrix
        try:
            self.inv_cov = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            raise LoggedError(self.log, "The covariance matrix is not invertible!")

        # Precompute normalization constant if needed
        if getattr(self, "normalized", True):
            # Calculate log determinant using slogdet for numerical stability
            sign, logdet = np.linalg.slogdet(self.cov)
            if sign <= 0:
                raise LoggedError(
                    self.log, "The covariance matrix is not positive definite!"
                )

            # Normalization constant: -0.5 * (k * log(2π) + log|Σ|)
            k = len(self.mean)
            self.log_norm = -0.5 * (k * np.log(2 * np.pi) + logdet)
        else:
            self.log_norm = 0.0

    def logp(self, **params_values):
        """
        Computes the log-likelihood for a given set of parameters.
        """
        # Prepare the vector of sampled parameter values
        x = np.array([params_values[p] for p in self.input_params])

        # Calculate (x - μ)
        delta = x - self.mean

        # Calculate chi-squared: (x - μ)ᵀ Σ⁻¹ (x - μ) using efficient function
        chi2 = chi_squared(self.inv_cov, delta)

        # Return log-likelihood
        # For normalized: -0.5 * chi2 + log_norm
        # For unnormalized: -0.5 * chi2
        return -0.5 * chi2 + self.log_norm
