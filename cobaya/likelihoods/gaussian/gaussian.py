"""
.. module:: likelihoods.gaussian

:Synopsis: Gaussian mock likelihood, for tests
:Author: Jesus Torrado

Gaussian *mock* likelihoods, aimed for tests.

The mean and covariance matrix for one or more modes must be specified with the options 
``mean`` and ``covmat`` respectively. The dimensionality of the likelihood and the number
of modes are guessed from this options (if they are consistent to each other).

The pdf is normalised to 1 when integrated over an infinite domain,
regardless of the number of modes.

The following example defines 3 modes in 2 dimensions, and expects two parameters whose
names start with ``test_`` and must be defined in the ``params`` block:

.. code-block:: yaml

   likelihood:
     gaussian:
       mock_prefix: test_
       mean: [ [0.1,0.1],
               [0.3,0.3],
               [0.4,0.5] ]
       cov:  [ [[0.01, 0],
                [0,    0.05]],
               [[0.02,  0.003],
                [0.003, 0.01]],
               [[0.01, 0],
                [0,    0.01]] ]


Derived parameters can be tracked, as many as sampled parameters times the number of modes,
and they represent the standarised parameters of each of the modes, i.e. those distributed
as :math:`\mathcal{N}(0,I)` around each mode (notice that if a mode is close to the
boundary of the prior, you should not expect to recoved a unit covariance matrix from the
sample).

A delay (in seconds) in the likelihood evaluation can be specified with the keyword
``delay``.

.. note::

   This module also provides functions to generate random means and covariances
   -- see automatic documentation below.

Options and defaults
--------------------

Simply copy this block in your input ``yaml`` file and modify whatever options you want
(you can delete the rest).

.. literalinclude:: ../cobaya/likelihoods/gaussian/defaults.yaml
   :language: yaml

"""

# Python 2/3 compatibility
from __future__ import division

# Global
import numpy as np
from scipy.stats import multivariate_normal, uniform, random_correlation
from cobaya.mpi import get_mpi_size

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)


class gaussian(Likelihood):
    """
    Gaussian *mock* likelihood.
    """
    def initialise(self):
        """
        Initialises the gaussian distributions.
        """
        log.info("Initialising")
        # Load mean and cov, and check consistency of n_modes and dimensionality
        if self.mean is not None and self.cov is not None:
            # Wrap them in the right arrays (n_mode, param) and check consistency
            self.mean = np.atleast_1d(self.mean)
            while len(np.array(self.mean).shape) < 2:
                self.mean = np.array([self.mean])
            mean_n_modes, mean_dim = self.mean.shape
            self.cov = np.atleast_1d(self.cov)
            while len(np.array(self.cov).shape) < 3:
                self.cov = np.array([self.cov])
            cov_n_modes, cov_dim, cov_dim_2 = self.cov.shape
            if cov_dim != cov_dim_2:
                log.error("The covariance matrix(/ces) do not appear to be square!\n"
                          "Got %r", self.cov)
                raise HandledException
            if mean_dim != cov_dim:
                log.error(
                    "The dimensionalities guessed from mean(s) and cov(s) do not match!")
                raise HandledException
            if mean_n_modes != cov_n_modes:
                log.error(
                    "The numbers of modes guessed from mean(s) and cov(s) do not match!")
                raise HandledException
            if mean_dim != self.d():
                log.error(
                    "The dimensionality is %d (guessed from given means and covmats) "
                    "but was given %d mock parameters instead. "+
                    ("Maybe you forgot to specify the prefix by which to identify them?"
                     if self.mock_prefix else ""), mean_dim, len(self.input_params))
                raise HandledException
            self.n_modes = mean_n_modes
            if len(self.output_params) != self.d()*self.n_modes:
                log.error(
                    "The number of derived parameters must be equal to the dimensionality times "
                    "the number of modes, i.e. %d x %d = %d, but was given %d derived parameters.",
                    self.d(), self.n_modes, self.d()*self.n_modes, len(self.output_params))
                raise HandledException
        else:
            log.error("You must specify both a mean (or a list of them) and a "
                      "covariance matrix, or a list of them.")
            raise HandledException
        self.gaussians = [multivariate_normal(mean=mean, cov=cov)
                          for mean, cov in zip(self.mean, self.cov)]
        # Prepare the transformation(s) for the derived parameters
        self.choleskyL = [np.linalg.cholesky(cov) for cov in self.cov]

    def logp(self, **params_values):
        """
        Computes the log-likelihood for a given set of parameters.
        """
        self.wait()
        # Prepare the vector of sampled parameter values
        x = np.array([params_values[p] for p in self.input_params])
        # Fill the derived parameters
        derived = params_values.get("derived")
        if derived != None:
            for i in range(self.n_modes):
                standard = np.linalg.inv(self.choleskyL[i]).dot((x-self.mean[i]))
                derived.update(dict(
                    [(p,v) for p,v in
                     zip(self.output_params.keys()[i*self.d():(i+1)*self.d()],standard)]))
        # Compute the likelihood and return
        return (-np.log(self.n_modes) +
                 np.log(sum([gauss.pdf(x) for gauss in self.gaussians])))


# Scripts to generate random means and covariances ########################################

def random_mean(ranges, n_modes=1):
    """
    Returns a uniformly sampled point (as an array) within a list of limits `ranges`.

    The output of this function can be used directly as the value of the option `mean` of
    the :class:`likelihoods.gaussian`.

    If `n_modes>1`, returns an array of such points.
    """
    if get_mpi_size():
        print "WARNING! Using with MPI: different process will produce different random results."
    mean = np.array([uniform.rvs(loc=r[0], scale=r[1]-r[0], size=n_modes) for r in ranges])
    mean = mean.T
    if n_modes == 1:
        mean = mean[0]
    return mean

def random_cov(ranges, O_std_min=1e-2, O_std_max=1, n_modes=1):
    """
    Returns a random covariance matrix, with standard deviations sampled log-uniformly
    from the lenght of the parameter ranges times `O_std_min` and `O_std_max`, and
    uniformly sampled correlation coefficients betweem `rho_min` and `rho_max`.

    The output of this function can be used directly as the value of the option `cov` of
    the :class:`likelihoods.gaussian`.

    If `n_modes>1`, returns a list of such matrices.
    """
    if get_mpi_size():
        print "WARNING! Using with MPI: different process will produce different random results."
    dim = len(ranges)
    scales = np.array([r[1]-r[0] for r in ranges])
    cov = []
    for i in range(n_modes):
        stds = scales * 10**(uniform.rvs(size=dim, loc=np.log10(O_std_min),
                                         scale=np.log10(O_std_max/O_std_min)))
        this_cov = np.diag(stds).dot(
                   (random_correlation.rvs(dim*stds/sum(stds)) if dim>1 else np.eye(1))
                   .dot(np.diag(stds)))
        # Symmetrise (numerical noise is usually introduced in the last step)
        cov += [(this_cov+this_cov.T)/2]
    if n_modes == 1:
        cov = cov[0]
    return cov
