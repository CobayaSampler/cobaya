"""
.. module:: samplers.nautilus

:Synopsis: Interface for the external ML-boosted nested sampler Nautilus.
:Author: Johannes U. Lange (for Nautilus), Jesús Torrado (for the Cobaya wrapper)
"""

from typing import TYPE_CHECKING, Union

import numpy as np

from cobaya.collection import SampleCollection
from cobaya.component import ComponentNotInstalledError
from cobaya.install import do_package_install
from cobaya.log import get_logger
from cobaya.mpi import is_main_process, share_mpi
from cobaya.sampler import Sampler
from cobaya.tools import NumberWithUnits, VersionCheckError, check_module_version

# Avoid importing GetDist if not necessary
if TYPE_CHECKING:
    from getdist import MCSamples  # type: ignore


class nautilus(Sampler):
    r"""
    Nautilus sampler \cite{Lange:2023ydq}, a ML-boosted nested sampler..
    """

    sampler_type: str = "nested"

    # Installation and external output code handling
    package_install = {
        "pip": "nautilus-sampler",
        "min_version": "1.0.6",
    }

    # Variables from yaml
    n_live: NumberWithUnits
    verbose: bool

    def initialize(self):
        """
        Imports the Nautilus sampler and prepares its ``Sampler`` and ``Prior`` classes.
        """
        nautilus = self.load_nautilus(self.log)  # Handled install/version check errors
        # Prepare prior
        self.nautilus_prior = nautilus.Prior()
        for i, p in enumerate(self.model.parameterization.sampled_params()):
            self.nautilus_prior.add_parameter(p, dist=self.model.prior.pdf[i])
        # Prepare likelihood, including external priors, and sampler
        self.nautilus_sampler_kwargs = {
            "n_live": NumberWithUnits(
                self.n_live, "d", scale=self.model.prior.d(), dtype=int
            ).value,
        }

        def loglikelihood(params_dict):
            logpriors_ext = self.model.prior.logps_external(params_dict)
            loglikes, derived = self.model.loglikes(params_dict)
            return (
                sum(logpriors_ext) + sum(loglikes),
                list(logpriors_ext) + list(loglikes) + list(derived),
            )

        self.nautilus_sampler = nautilus.Sampler(
            self.nautilus_prior,
            loglikelihood,
            **self.nautilus_sampler_kwargs,
        )
        # Prepare runner method kwargs
        self.nautilus_run_kwargs = {"verbose": self.verbose}
        # Products
        self.collection = None
        self.logZ = None
        self.logZstd = None

    def run(self):
        self.nautilus_sampler.run(**self.nautilus_run_kwargs)
        # Process results
        self.collection = SampleCollection(
            self.model, self.output, name="1", sample_type="nested"
        )
        for X, w, logpost, derived in zip(
            *self.nautilus_sampler.posterior(return_blobs=True)
        ):
            weight = np.exp(w)
            if weight <= 0:
                continue
            n_prior_ext = len(self.model.prior.external)
            n_likes = len(self.model.likelihood)
            logpriors = [self.model.prior.logps_internal(X)]
            logpriors += list(derived[:n_prior_ext])
            loglikes = derived[n_prior_ext : n_prior_ext + n_likes]
            derived_params = derived[n_prior_ext + n_likes :]
            self.collection.add(
                values=X,
                logpost=logpost + logpriors[0],
                logpriors=logpriors,
                loglikes=loglikes,
                derived=derived_params,
                weight=weight,
            )
        self.logZ = self.nautilus_sampler.log_z
        self.logZstd = None

    def samples(
        self,
        combined: bool = False,
        skip_samples: float = 0,
        to_getdist: bool = False,
    ) -> Union[SampleCollection, "MCSamples", None]:
        """
        Returns the sample of the posterior built out of dead points.

        Parameters
        ----------
        combined: bool, default: False
            If ``True`` returns the same, single posterior for all processes. Otherwise,
            it is only returned for the root process (this behaviour is kept for
            compatibility with the equivalent function for MCMC).
        skip_samples: int or float, default: 0
            No effect (skipping initial samples from a sorted nested sampling sample would
            bias it). Raises a warning if greater than 0.
        to_getdist: bool, default: False
            If ``True``, returns a single :class:`getdist.MCSamples` instance, containing
            all samples, for all MPI processes (``combined`` is ignored).

        Returns
        -------
        SampleCollection, getdist.MCSamples
           The posterior sample.
        """
        if skip_samples:
            self.mpi_warning(
                "Initial samples should not be skipped in nested sampling. "
                "Ignoring 'skip_samples' keyword."
            )
        collection = self.collection
        if not combined and not to_getdist:
            return collection  # None for MPI ranks > 0
        # In all remaining cases, we return the same for all ranks
        if to_getdist and is_main_process():
            collection = collection.to_getdist()
        return share_mpi(collection)

    def products(
        self,
        combined: bool = False,
        skip_samples: float = 0,
        to_getdist: bool = False,
    ) -> dict:
        """
        Returns the products of the sampling process.

        Parameters
        ----------
        combined: bool, default: False
            If ``True`` returns the same, single posterior for all processes. Otherwise,
            it is only returned for the root process (this behaviour is kept for
            compatibility with the equivalent function for MCMC).
        skip_samples: int or float, default: 0
            No effect (skipping initial samples from a sorted nested sampling sample would
            bias it). Raises a warning if greater than 0.
        to_getdist: bool, default: False
            If ``True``, returns :class:`getdist.MCSamples` instances for the full
            posterior sample and the clusters, for all MPI processes (``combined`` is
            ignored).

        Returns
        -------
        dict, None
            A dictionary containing the :class:`cobaya.collection.SampleCollection` of
            accepted steps under ``"sample"``, the log-evidence and its uncertainty
            under ``logZ`` and ``logZstd`` respectively, and the same for the individual
            clusters, if present, under the ``clusters`` key.

        Notes
        -----
        If either ``combined`` or ``to_getdist`` are ``True``, the same products dict is
        returned for all processes. Otherwise, ``None`` is returned for processes of rank
        larger than 0.
        """
        products = {}
        if is_main_process():
            products = {
                "logZ": self.logZ,
                "logZstd": self.logZstd,
                "sample": self.samples(
                    combined=combined, skip_samples=skip_samples, to_getdist=to_getdist
                ),
            }
        do_bcast = combined or to_getdist
        if do_bcast:
            return share_mpi(products)
        return products

    @classmethod
    def load_nautilus(cls, logger=None):
        if logger is None:
            logger = get_logger(cls.__name__)
        try:
            import nautilus  # type: ignore

        except ModuleNotFoundError as excpt:
            raise ComponentNotInstalledError(
                logger,
                "'nautilus' is apparently not installed. Run 'cobaya-install nautilus' or"
                f" 'pip install {cls.package_install['pip']}'.",
            ) from excpt
        check_module_version(nautilus, cls.package_install["min_version"])
        return nautilus

    @classmethod
    def is_installed(cls, **kwargs):
        try:
            cls.load_nautilus()
        except (ComponentNotInstalledError, VersionCheckError):
            return False
        return True

    @classmethod
    def install(cls, code=True, **kwargs):
        logger = get_logger(cls.__name__)
        if not code:
            logger.info("Code not requested. Nothing to do.")
            return True
        return do_package_install(
            cls.__name__, package_install=cls.package_install, logger=logger
        )
