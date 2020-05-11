r"""
.. module:: samplers.minimize

:Synopsis: Posterior/likelihood *maximization* (i.e. -log(post) and chi^2 minimization).
:Author: Jesus Torrado

This is a **maximizer** for posteriors or likelihoods, based on
`scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
and `Py-BOBYQA <https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/index.html>`_
(added in 2.0).

.. note::

   BOBYQA tends to work better on Cosmological problems with the default settings.

.. |br| raw:: html

   <br />

.. note::
   **If you use BOBYQA, please cite it as:**
   |br|
   `C. Cartis, J. Fiala, B. Marteau, L. Roberts,
   "Improving the Flexibility and Robustness of Model-Based Derivative-Free Optimization Solvers"
   (arXiv:1804.00154) <https://arxiv.org/abs/1804.00154>`_
   |br|
   `C. Cartis, L. Roberts, O. Sheridan-Methven,
   "Escaping local minima with derivative-free methods: a numerical investigation"
   (arXiv:1812.11343) <https://arxiv.org/abs/1812.11343>`_
   |br|
   `M.J.D. Powell,
   "The BOBYQA Algorithm for Bound Constrained Optimization without Derivatives",
   (Technical Report 2009/NA06, DAMTP, University of Cambridge)
   <http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf>`_

   **If you use scipy**, you can find `the appropriate references here
   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

It works more effectively when run on top of a Monte Carlo sample: just change the sampler
for ``minimize`` with the desired options, and it will use as a starting point the
*maximum a posteriori* (MAP) or best fit (maximum likelihood, or minimal :math:`\chi^2`)
found so far, as well as the covariance matrix of the sample for rescaling of the
parameter jumps.

As text output, it produces two different files:

- ``[output prefix].minimum.txt``, in
  :ref:`the same format as Cobaya samples <output_format>`,
  but containing a single line.

- ``[output prefix].minimum``, the equivalent **GetDist-formatted** file.

If ``ignore_prior: True``, those files are named ``.bestfit[.txt]`` instead of ``minimum``,
and contain the best-fit (maximum of the likelihood) instead of the MAP
(maximum of the posterior).

When called from a Python script, Cobaya's ``run`` function returns the updated info
and the products described below in the method
:func:`products <samplers.minimize.minimize.products>`.

It is recommended to run a couple of parallel MPI processes:
it will finally pick the best among the results.

.. warning::

   Since Cobaya is often used on likelihoods featuring numerical noise (e.g. Cosmology),
   we have reduced the default accuracy criterion for the minimizers, so that they
   converge in a limited amount of time. If your posterior is fast to evaluate, you may
   want to refine the convergence parameters (see ``override`` options in the ``yaml``
   below).

"""

# Global
import os
import numpy as np
from scipy.optimize import minimize as scpminimize
from typing import Mapping, Optional
import re
import pybobyqa
import logging
from copy import deepcopy

# Local
from cobaya.sampler import Minimizer
from cobaya.conventions import _undo_chi2_name
from cobaya.mpi import get_mpi_size, get_mpi_comm, is_main_process, get_mpi_rank, \
    more_than_one_process
from cobaya.collection import OnePoint, Collection
from cobaya.log import LoggedError
from cobaya.tools import read_dnumber, recursive_update
from cobaya.sampler import CovmatSampler

# Handling scpiy vs BOBYQA
evals_attr = {"scipy": "fun", "bobyqa": "f"}

# Conventions conventions
getdist_ext_ignore_prior = {True: ".bestfit", False: ".minimum"}
get_collection_extension = (
    lambda ignore_prior: getdist_ext_ignore_prior[ignore_prior] + ".txt")


class minimize(Minimizer, CovmatSampler):
    ignore_prior: bool
    confidence_for_unbounded: float
    method: str
    override_bobyqa: Optional[Mapping]
    override_scipy: Optional[Mapping]
    seed: Optional[int]

    def initialize(self):
        self.mpi_info("Initializing")
        self.max_evals = read_dnumber(self.max_evals, self.model.prior.d())
        # Configure target
        method = self.model.loglike if self.ignore_prior else self.model.logpost
        kwargs = {"make_finite": True}
        if self.ignore_prior:
            kwargs["return_derived"] = False
        self.logp = lambda x: method(x, **kwargs)
        # Try to load info from previous samples.
        # If none, sample from reference (make sure that it has finite like/post)
        initial_point = None
        if self.output:
            files = self.output.find_collections()
            collection_in = None
            if files:
                if more_than_one_process():
                    if 1 + get_mpi_rank() <= len(files):
                        collection_in = Collection(
                            self.model, self.output, name=str(1 + get_mpi_rank()),
                            resuming=True)
                else:
                    collection_in = self.output.load_collections(self.model,
                                                                 concatenate=True)
            if collection_in:
                initial_point = (
                    collection_in.bestfit() if self.ignore_prior else collection_in.MAP())
                initial_point = initial_point[
                    list(self.model.parameterization.sampled_params())].values
                self.log.info("Starting from %s of previous chain:",
                              "best fit" if self.ignore_prior else "MAP")
        if initial_point is None:
            this_logp = -np.inf
            while not np.isfinite(this_logp):
                initial_point = self.model.prior.reference()
                this_logp = self.logp(initial_point)
            self.log.info("Starting from random initial point:")
        self.log.info(
            dict(zip(self.model.parameterization.sampled_params(), initial_point)))

        self._bounds = self.model.prior.bounds(
            confidence_for_unbounded=self.confidence_for_unbounded)

        # TODO: if ignore_prior, one should use *like* covariance (this is *post*)
        covmat = self._load_covmat(self.output)[0]

        # scale by conditional parameter widths (since not using correlation structure)
        scales = np.minimum(1 / np.sqrt(np.diag(np.linalg.inv(covmat))),
                            (self._bounds[:, 1] - self._bounds[:, 0]) / 3)

        # Cov and affine transformation
        # Transform to space where initial point is at centre, and cov is normalised
        # Cannot do rotation, as supported minimization routines assume bounds aligned
        # with the parameter axes.
        self._affine_transform_matrix = np.diag(1 / scales)
        self._inv_affine_transform_matrix = np.diag(scales)
        self._scales = scales
        self._affine_transform_baseline = initial_point
        initial_point = self.affine_transform(initial_point)
        np.testing.assert_allclose(initial_point, np.zeros(initial_point.shape))
        bounds = np.array([self.affine_transform(self._bounds[:, i]) for i in range(2)]).T
        # Configure method
        if self.method.lower() == "bobyqa":
            self.minimizer = pybobyqa.solve
            self.kwargs = {
                "objfun": (lambda x: -self.logp_transf(x)),
                "x0": initial_point,
                "bounds": np.array(list(zip(*bounds))),
                "seek_global_minimum": (
                    True if get_mpi_size() in [0, 1] else False),
                "maxfun": int(self.max_evals)}
            self.kwargs = recursive_update(
                deepcopy(self.kwargs), self.override_bobyqa or {})
            self.log.debug("Arguments for pybobyqa.solve:\n%r",
                           {k: v for k, v in self.kwargs.items() if k != "objfun"})
        elif self.method.lower() == "scipy":
            self.minimizer = scpminimize
            self.kwargs = {
                "fun": (lambda x: -self.logp_transf(x)),
                "x0": initial_point,
                "bounds": bounds,
                "options": {
                    "maxiter": self.max_evals,
                    "disp": (self.log.getEffectiveLevel() == logging.DEBUG)}}
            self.kwargs = recursive_update(
                deepcopy(self.kwargs), self.override_scipy or {})
            self.log.debug("Arguments for scipy.optimize.minimize:\n%r",
                           {k: v for k, v in self.kwargs.items() if k != "fun"})
        else:
            methods = ["bobyqa", "scipy"]
            raise LoggedError(
                self.log, "Method '%s' not recognized. Try one of %r.", self.method,
                methods)

    def affine_transform(self, x):
        return (x - self._affine_transform_baseline) / self._scales

    def inv_affine_transform(self, x):
        # fix up rounding errors on bounds to avoid -np.inf likelihoods
        return np.clip(x * self._scales + self._affine_transform_baseline,
                       self._bounds[:, 0], self._bounds[:, 1])

    def logp_transf(self, x):
        return self.logp(self.inv_affine_transform(x))

    def _run(self):
        """
        Runs `scipy.minimize`
        """
        self.log.info("Starting minimization.")
        try:
            self.result = self.minimizer(**self.kwargs)
        except:
            self.log.error("Minimizer '%s' raised an unexpected error:", self.method)
            raise
        self.success = (self.result.success if self.method.lower() == "scipy" else
                        self.result.flag == self.result.EXIT_SUCCESS)
        if self.success:
            self.log.info("Finished successfully!")
        else:
            if self.method.lower() == "bobyqa":
                reason = {
                    self.result.EXIT_MAXFUN_WARNING:
                        "Maximum allowed objective evaluations reached. "
                        "This is the most likely return value when using multiple restarts.",
                    self.result.EXIT_SLOW_WARNING:
                        "Maximum number of slow iterations reached.",
                    self.result.EXIT_FALSE_SUCCESS_WARNING:
                        "Py-BOBYQA reached the maximum number of restarts which decreased the"
                        " objective, but to a worse value than was found in a previous run.",
                    self.result.EXIT_INPUT_ERROR:
                        "Error in the inputs.",
                    self.result.EXIT_TR_INCREASE_ERROR:
                        "Error occurred when solving the trust region subproblem.",
                    self.result.EXIT_LINALG_ERROR:
                        "Linear algebra error, e.g. the interpolation points produced a "
                        "singular linear system."
                }[self.result.flag]
            else:
                reason = ""
            self.log.error("Finished unsuccessfully." +
                           (" Reason: " + reason if reason else ""))
        self.process_results()

    def process_results(self):
        """
        Determines success (or not), chooses best (if MPI)
        and produces output (if requested).
        """
        evals_attr_ = evals_attr[self.method.lower()]
        # If something failed
        if not hasattr(self, "result"):
            return
        if more_than_one_process():
            results = get_mpi_comm().gather(self.result, root=0)
            successes = get_mpi_comm().gather(self.success, root=0)
            _affine_transform_baselines = get_mpi_comm().gather(
                self._affine_transform_baseline, root=0)
            if is_main_process():
                mins = [(getattr(r, evals_attr_) if s else np.inf)
                        for r, s in zip(results, successes)]
                i_min = np.argmin(mins)
                self.result = results[i_min]
                self._affine_transform_baseline = _affine_transform_baselines[i_min]
        else:
            successes = [self.success]
        if is_main_process():
            if not any(successes):
                raise LoggedError(
                    self.log, "Minimization failed! Here is the raw result object:\n%s",
                    str(self.result))
            elif not all(successes):
                self.log.warning('Some minimizations failed!')
            elif more_than_one_process():
                if max(mins) - min(mins) > 1:
                    self.log.warning('Big spread in minima: %r', mins)
                elif max(mins) - min(mins) > 0.2:
                    self.log.warning('Modest spread in minima: %r', mins)

            logp_min = -np.array(getattr(self.result, evals_attr_))
            x_min = self.inv_affine_transform(self.result.x)
            self.log.info("-log(%s) minimized to %g",
                          "likelihood" if self.ignore_prior else "posterior", -logp_min)
            recomputed_post_min = self.model.logposterior(x_min, cached=False)
            recomputed_logp_min = (sum(recomputed_post_min.loglikes) if self.ignore_prior
                                   else recomputed_post_min.logpost)
            if not np.allclose(logp_min, recomputed_logp_min, atol=1e-2):
                raise LoggedError(
                    self.log, "Cannot reproduce log minimum to within 0.01. Maybe your "
                              "likelihood is stochastic or large numerical error? "
                              "Recomputed min: %g (was %g) at %r",
                    recomputed_logp_min, logp_min, x_min)
            self.minimum = OnePoint(
                self.model, self.output, name="",
                extension=get_collection_extension(self.ignore_prior))
            self.minimum.add(x_min, derived=recomputed_post_min.derived,
                             logpost=recomputed_post_min.logpost,
                             logpriors=recomputed_post_min.logpriors,
                             loglikes=recomputed_post_min.loglikes)
            self.log.info(
                "Parameter values at minimum:\n%s", self.minimum.data.to_string())
            self.minimum.out_update()
            self.dump_getdist()

    def products(self):
        r"""
        Returns a dictionary containing:

        - ``minimum``: :class:`OnePoint` that maximizes the posterior or likelihood
          (depending on ``ignore_prior``).

        - ``result_object``: instance of results class of
          `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`_
          or `pyBOBYQA
          <https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/userguide.html>`_.

        - ``M``: inverse of the affine transform matrix (see below).
          ``None`` if no transformation applied.

        - ``X0``: offset of the affine transform matrix (see below)
          ``None`` if no transformation applied.

        If non-trivial ``M`` and ``X0`` are returned, this means that the minimizer has
        been working on an affine-transformed parameter space :math:`x^\prime`, from which
        the real space points can be obtained as :math:`x = M x^\prime + X_0`. This inverse
        transformation needs to be applied to the coordinates appearing inside the
        ``result_object``.
        """
        if is_main_process():
            return {"minimum": self.minimum, "result_object": self.result,
                    "M": self._inv_affine_transform_matrix,
                    "X0": self._affine_transform_baseline}

    def getdist_point_text(self, params, weight=None, minuslogpost=None):
        lines = []
        if weight is not None:
            lines.append('  weight    = %s' % weight)
        if minuslogpost is not None:
            lines.append(' -log(Like) = %s' % minuslogpost)
            lines.append('  chi-sq    = %s' % (2 * minuslogpost))
        lines.append('')
        labels = self.model.parameterization.labels()
        label_list = list(labels)
        if hasattr(params, 'chi2_names'):
            label_list += params.chi2_names
        width = max(len(lab) for lab in label_list) + 2

        def add_section(pars):
            for p, val in pars:
                lab = labels.get(p, p)
                num = label_list.index(p) + 1
                if isinstance(val, (float, np.floating)) and len(str(val)) > 10:
                    lines.append("%5d  %-17.9e %-*s %s" % (num, val, width, p, lab))
                else:
                    lines.append("%5d  %-17s %-*s %s" % (num, val, width, p, lab))

        # num_sampled = len(self.model.parameterization.sampled_params())
        # num_derived = len(self.model.parameterization.derived_params())
        add_section(
            [(p, params[p]) for p in self.model.parameterization.sampled_params()])
        lines.append('')
        add_section([[p, value] for p, value in
                     self.model.parameterization.constant_params().items()])
        lines.append('')
        add_section(
            [[p, params[p]] for p in self.model.parameterization.derived_params()])
        if hasattr(params, 'chi2_names'):
            labels.update({p: r'\chi^2_{\rm %s}' % (
                _undo_chi2_name(p).replace("_", r"\ "))
                           for p in params.chi2_names})
            add_section([[chi2, params[chi2]] for chi2 in params.chi2_names])
        return "\n".join(lines)

    def dump_getdist(self):
        if not self.output:
            return
        getdist_bf = self.getdist_point_text(self.minimum,
                                             minuslogpost=self.minimum['minuslogpost'])
        out_filename = os.path.join(
            self.output.folder,
            self.output.prefix + getdist_ext_ignore_prior[self.ignore_prior])
        with open(out_filename, 'w', encoding="utf-8") as f:
            f.write(getdist_bf)

    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        ignore_prior = bool(info.get("ignore_prior", False))
        ext_collection = get_collection_extension(ignore_prior)
        ext_getdist = getdist_ext_ignore_prior[ignore_prior]
        regexps = [
            re.compile(output.prefix_regexp_str + re.escape(ext.lstrip(".")) + "$")
            for ext in [ext_collection, ext_getdist]]
        return [(r, None) for r in regexps]

    @classmethod
    def check_force_resume(cls, output, info=None):
        """
        Performs the necessary checks on existing files if resuming or forcing
        (including deleting some output files when forcing).
        """
        if output.is_resuming():
            output.log.warning("Minimizer does not support resuming. Ignoring.")
            output.set_resuming(False)
        super().check_force_resume(output, info=info)
