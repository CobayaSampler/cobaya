r"""
.. module:: samplers.Profile

:Synopsis: Posterior/likelihood *profiling*.
:Author: Giacomo Galloni

This is a **profiler** for posteriors or likelihoods, based on
`scipy.optimize.Minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_,
 `Py-BOBYQA <https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/index.html>`_, and
 `iminuit <https://iminuit.readthedocs.io/>`_.

The default is BOBYQA, which tends to work better than scipy on Cosmological problems with default
settings.

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
   <https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf>`_

   **If you use scipy**, you can find `the appropriate references here
   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

   **If you use iminuit**, see the `references here
   <https://iminuit.readthedocs.io/en/stable/citation.html>`_.

It works more effectively when run on top of a Monte Carlo sample: it will use the maximum
a posteriori as a starting point (or the best fit, depending on whether the prior is
ignored, :ref:`see below <minimize_like>`), and the recovered covariance matrix of the
posterior to rescale the variables.

To take advantage of a previous run with a Monte Carlo sampler, either:

- change the ``sampler`` to ``profile`` in the input file,

- or, if running from the shell, repeat the ``cobaya-run`` command used for the original
  run, adding the ``--profile`` flag.

When called from a Python script, Cobaya's ``run`` function returns the updated info
and the products described below in the method
:func:`samplers.profile.Profile.products` (see below).

Independently from the minimizer used, the profiler will run a sequential for loop over
the requested profiled points (depending on the set of likelihoods, this can be very fast
or very slow). At the end, it will pick the best among the results and store them in a
single collection representing the profile.

If text output is requested, it produces two different files depending on the settings:

- if the priors are ignored, ``[output prefix].like_profile.txt``, in
  :ref:`the same format as Cobaya samples <output_format>`,
  but containing a number of lines equal to the requested profiled points.

- if the priors are not ignored, ``[output prefix].post_profile.txt``, the equivalent **GetDist-formatted** file.

Note that the profiled parameter will appear together with the others, but will be set
to the requested values throughout the profiling.

.. warning::

   For historical reasons, in the first two lines of the GetDist-formatted output file
   ``-log(Like)`` indicates the negative log-**posterior**, and similarly ``chi-sq`` is
   :math:`-2` times the log-**posterior**. The actual log-likelihood can be obtained as
   :math:`-2` times the sum of the individual :math:`\chi^2` (``chi2__``, with double
   underscore) in the table that follows these first lines.

Since typically the goal of a profiling is to get the most accurate parameter
behavior as possible, it is recommended to run a couple of parallel MPI processes:
this will help in finding the global minimum in each point. Still, it may be required to modify the default settings of the minimizers (see below).

.. warning::

   Since Cobaya is often used on likelihoods featuring numerical noise (e.g. Cosmology),
   we have reduced the default accuracy criterion for the minimizers, so that they
   converge in a limited amount of time. If your posterior is fast to evaluate, you may
   want to refine the convergence parameters (see ``override`` options in the ``yaml``
   below).


.. _minimize_like:

Maximizing the likelihood instead of the posterior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To maximize the likelihood, add ``ignore_prior: True`` in the ``profile`` input block.

"""

# Global
import os
import re
from itertools import chain
from typing import Optional, Union
import numpy as np
from scipy import optimize
import pybobyqa
from pybobyqa import controller

# Local
from cobaya.model import get_model
from cobaya.sampler import Profiler
from cobaya.collection import OnePoint, SampleCollection
from cobaya.log import LoggedError
from cobaya.tools import read_dnumber, recursive_update
from cobaya.sampler import CovmatSampler
from cobaya import mpi

# Handling scipy vs BOBYQA vs iMinuit
evals_attr = {"scipy": "fun", "bobyqa": "f", "iminuit": "fun"}
valid_methods = tuple(evals_attr)

# Conventions conventions
getdist_ext_ignore_prior = {True: ".like_profile", False: ".post_profile"}
get_collection_extension = (
    lambda ignore_prior: getdist_ext_ignore_prior[ignore_prior] + ".txt")

_bobyqa_errors = {
    controller.EXIT_MAXFUN_WARNING:
        "Maximum allowed objective evaluations reached. "
        "This is the most likely return value when using multiple restarts.",
    controller.EXIT_SLOW_WARNING:
        "Maximum number of slow iterations reached.",
    controller.EXIT_FALSE_SUCCESS_WARNING:
        "Py-BOBYQA reached the maximum number of restarts which decreased the"
        " objective, but to a worse value than was found in a previous run.",
    controller.EXIT_INPUT_ERROR:
        "Error in the inputs.",
    controller.EXIT_TR_INCREASE_ERROR:
        "Error occurred when solving the trust region subproblem.",
    controller.EXIT_LINALG_ERROR:
        "Linear algebra error, e.g. the interpolation points produced a "
        "singular linear system."}


class Profile(Profiler, CovmatSampler):
    file_base_name = 'profile'

    profiled_param: str
    profiled_values: list
    start: Optional[float]
    stop: Optional[float]
    steps: Optional[int]
    ignore_prior: bool
    confidence_for_unbounded: float
    method: str
    best_of: int
    override_bobyqa: Optional[dict]
    override_scipy: Optional[dict]
    override_iminuit: Optional[dict]
    max_evals: Union[str, int]

    def initialize(self):
        """
        Initializes the profiler: sets the boundaries of the problem, selects starting
        points and sets up the affine transformation.
        """
        if self.method not in evals_attr:
            raise LoggedError(self.log, "Method '%s' not recognized. Try one of %r.",
                              self.method, list(evals_attr))
        self.mpi_info("Initializing")
        self.max_iter = int(read_dnumber(self.max_evals, self.model.prior.d()))
        # Get profiled parameter, its values and its index in the sampled parameters
        assert self.profiled_param in self.model.parameterization.sampled_params()
        self.index_profiled_param = list(self.model.parameterization.sampled_params()).index(self.profiled_param)
        self.profiled_values = self.get_profiled_values()
        self.steps = len(self.profiled_values)
        # Configure targets and store models
        models = []
        logps = []
        for value in self.profiled_values:
            model = self.get_profiled_model(value)
            models.append(model)
            logps.append(self.get_logp(model))
        self.logps = logps
        self.models = models
        # Try to load info from previous samples.
        # If none, sample from reference (make sure that it has finite like/post)
        assert self.best_of > 0
        num_starts = int(np.ceil(self.best_of / mpi.size()))
        if self.output:
            files = self.output.find_collections()
        else:
            files = None

        self.profiled_initial_points = {}
        for idx in range(self.steps):
            initial_points = []
            for start in range(num_starts):
                initial_point = None
                if files:
                    collection_in: Optional[SampleCollection]
                    if mpi.more_than_one_process() or num_starts > 1:
                        index = 1 + mpi.rank() * num_starts + start
                        if index <= len(files):
                            collection_in = SampleCollection(
                                self.model, self.output, name=str(index), resuming=True)
                        else:
                            collection_in = None
                    else:
                        collection_in = self.output.load_collections(self.model,
                                                                    concatenate=True)
                    if collection_in:
                        initial_point = (collection_in.bestfit() if self.ignore_prior
                                        else collection_in.MAP())
                        initial_point = initial_point[
                            list(self.models[idx].parameterization.sampled_params())].values
                        self.log.info(
                            "Run %d/%d will start from %s of previous sample:",
                            start + 1, num_starts, "best fit" if self.ignore_prior else "MAP"
                        )
                        # Compute covmat if input but no .covmat file (e.g. with PolyChord)
                        # Prefer old over `covmat` definition in yaml (same as MCMC)
                        self.covmat = collection_in.cov(derived=False)
                        self.covmat_params = list(
                            self.models[idx].parameterization.sampled_params())
                if initial_point is None:
                    for _ in range(self.max_iter // 10 + 5):
                        initial_point = self.models[idx].prior.reference(random_state=self._rng)
                        if np.isfinite(self.logps[idx](initial_point)):
                            break
                    else:
                        raise LoggedError(self.log, "Could not find random starting point "
                                                    "giving finite posterior")
                    self.log.info(
                        "Run %d/%d (profiled point %d out of %d) will start from random initial point:",
                        start + 1, num_starts, idx + 1, self.steps
                    )
                self.log.info(
                    dict(zip(self.models[idx].parameterization.sampled_params(), initial_point)))
                initial_points.append(initial_point)
            self.profiled_initial_points[idx] = initial_points

        self._bounds = self.models[0].prior.bounds(
            confidence_for_unbounded=self.confidence_for_unbounded)
        # TODO: if ignore_prior, one should use *like* covariance (this is *post*)
        covmat = self._load_covmat(prefer_load_old=self.output)[0]
        # Remove profiled parameter from covmat
        covmat = np.delete(covmat, self.index_profiled_param, axis=0)  # delete row
        covmat = np.delete(covmat, self.index_profiled_param, axis=1)  # delete column
        # scale by conditional parameter widths (since not using correlation structure)
        scales = np.minimum(1 / np.sqrt(np.diag(np.linalg.inv(covmat))),
                            (self._bounds[:, 1] - self._bounds[:, 0]) / 3)
        # Cov and affine transformation
        # Transform to space where initial point is at centre, and cov is normalised
        # Cannot do rotation, as supported minimization routines assume bounds aligned
        # with the parameter axes.
        self._affine_transform_matrix = np.diag(1 / scales)
        self._inv_affine_transform_matrix = np.diag(scales)
        self._affine_transform_baseline = None
        self._scales = scales
        self.kwargs = None
        self.results = []
        self.minima = SampleCollection(
            self.model, self.output, name="",
            extension=get_collection_extension(self.ignore_prior))
        self.full_sets_of_mins = []
        self._affine_transform_baselines = []

    def get_profiled_values(self):
        """
        Returns the values of the profiled parameter at which the likelihood/posterior
        must be evaluated.
        """
        if self.profiled_values is not None:
            return self.profiled_values
        return np.linspace(self.start, self.stop, self.steps, endpoint=True)

    def get_profiled_model(self, value):
        """Returns a new model with the profiled parameter fixed to a given value."""
        new_model = self.model.info()
        new_model["params"][self.profiled_param] = {"value": value}
        return get_model(new_model)

    def get_logp(self, model):
        """Returns the logp function of the model."""
        method = model.loglike if self.ignore_prior else model.logpost
        kwargs = {"make_finite": True}
        if self.ignore_prior:
            kwargs["return_derived"] = False
        return lambda x: method(x, **kwargs)

    def affine_transform(self, x):
        """Transforms a point into the search space."""
        return (x - self._affine_transform_baseline) / self._scales

    def inv_affine_transform(self, x):
        """Transforms a point from the search space back into the parameter space."""
        # Fix up rounding errors on bounds to avoid -np.inf likelihoods
        return np.clip(x * self._scales + self._affine_transform_baseline,
                       self._bounds[:, 0], self._bounds[:, 1])

    def run(self):
        """Runs multiple minimizations to profile the likelihood/posterior."""
        for idx in range(self.steps):
            results = []
            successes = []

            def minuslogp_transf(x):
                return -self.logps[idx](self.inv_affine_transform(x))

            self.log.info("Running profiled point %d out of %d (%s = %s).", idx + 1, len(self.profiled_values), self.profiled_param, self.profiled_values[idx])
            for i, initial_point in enumerate(self.profiled_initial_points[idx]):
                self.log.info(
                    "Starting run %d/%d",
                    i + 1, len(self.profiled_initial_points[idx]))
                self.log.debug("Starting point: %r", initial_point)
                self._affine_transform_baseline = initial_point
                initial_point = self.affine_transform(initial_point)
                np.testing.assert_allclose(initial_point, np.zeros(initial_point.shape))
                bounds = np.array(
                    [self.affine_transform(self._bounds[:, i]) for i in range(2)]).T
                try:
                    # Configure method
                    if self.method.lower() == "bobyqa":
                        self.kwargs = {
                            "objfun": minuslogp_transf,
                            "x0": initial_point,
                            "bounds": np.array(list(zip(*bounds))),
                            "maxfun": self.max_iter,
                            "rhobeg": 1.,
                            "do_logging": self.is_debug()}
                        self.kwargs = recursive_update(self.kwargs,
                                                    self.override_bobyqa or {})
                        self.log.debug("Arguments for pybobyqa.solve:\n%r",
                                    {k: v for k, v in self.kwargs.items() if
                                        k != "objfun"})
                        result = pybobyqa.solve(**self.kwargs)
                        success = result.flag == result.EXIT_SUCCESS
                        if not success:
                            self.log.error(
                                "Finished unsuccessfully. Reason: %s",
                                _bobyqa_errors[result.flag]
                            )
                    elif self.method.lower() == "iminuit":
                        try:
                            import iminuit
                        except ImportError:
                            raise LoggedError(
                                self.log, "You need to install iminuit to use the "
                                        "'iminuit' minimizer. Try 'pip install iminuit'.")
                        self.kwargs = {
                            "fun": minuslogp_transf,
                            "x0": initial_point,
                            "bounds": bounds,
                            "options": {
                                "maxfun": self.max_iter,
                                "disp": self.is_debug()}}
                        self.kwargs = recursive_update(
                            self.kwargs, self.override_iminuit or {}
                        )
                        self.log.debug(
                            "Arguments for iminuit.Minimize:\n%r",
                            {k: v for k, v in self.kwargs.items() if k != "fun"},
                        )
                        result = iminuit.minimize(**self.kwargs, method="migrad")
                        if not (success := result.success):
                            self.log.error(result.message)
                        if mpi.get_mpi_size() > 1:
                            result.pop("minuit")  # problem with pickle/mpi?
                    else:
                        self.kwargs = {
                            "fun": minuslogp_transf,
                            "x0": initial_point,
                            "bounds": bounds,
                            "options": {
                                "maxiter": self.max_iter,
                                "disp": self.is_debug()}}
                        self.kwargs = recursive_update(self.kwargs, self.override_scipy or {})
                        self.log.debug("Arguments for scipy.optimize.Minimize:\n%r",
                                    {k: v for k, v in self.kwargs.items() if k != "fun"})
                        result = optimize.minimize(**self.kwargs)
                        if not (success := result.success):
                            self.log.error("Finished unsuccessfully.")
                    if success:
                        self.log.info("Run %d/%d converged.", i + 1, len(self.profiled_initial_points[idx]))
                except Exception as excpt:
                    self.log.error("Minimizer '%s' raised an unexpected error:", self.method)
                    raise excpt
                results += [result]
                successes += [success]

            self.process_results(self.models[idx], *mpi.zip_gather(
                [results, successes, self.profiled_initial_points[idx],
                [self._inv_affine_transform_matrix] * len(self.profiled_initial_points[idx])]))
            self.log.info("Finished profiled point %d out of %d.", idx + 1, len(self.profiled_initial_points))
        self.log.info("Finished profiling.\nProfiled parameter: %s\nResults: %s",
                      self.profiled_param, self.minima)
        self.dump_txt()

    @mpi.set_from_root(("_inv_affine_transform_matrix", "_affine_transform_baselines",
                        "results", "minima", "full_sets_of_mins"))
    def process_results(self, model, results, successes, affine_transform_baselines,
                        transform_matrices):
        """
        Determines success (or not), chooses best (if MPI or multiple starts)
        and produces output (if requested).
        """
        evals_attr_ = evals_attr[self.method.lower()]
        results = list(chain(*results))
        successes = list(chain(*successes))
        affine_transform_baselines = list(chain(*affine_transform_baselines))
        transform_matrices = list(chain(*transform_matrices))
        if len(results) > 1:
            mins = [(getattr(r, evals_attr_) if s else np.inf)
                    for r, s in zip(results, successes)]
            i_min: int = np.argmin(mins)  # type: ignore
        else:
            i_min = 0
        result = results[i_min]
        # Store results for profiled point
        self.results.append(result)
        self._affine_transform_baselines.append(affine_transform_baselines[i_min])
        self._affine_transform_baseline = affine_transform_baselines[i_min]
        self._inv_affine_transform_matrix = transform_matrices[i_min]
        if not any(successes):
            raise LoggedError(
                self.log, "Minimization failed! Here is the raw result object:\n%s",
                str(result))
        elif not all(successes):
            self.log.warning('Some minimizations failed!')
        elif len(results) > 1:
            self.log.info('Finished successfully!')
            # noinspection PyUnboundLocalVariable
            if max(mins) - min(mins) > 1:
                self.log.warning('Big spread in minima: %r', mins)
            elif max(mins) - min(mins) > 0.2:
                self.log.warning('Modest spread in minima: %r', mins)
        logp_min = -np.array(getattr(result, evals_attr_))
        x_min = self.inv_affine_transform(result.x)
        self.log.info("-log(%s) minimized to %g",
                      "likelihood" if self.ignore_prior else "posterior", -logp_min)
        recomputed_post_min = model.logposterior(x_min, cached=False)
        recomputed_logp_min = (recomputed_post_min.loglike if self.ignore_prior
                               else recomputed_post_min.logpost)
        if not np.allclose(logp_min, recomputed_logp_min, atol=1e-2):
            raise LoggedError(
                self.log, "Cannot reproduce log minimum to within 0.01. Maybe your "
                          "likelihood is stochastic or large numerical error? "
                          "Recomputed min: %g (was %g) at %r",
                recomputed_logp_min, logp_min, x_min)
        minimum = OnePoint(model, self.output, name="",
                                extension=get_collection_extension(self.ignore_prior))
        minimum.add(x_min, derived=recomputed_post_min.derived,
                         logpost=recomputed_post_min.logpost,
                         logpriors=recomputed_post_min.logpriors,
                         loglikes=recomputed_post_min.loglikes)
        minimum.data.insert(0, self.profiled_param, model.parameterization.constant_params()[self.profiled_param])
        # Add minimum to collection
        self.minima._append(minimum)
        self.log.info(
            "Parameter values at minimum:\n%s", minimum.data.to_string())
        if len(results) > 1:
            all_mins = {
                f"{i}": (getattr(res[0], evals_attr_), res[1])
                for i, res in enumerate(zip(results, successes))
            }
            full_set_of_mins = all_mins
            self.log.info("Full set of minima:\n%s", full_set_of_mins)
            self.full_sets_of_mins.append(full_set_of_mins)

    @mpi.set_from_root(("minima", ))
    def dump_txt(self):
        """Writes the results of the profiling to a text file."""
        if not self.output:
            return
        ext_collection = get_collection_extension(self.ignore_prior)
        file_name, _ = self.output.prepare_collection(name="", extension=ext_collection)
        if os.path.exists(file_name):
            pass
        with open(file_name, "w", encoding="utf-8") as out:
            out.write("#" + " ".join(
                f(col) for f, col
                in zip(self.minima._header_formatter, self.minima.data.columns))[1:] + "\n")
        with open(file_name, "a", encoding="utf-8") as out:
            np.savetxt(out, self.minima.data.to_numpy(dtype=np.float64),
                       fmt=self.minima._numpy_fmts)

    def products(self):
        r"""
        Returns a dictionary containing:

        - ``minima``: :class:`SampleCollection` that maximizes the posterior
          or likelihood (depending on ``ignore_prior``) in each profiled point.

        - ``profiled_param``: name of the profiled parameter.

        - ``profiled_values``: requested values of the profiled parameter where
          the minimization was performed.

        - ``results_object``: instances of results class of
          `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`_
          or `pyBOBYQA
          <https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/userguide.html>`_
          or `iMinuit <https://iminuit.readthedocs.io/en/stable/citation.html>`.

        - ``full_sets_of_mins``: dictionaries of minima obtained from multiple initial
          points and multiple profiled points. For each it stores the value of the
          minimized function and a boolean indicating whether the minimization was
          successful or not.
          This returns an empty list if only one initial point was run.

        - ``M``: inverse of the affine transform matrix (see below).
          ``None`` if no transformation applied.

        - ``X0s``: offsets of the affine transform matrix (see below)
          ``None`` if no transformation applied.

        If non-trivial ``M`` and ``X0s`` are returned, this means that the minimizer has
        been working on an affine-transformed parameter space :math:`x^\prime`. Indicating with ``idx`` the index spanning the profiled points, the real space points can be obtained as :math:`x = M x^\prime + X_0^{\rm idx}`.
        This inverse transformation needs to be applied to the coordinates appearing
        inside the ``results_object``.
        """
        return {"minima": self.minima, "profiled_param": self.profiled_param,
                "profiled_values": self.profiled_values, "result_objects": self.results,
                "full_sets_of_mins": self.full_sets_of_mins,
                "M": self._inv_affine_transform_matrix,
                "X0s": self._affine_transform_baselines}

    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        """
        Returns a list of tuples `(regexp, root)` of output files potentially produced.
        If `root` in the tuple is `None`, `output.folder` is used.

        If `minimal=True`, returns regexp's for the files that should really not be there when we are not resuming.
        """
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
            if mpi.is_main_process():
                raise LoggedError(
                    output.log, "Minimizer does not support resuming. "
                                "If you want to start over, force "
                                "('-f', '--force', 'force: True')")
        super().check_force_resume(output, info=info)

    @classmethod
    def _get_desc(cls, info=None):
        if info is None:
            method = None
        else:
            method = info.get("method", cls.get_defaults()["method"])
        desc_bobyqa = (r"Py-BOBYQA implementation "
                       r"\cite{2018arXiv180400154C,2018arXiv181211343C} of the BOBYQA "
                       r"minimization algorithm \cite{BOBYQA}")
        desc_scipy = (r"Scipy minimizer \cite{2020SciPy-NMeth} (check citation for the "
                      r"actual algorithm used at \url{https://docs.scipy.org/doc/scipy/re"
                      r"ference/generated/scipy.optimize.Minimize.html}")
        desc_iminuit = (r"iminuit minimizer(check citation for the "
                        r"actual algorithm used at \url{https://iminuit.readthedocs.io/en/stable/reference.html#scipy-like-interface}")
        if method and method.lower() == "bobyqa":
            return desc_bobyqa
        elif method and method.lower() == "scipy":
            return desc_scipy
        elif method and method.lower() == "iminuit":
            return desc_iminuit
        else:  # unknown method or no info passed (None)
            return ("Minimizer -- method unknown, possibly one of:"
                    "\na) " + desc_bobyqa + "\nb) " + desc_scipy)
