"""
.. module:: samplers.mcmc

:Synopsis: Blocked fast-slow Metropolis sampler (Lewis 1304.4473)
:Author: Antony Lewis (for the CosmoMC sampler, wrapped for cobaya by Jesus Torrado)
"""

import datetime
import re
from collections.abc import Callable, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Union

import numpy as np
from pandas import DataFrame

from cobaya import mpi
from cobaya.collection import (
    OneSamplePoint,
    SampleCollection,
    apply_temperature_cov,
    remove_temperature,
    remove_temperature_cov,
)
from cobaya.conventions import Extension, OutPar, get_version, line_width
from cobaya.functions import inverse_cholesky
from cobaya.log import LoggedError, always_stop_exceptions
from cobaya.model import LogPosterior
from cobaya.mpi import (
    get_mpi_size,
    is_main_process,
    more_than_one_process,
    share_mpi,
    sync_processes,
)
from cobaya.sampler import CovmatSampler
from cobaya.samplers.mcmc.proposal import BlockedProposer
from cobaya.tools import NumberWithUnits, get_external_function, load_DataFrame
from cobaya.typing import empty_dict
from cobaya.yaml import yaml_dump_file

# Avoid importing GetDist if not necessary
if TYPE_CHECKING:
    from getdist import MCSamples


class MCMC(CovmatSampler):
    r"""
    Adaptive, speed-hierarchy-aware MCMC sampler (adapted from CosmoMC)
    \cite{Lewis:2002ah,Lewis:2013hha}.
    """

    sampler_type: str = "mcmc"
    supports_periodic_params = True
    _at_resume_prefer_new = CovmatSampler._at_resume_prefer_new + [
        "burn_in",
        "callback_function",
        "callback_every",
        "max_tries",
        "output_every",
        "learn_every",
        "learn_proposal_Rminus1_max",
        "learn_proposal_Rminus1_max_early",
        "learn_proposal_Rminus1_min",
        "max_samples",
        "Rminus1_stop",
        "Rminus1_cl_stop",
        "Rminus1_cl_level",
        "covmat",
        "covmat_params",
    ]
    _at_resume_prefer_old = CovmatSampler._at_resume_prefer_old + [
        "proposal_scale",
        "blocking",
    ]
    _prior_rejections: int = 0
    file_base_name = "mcmc"

    # instance variables from yaml
    burn_in: NumberWithUnits
    learn_every: NumberWithUnits
    output_every: NumberWithUnits
    callback_every: NumberWithUnits
    temperature: float
    max_tries: NumberWithUnits
    max_samples: int
    drag: bool
    callback_function: Callable | None
    blocking: Sequence | None
    proposal_scale: float
    learn_proposal: bool
    learn_proposal_Rminus1_max: float
    learn_proposal_Rminus1_max_early: float
    Rminus1_cl_level: float
    Rminus1_stop: float
    Rminus1_cl_stop: float
    Rminus1_single_split: int
    learn_proposal_Rminus1_min: float
    measure_speeds: bool
    oversample_thin: int
    oversample_power: float

    def set_instance_defaults(self):
        """Ensure that checkpoint attributes are initialized correctly."""
        super().set_instance_defaults()
        # checkpoint variables
        self.converged = False
        self.mpi_size = None
        self.Rminus1_last = np.inf

    def initialize(self):
        """
        Initializes the sampler: creates the proposal distribution and draws the initial
        sample.
        """
        if not self.model.prior.d():
            raise LoggedError(self.log, "No parameters being varied for sampler")
        self.log.debug("Initializing")
        if self.callback_every is None:
            self.callback_every = self.learn_every
        self._quants_d_units = []
        for q in ["max_tries", "learn_every", "callback_every", "burn_in"]:
            number = NumberWithUnits(getattr(self, q), "d", dtype=int)
            self._quants_d_units.append(number)
            setattr(self, q, number)
        self.output_every = NumberWithUnits(self.output_every, "s", dtype=int)
        if self.temperature is None:
            self.temperature = 1
        elif self.temperature < 1:
            self.mpi_warning("Sampling temperatures <1 can lead to innacurate inference.")
        if is_main_process():
            if self.output.is_resuming() and (max(self.mpi_size or 0, 1) != mpi.size()):
                raise LoggedError(
                    self.log,
                    "Cannot resume a run with a different number of chains: "
                    "was %d and now is %d.",
                    max(self.mpi_size or 0, 1),
                    mpi.size(),
                )
        sync_processes()
        # One collection per MPI process: `name` is the MPI rank + 1
        name = str(1 + mpi.rank())
        self.collection = SampleCollection(
            self.model,
            self.output,
            name=name,
            resuming=self.output.is_resuming(),
            temperature=self.temperature,
            sample_type="mcmc",
            is_batch=more_than_one_process(),
        )
        self.current_point = OneSamplePoint(self.model)
        # Use standard MH steps by default
        self.get_new_sample = self.get_new_sample_metropolis
        # Prepare callback function
        if self.callback_function:
            self.callback_function_callable = get_external_function(
                self.callback_function
            )
        # Useful for getting last points added inside callback function
        self.last_point_callback = 0
        self.i_learn = 1
        # Monitoring/restore progress
        if is_main_process():
            cols = ["N", "timestamp", "acceptance_rate", "Rminus1", "Rminus1_cl"]
            self.progress = DataFrame(columns=cols)
            if self.output and not self.output.is_resuming():
                header_fmt = {"N": 6 * " " + "N", "timestamp": 17 * " " + "timestamp"}
                with open(
                    self.progress_filename(), "w", encoding="utf-8"
                ) as progress_file:
                    progress_file.write(
                        "# "
                        + " ".join(
                            [
                                header_fmt.get(col, ((7 + 8) - len(col)) * " " + col)
                                for col in self.progress.columns
                            ]
                        )
                        + "\n"
                    )
        sync_processes()
        # Get first point, to be discarded -- not possible to determine its weight
        # Still, we need to compute derived parameters, since, as the proposal "blocked",
        # we may be saving the initial state of some block.
        # NB: if resuming but nothing was written (burn-in not finished): re-start
        existing_chain_this_process = bool(len(self.collection))
        existing_chains_any_process = bool(sum(mpi.allgather(len(self.collection))))
        if self.output.is_resuming() and existing_chain_this_process:
            last = len(self.collection) - 1
            initial_point = (
                self.collection[self.collection.sampled_params].iloc[last]
            ).to_numpy(dtype=np.float64, copy=True)
            results = LogPosterior(
                logpost=-remove_temperature(
                    self.collection[OutPar.minuslogpost].iloc[last], self.temperature
                ),
                logpriors=-(
                    self.collection[self.collection.minuslogprior_names]
                    .iloc[last]
                    .to_numpy(dtype=np.float64, copy=True)
                ),
                loglikes=-0.5
                * (
                    self.collection[self.collection.chi2_names]
                    .iloc[last]
                    .to_numpy(dtype=np.float64, copy=True)
                ),
                derived=(
                    self.collection[self.collection.derived_params]
                    .iloc[last]
                    .to_numpy(dtype=np.float64, copy=True)
                ),
            )
        else:
            # NB: max_tries adjusted to dim instead of #cycles (blocking not computed yet)
            self.max_tries.set_scale(self.model.prior.d())
            self.log.info("Getting initial point... (this may take a few seconds)")
            initial_point, results = self.model.get_valid_point(
                max_tries=int(min(self.max_tries.value, 1e7)), random_state=self._rng
            )
        self.current_point.add(initial_point, results)
        self.log.info("Initial point: %s", self.current_point)
        sync_processes()
        # If resuming but no existing chains, assume failed run and ignore blocking
        # if speeds measurement requested
        if (
            self.output.is_resuming()
            and not existing_chains_any_process
            and self.measure_speeds
        ):
            self.blocking = None
        if self.measure_speeds and self.blocking:
            self.mpi_warning(
                "Parameter blocking manually/previously fixed: "
                "speeds will not be measured."
            )
        elif self.measure_speeds:
            n = None if self.measure_speeds is True else int(self.measure_speeds)
            self.model.measure_and_set_speeds(n=n, discard=0, random_state=self._rng)
        # Set up blocked proposer
        self.set_proposer_blocking()
        self.set_proposer_initial_covmat(load=True)
        # sanity check whether initial dispersion of points is plausible given the
        # covariance being used
        if not self.output.is_resuming() and more_than_one_process():
            initial_mean = np.mean(np.array(mpi.allgather(initial_point)), axis=0)
            delta = initial_point - initial_mean
            diag, rot = np.linalg.eigh(self.proposer.get_covariance())
            max_dist = np.max(np.abs(rot.T.dot(delta)) / np.sqrt(diag))
            self.log.debug("Max dist to start mean: %s", max_dist)
            max_dist = mpi.gather(max_dist)
            if mpi.is_main_process() and np.max(max_dist) > 12:
                self.mpi_warning(
                    "The initial points are widely dispersed compared to "
                    "the proposal covariance, it may take a long time to "
                    "burn in (max dist to start mean: %s)",
                    max_dist,
                )
        # Max #(learn+convergence checks) to wait,
        # in case one process dies/hangs without raising error
        self.been_waiting = 0
        self.max_waiting = max(50, self.max_tries.unit_value)
        # Burning-in countdown -- the +1 accounts for the initial point (always accepted)
        self.burn_in_left = self.burn_in.value * self.current_point.output_thin + 1
        self._msg_ready = "Ready to check convergence" + (
            " and learn a new proposal covmat" if self.learn_proposal else ""
        )
        # Initial dummy checkpoint
        # (needed when 1st "learn point" not reached in prev. run)
        self.write_checkpoint()

    @property
    def i_last_slow_block(self):
        """Block-index of the last block considered slow, if binary fast/slow split."""
        if self.drag:
            return next(i for i, o in enumerate(self.oversampling_factors) if o != 1) - 1
        self.log.warning("`i_last_slow_block` is only well defined when dragging.")
        return 0

    @property
    def slow_blocks(self):
        """Parameter blocks which are considered slow, in binary fast/slow splits."""
        return self.blocks[: 1 + self.i_last_slow_block]

    @property
    def slow_params(self):
        """Parameters which are considered slow, in binary fast/slow splits."""
        return list(chain(*self.slow_blocks))

    @property
    def n_slow(self):
        """Number of parameters which are considered slow, in binary fast/slow splits."""
        return len(self.slow_params)

    @property
    def fast_blocks(self):
        """Parameter blocks which are considered fast, in binary fast/slow splits."""
        return self.blocks[self.i_last_slow_block + 1 :]

    @property
    def fast_params(self):
        """Parameters which are considered fast, in binary fast/slow splits."""
        return list(chain(*self.fast_blocks))

    @property
    def n_fast(self):
        """Number of parameters which are considered fast, in binary fast/slow splits."""
        return len(self.fast_params)

    def get_acceptance_rate(self, first=0, last=None) -> np.floating:
        """
        Computes the current acceptance rate, optionally only for ``[first:last]``
        subchain.
        """
        return ((last or self.n()) - (first or 0)) / self.collection[OutPar.weight][
            first:last
        ].sum()

    def set_proposer_blocking(self):
        """Sets up the blocked proposer."""
        if self.blocking:
            # Includes the case in which we are resuming
            self.blocks, self.oversampling_factors = self.model.check_blocking(
                self.blocking
            )
        else:
            self.blocks, self.oversampling_factors = (
                self.model.get_param_blocking_for_sampler(
                    oversample_power=self.oversample_power, split_fast_slow=self.drag
                )
            )
        # Turn off dragging if one block, or if speed differences < 2x, or no differences
        if self.drag:
            if len(self.blocks) == 1:
                self.drag = False
                self.mpi_warning(
                    "Dragging disabled: not possible if there is only one block."
                )
            if max(self.oversampling_factors) / min(self.oversampling_factors) < 2:
                self.drag = False
                self.mpi_warning("Dragging disabled: speed ratios < 2.")
        if self.drag:
            # The definition of oversample_power=1 as spending the same amount of time in
            # the slow and fast block would suggest a 1/2 factor here, but this additional
            # factor of 2 w.r.t. oversampling should produce an equivalent exploration
            # efficiency.
            self.drag_interp_steps = int(
                np.round(
                    self.oversampling_factors[self.i_last_slow_block + 1]
                    * self.n_fast
                    / self.n_slow
                )
            )
            if self.drag_interp_steps < 2:
                self.drag = False
                self.mpi_warning(
                    "Dragging disabled: "
                    "speed ratio and fast-to-slow ratio not large enough."
                )
        # Define proposer and other blocking-related quantities
        if self.drag:
            self.get_new_sample = self.get_new_sample_dragging
            self.mpi_info("Dragging with number of interpolating steps:")
            max_width = len(str(self.drag_interp_steps))
            self.mpi_info("* %" + "%d" % max_width + "d : %r", 1, self.slow_blocks)
            self.mpi_info(
                "* %" + "%d" % max_width + "d : %r",
                self.drag_interp_steps,
                self.fast_blocks,
            )
        elif np.any(np.array(self.oversampling_factors) > 1):
            self.mpi_info("Oversampling with factors:")
            max_width = len(str(max(self.oversampling_factors)))
            for f, b in zip(self.oversampling_factors, self.blocks):
                self.mpi_info("* %" + "%d" % max_width + "d : %r", f, b)
            if self.oversample_thin:
                self.current_point.output_thin = int(
                    np.round(
                        sum(
                            len(b) * o
                            for b, o in zip(
                                self.blocks,
                                self.oversampling_factors,
                            )
                        )
                        / self.model.prior.d()
                    )
                )
        # Save blocking in updated info, in case we want to resume
        self._updated_info["blocking"] = list(zip(self.oversampling_factors, self.blocks))
        sampled_params_list = list(self.model.parameterization.sampled_params())
        blocks_indices = [[sampled_params_list.index(p) for p in b] for b in self.blocks]
        self.proposer = BlockedProposer(
            blocks_indices,
            self._rng,
            oversampling_factors=self.oversampling_factors,
            i_last_slow_block=(self.i_last_slow_block if self.drag else None),
            proposal_scale=self.proposal_scale,
        )
        # Cycle length, taking into account oversampling/dragging
        if self.drag:
            self.cycle_length = self.n_slow
        else:
            self.cycle_length = sum(
                len(b) * o for b, o in zip(blocks_indices, self.oversampling_factors)
            )
        self.mpi_debug("Cycle length in steps: %r", self.cycle_length)
        for number in self._quants_d_units:
            number.set_scale(self.cycle_length // self.current_point.output_thin)

    def set_proposer_initial_covmat(self, load=False):
        """Creates/loads an initial covariance matrix and sets it in the Proposer."""
        if load:
            # Build the initial covariance matrix of the proposal, or load from checkpoint
            self._initial_covmat, where_nan = self._load_covmat(
                prefer_load_old=self.output.is_resuming()
            )
            if np.any(where_nan) and self.learn_proposal:
                # We want to start learning the covmat earlier.
                self.mpi_info(
                    "Covariance matrix "
                    + ("not present" if np.all(where_nan) else "not complete")
                    + ". We will start learning the covariance of the proposal "
                    "earlier: R-1 = %g (would be %g if all params loaded).",
                    self.learn_proposal_Rminus1_max_early,
                    self.learn_proposal_Rminus1_max,
                )
                self.learn_proposal_Rminus1_max = self.learn_proposal_Rminus1_max_early
            self.mpi_debug(
                "Sampling with covmat:\n%s",
                DataFrame(
                    self._initial_covmat,
                    columns=self.model.parameterization.sampled_params(),
                    index=self.model.parameterization.sampled_params(),
                ).to_string(line_width=line_width),
            )
        self.proposer.set_covariance(
            apply_temperature_cov(self._initial_covmat, self.temperature)
        )

    def _get_last_nondragging_block(self, blocks, speeds):
        # blocks and speeds are already sorted
        log_differences = np.zeros(len(blocks) - 1)
        for i in range(len(blocks) - 1):
            log_differences[i] = np.log(np.min(speeds[: i + 1])) - np.log(
                np.min(speeds[i + 1 :])
            )
        i_max = np.argmin(log_differences)
        return i_max

    def run(self):
        """
        Runs the sampler.
        """
        self.mpi_info(
            "Sampling!"
            + (
                " (NB: no accepted step will be saved until %d burn-in samples "
                % self.burn_in.value
                + "have been obtained)"
                if self.burn_in.value
                else ""
            )
        )
        self.n_steps_raw = 0
        last_output: float = 0
        last_n = self.n()
        state_check_every = 1
        with mpi.ProcessState(self) as state:
            while last_n < self.max_samples and not self.converged:
                self.get_new_sample()
                self.n_steps_raw += 1
                if self.output_every.unit:
                    # if output_every in sec, print some info
                    # and dump at fixed time intervals
                    now = datetime.datetime.now()
                    now_sec = now.timestamp()
                    if now_sec >= last_output + self.output_every.value:
                        self.do_output(now)
                        last_output = now_sec
                        state.check_error()
                if self.current_point.weight == 1:
                    # have added new point
                    # Callback function
                    n = self.n()
                    if n != last_n:
                        # and actually added
                        last_n = n
                        if (
                            self.callback_function
                            and not (max(n, 1) % self.callback_every.value)
                            and self.current_point.weight == 1
                        ):
                            self.callback_function_callable(self)
                            self.last_point_callback = len(self.collection)

                        if more_than_one_process():
                            # Checking convergence and (optionally) learning
                            # the covmat of the proposal
                            if self.check_ready() and state.set(mpi.State.READY):
                                self.log.info(
                                    "%s (waiting for the rest...)", self._msg_ready
                                )
                            if state.all_ready():
                                self.mpi_info("All chains are r%s", self._msg_ready[1:])
                                self.check_convergence_and_learn_proposal()
                                self.i_learn += 1
                        else:
                            if self.check_ready():
                                self.log.debug(self._msg_ready)
                                self.check_convergence_and_learn_proposal()
                                self.i_learn += 1
                elif self.current_point.weight % state_check_every == 0:
                    state.check_error()
                    # more frequent checks near beginning
                    state_check_every = min(10, state_check_every + 1)

            if last_n == self.max_samples:
                self.log.info(
                    "Reached maximum number of accepted steps allowed (%s). Stopping.",
                    self.max_samples,
                )

            # Write the last batch of samples ( < output_every (not in sec))
            self.collection.out_update()

        ns = mpi.gather(self.n())
        self.mpi_info("Sampling complete after %d accepted steps.", sum(ns))

    def n(self, burn_in=False):
        """
        Returns the total number of accepted steps taken, including or not burn-in steps
        depending on the value of the `burn_in` keyword.
        """
        return len(self.collection) + (
            0
            if not burn_in
            else (
                self.burn_in.value
                - self.burn_in_left // self.current_point.output_thin
                + 1
            )
        )

    def get_new_sample_metropolis(self):
        """
        Draws a new trial point from the proposal pdf and checks whether it is accepted:
        if it is accepted, it saves the old one into the collection and sets the new one
        as the current state; if it is rejected increases the weight of the current state
        by 1.

        Returns
        -------
        ``True`` for an accepted step, ``False`` for a rejected one.
        """
        trial = self.current_point.values.copy()
        self.proposer.get_proposal(trial)
        trial = self.model.prior.reduce_periodic(trial, copy=False)
        trial_results = self.model.logposterior(trial)
        accept = self.metropolis_accept(trial_results.logpost, self.current_point.logpost)
        self.process_accept_or_reject(accept, trial, trial_results)
        return accept

    def get_new_sample_dragging(self):
        """
        Draws a new trial point in the slow subspace, and gets the corresponding trial
        in the fast subspace by "dragging" the fast parameters.
        Finally, checks the acceptance of the total step using the "dragging" pdf:
        if it is accepted, it saves the old one into the collection and sets the new one
        as the current state; if it is rejected increases the weight of the current state
        by 1.

        Returns
        -------
        ``True`` for an accepted step, ``False`` for a rejected one.
        """
        # Prepare starting and ending points *in the SLOW subspace*
        # "start_" and "end_" mean here the extremes in the SLOW subspace
        current_start_point = self.current_point.values
        current_start_logpost = self.current_point.logpost
        current_end_point = current_start_point.copy()
        self.proposer.get_proposal_slow(current_end_point)
        current_end_point = self.model.prior.reduce_periodic(
            current_end_point, copy=False
        )
        self.log.debug("Proposed slow end-point: %r", current_end_point)
        # Save derived parameters of delta_slow jump, in case I reject all the dragging
        # steps but accept the move in the slow direction only
        current_end = self.model.logposterior(current_end_point)
        if current_end.logpost == -np.inf:
            self.current_point.weight += 1
            return False
        # accumulators for the "dragging" probabilities to be metropolis-tested
        # at the end of the interpolation
        start_drag_logpost_acc = current_start_logpost
        end_drag_logpost_acc = current_end.logpost
        # don't compute derived during drag, unless must be computed anyway
        derived = self.model.requires_derived

        # alloc mem
        delta_fast = np.empty(len(current_start_point))
        # start dragging
        for i_step in range(1, 1 + self.drag_interp_steps):
            self.log.debug("Dragging step: %d", i_step)
            # take a step in the fast direction in both slow extremes
            delta_fast[:] = 0.0
            self.proposer.get_proposal_fast(delta_fast)
            delta_fast = self.model.prior.reduce_periodic(delta_fast, copy=False)
            self.log.debug("Proposed fast step delta: %r", delta_fast)
            proposal_start_point = current_start_point + delta_fast
            # get the new extremes for the interpolated probability
            # (reject if any of them = -inf; avoid evaluating both if just one fails)
            # Force the computation of the (slow blocks) derived params at the starting
            # point, but discard them, since they contain the starting point's fast ones,
            # not used later -- save the end point's ones.
            proposal_start_logpost = self.model.logposterior(
                proposal_start_point,
                return_derived=bool(derived),
                _no_check=True,
            ).logpost
            if proposal_start_logpost != -np.inf:
                proposal_end_point = current_end_point + delta_fast
                proposal_end = self.model.logposterior(
                    proposal_end_point,
                    return_derived=bool(derived),
                    _no_check=True,
                )
                if proposal_end.logpost != -np.inf:
                    # create the interpolated probability and do a Metropolis test
                    frac = i_step / (1 + self.drag_interp_steps)
                    proposal_interp_logpost = (
                        1 - frac
                    ) * proposal_start_logpost + frac * proposal_end.logpost
                    current_interp_logpost = (
                        1 - frac
                    ) * current_start_logpost + frac * current_end.logpost
                    accept_drag = self.metropolis_accept(
                        proposal_interp_logpost, current_interp_logpost
                    )
                    if accept_drag:
                        # If the dragging step was accepted, do the drag
                        current_start_point = proposal_start_point
                        current_start_logpost = proposal_start_logpost
                        current_end_point = proposal_end_point
                        current_end = proposal_end
                else:
                    accept_drag = False
            else:
                accept_drag = False
            self.log.debug(
                "Dragging step: %s", ("accepted" if accept_drag else "rejected")
            )

            # In any case, update the dragging probability for the final metropolis test
            start_drag_logpost_acc += current_start_logpost
            end_drag_logpost_acc += current_end.logpost
        # Test for the TOTAL step
        n_average = 1 + self.drag_interp_steps
        accept = self.metropolis_accept(
            end_drag_logpost_acc / n_average, start_drag_logpost_acc / n_average
        )
        if accept and not derived:
            # recompute with derived parameters (slow parameter ones should be cached)
            current_end = self.model.logposterior(current_end_point)

        self.process_accept_or_reject(accept, current_end_point, current_end)
        self.log.debug("TOTAL step: %s", ("accepted" if accept else "rejected"))
        return accept

    def metropolis_accept(self, logp_trial, logp_current):
        """
        Symmetric-proposal Metropolis-Hastings test.

        Returns
        -------
        ``True`` or ``False``.
        """
        if logp_trial == -np.inf:
            return False
        if logp_trial > logp_current:
            return True
        posterior_ratio = (logp_current - logp_trial) / self.temperature
        return self._rng.standard_exponential() > posterior_ratio

    def process_accept_or_reject(
        self, accept_state: bool, trial: np.ndarray, trial_results: LogPosterior
    ):
        """Processes the acceptance/rejection of the new point."""
        if accept_state:
            # add the old point to the collection (if not burning or initial point)
            if self.burn_in_left <= 0:
                if self.current_point.add_to_collection(self.collection):
                    self.log.debug(
                        "New sample, #%d: \n   %s", self.n(), self.current_point
                    )
                    # Update chain files, if output_every *not* in sec
                    if not self.output_every.unit:
                        if self.n() % self.output_every.value == 0:
                            self.collection.out_update()
            else:
                self.burn_in_left -= 1
                self.log.debug("Burn-in sample:\n   %s", self.current_point)
                if self.burn_in_left == 0 and self.burn_in:
                    self.log.info(
                        "Finished burn-in phase: discarded %d accepted steps.",
                        self.burn_in.value,
                    )
            # set the new point as the current one, with weight one
            self.current_point.add(trial, trial_results)
            self._prior_rejections = 0
        else:  # not accepted
            self.current_point.weight += 1
            if trial_results.logprior == -np.inf:
                self._prior_rejections += 1
            # Failure criterion: chain stuck! (but be more permissive during burn_in)
            # only stop if not a pure-prior rejection
            max_tries_now = self.max_tries.value * (
                1 + (10 - 1) * np.sign(self.burn_in_left)
            )
            if self.current_point.weight - self._prior_rejections > max_tries_now:
                self.collection.out_update()
                raise LoggedError(
                    self.log,
                    "The chain has been stuck for %d attempts, stopping sampling. "
                    "Make sure the reference point is sensible and initial covmat. "
                    "For parameters not included in an initial covmat, the 'proposal' "
                    "width set for each parameter should be of order of the expected "
                    "conditional posterior width, which may be much smaller than the "
                    "marginalized posterior width - choose a smaller "
                    "rather than larger value if in doubt. You can also decrease the "
                    "'proposal_scale' option for mcmc, though small values will sample "
                    "less efficiently once things converge. Or make your starting 'ref'"
                    "tighter around an expected best-fit value\n"
                    "Alternatively (though not advisable) make 'max_tries: np.inf' "
                    "(or 'max_tries: .inf' in yaml).\n"
                    "Current point: %s\nCurrent result: %s\n"
                    "Last proposal: %s\nWith rejected result: %s",
                    max_tries_now,
                    self.current_point,
                    self.current_point.results,
                    trial,
                    trial_results,
                )
            elif self.current_point.weight > max_tries_now and not getattr(
                self, "_prior_tries_warning", False
            ):
                self.log.warning("Proposal has been rejected %s times", max_tries_now)
                self._prior_tries_warning = True

    # Functions to check convergence and learn the covariance of the proposal distribution

    def check_ready(self):
        """
        Checks if the chain(s) is(/are) ready to check convergence and, if requested,
        learn a new covariance matrix for the proposal distribution.
        """
        n = len(self.collection)
        # If *just* (weight==1) got ready to check+learn
        if not (n % self.learn_every.value) and n > 0:
            self.log.info("Learn + convergence test @ %d samples accepted.", n)
            self.model.dump_timing()
            if more_than_one_process():
                self.been_waiting += 1
                if self.been_waiting > self.max_waiting:
                    raise LoggedError(
                        self.log,
                        "Waiting for too long for all chains to be ready. "
                        "Maybe one of them is stuck or died unexpectedly?",
                    )
            return True
        return False

    @np.errstate(all="ignore")
    def check_convergence_and_learn_proposal(self):
        """
        Checks the convergence of the sampling process, and, if requested,
        learns a new covariance matrix for the proposal distribution from the covariance
        of the last samples.
        """
        # Compute Rminus1 of means
        self.been_waiting = 0
        if more_than_one_process():
            # Compute and gather means and covs
            use_first = int(self.n() / 2)
            mean = self.collection.mean(first=use_first, tempered=True)
            cov = self.collection.cov(first=use_first, tempered=True)
            acceptance_rate = self.get_acceptance_rate(use_first)
            Ns, means, covs, acceptance_rates = mpi.array_gather(
                [self.n(), mean, cov, acceptance_rate]
            )
        else:
            # Compute and gather means, covs and CL intervals of last m-1 chain fractions
            m = 1 + self.Rminus1_single_split
            cut = int(len(self.collection) / m)
            try:
                acceptance_rate = self.get_acceptance_rate(cut)
                Ns = np.ones(m - 1) * cut
                ranges = [(i * cut, (i + 1) * cut - 1) for i in range(1, m)]
                means = np.array(
                    [
                        self.collection.mean(first=first, last=last, tempered=True)
                        for first, last in ranges
                    ]
                )
                covs = np.array(
                    [
                        self.collection.cov(first=first, last=last, tempered=True)
                        for first, last in ranges
                    ]
                )
            except always_stop_exceptions:
                raise
            except Exception:
                self.log.info(
                    "Not enough points in chain to check convergence. "
                    "Waiting for next checkpoint."
                )
                return
            acceptance_rates = None
        if is_main_process():
            self.progress.at[self.i_learn, "N"] = sum(Ns)
            self.progress.at[self.i_learn, "timestamp"] = (
                datetime.datetime.now().isoformat()
            )
            acceptance_rate = (
                np.average(acceptance_rates, weights=Ns)
                if acceptance_rates is not None
                else acceptance_rate
            )
            if self.oversample_thin > 1:
                weights_multi_str = (
                    " = 1/avg(%r)" % acceptance_rates.tolist()
                    if acceptance_rates is not None
                    else ""
                )
                self.log.info(
                    " - Average thinned weight: %.3f%s",
                    1 / acceptance_rate,
                    weights_multi_str,
                )
            else:
                accpt_multi_str = (
                    " = avg([%s])" % ", ".join("%.4f" % x for x in acceptance_rates)
                    if acceptance_rates is not None
                    else ""
                )
                self.log.info(
                    " - Acceptance rate: %.3f%s", acceptance_rate, accpt_multi_str
                )
            self.progress.at[self.i_learn, "acceptance_rate"] = acceptance_rate
            # "Within" or "W" term -- our "units" for assessing convergence
            # and our prospective new covariance matrix
            mean_of_covs = np.average(covs, weights=Ns, axis=0)
            # "Between" or "B" term
            # We don't weight with the number of samples in the chains here:
            # shorter chains will likely be outliers, and we want to notice them
            cov_of_means = np.atleast_2d(np.cov(means.T))  # , fweights=Ns)
            # For numerical stability, we turn mean_of_covs into correlation matrix:
            #   rho = (diag(Sigma))^(-1/2) * Sigma * (diag(Sigma))^(-1/2)
            # and apply the same transformation to the mean of covs (same eigenvals!)
            d = np.sqrt(np.diag(cov_of_means))
            corr_of_means = (cov_of_means / d).T / d
            norm_mean_of_covs = (mean_of_covs / d).T / d
            success_means = False
            converged_means = False
            # Cholesky of (normalized) mean of covs and eigvals of Linv*cov_of_means*L
            try:
                Linv = inverse_cholesky(norm_mean_of_covs)
            except np.linalg.LinAlgError:
                self.log.warning(
                    "Negative covariance eigenvectors. "
                    "This may mean that the covariance of the samples does not "
                    "contain enough information at this point. "
                    "Skipping learning a new covmat for now."
                )
            else:
                try:
                    eigvals = np.linalg.eigvalsh(Linv.dot(corr_of_means).dot(Linv.T))
                    success_means = True
                except np.linalg.LinAlgError:
                    self.log.warning(
                        "Could not compute eigenvalues. "
                        "Skipping learning a new covmat for now."
                    )
                else:
                    Rminus1 = max(np.abs(eigvals))
                    self.progress.at[self.i_learn, "Rminus1"] = Rminus1
                    # For real square matrices, a possible def of the cond number is:
                    condition_number = Rminus1 / min(np.abs(eigvals))
                    self.log.debug(" - Condition number = %g", condition_number)
                    self.log.debug(" - Eigenvalues = %r", eigvals)
                    accpt_multi_str = (
                        " = sum(%r)" % Ns.astype(int).tolist()
                        if more_than_one_process()
                        else ""
                    )
                    self.log.info(
                        " - Convergence of means: R-1 = %f after %d accepted steps%s",
                        Rminus1,
                        sum(Ns),
                        accpt_multi_str,
                    )
                    # Have we converged in means?
                    # (criterion must be fulfilled twice in a row)
                    converged_means = max(Rminus1, self.Rminus1_last) < self.Rminus1_stop
        else:
            mean_of_covs = None
            success_means = None
            converged_means = False
            Rminus1 = None
        success_means, converged_means = mpi.share((success_means, converged_means))
        # Check the convergence of the bounds of the confidence intervals
        # Same as R-1, but with the rms deviation from the mean bound
        # in units of the mean standard deviation of the chains
        if converged_means:
            if more_than_one_process():
                mcsamples = self.collection._sampled_to_getdist(
                    first=use_first, tempered=True
                )
                try:
                    bound = np.array(
                        [
                            [
                                mcsamples.confidence(
                                    i, limfrac=self.Rminus1_cl_level / 2.0, upper=which
                                )
                                for i in range(self.model.prior.d())
                            ]
                            for which in [False, True]
                        ]
                    ).T
                    success_bounds = True
                except Exception:
                    bound = None
                    success_bounds = False
                bounds = np.array(mpi.gather(bound))
            else:
                try:
                    mcsamples_list = [
                        self.collection._sampled_to_getdist(
                            first=i * cut, last=(i + 1) * cut - 1, tempered=True
                        )
                        for i in range(1, m)
                    ]
                except always_stop_exceptions:
                    raise
                except Exception:
                    self.log.info(
                        "Not enough points in chain to check c.l. convergence. "
                        "Waiting for next checkpoint."
                    )
                    return
                try:
                    bounds = [
                        np.array(
                            [
                                [
                                    mcs.confidence(
                                        i,
                                        limfrac=self.Rminus1_cl_level / 2.0,
                                        upper=which,
                                    )
                                    for i in range(self.model.prior.d())
                                ]
                                for which in [False, True]
                            ]
                        ).T
                        for mcs in mcsamples_list
                    ]
                    success_bounds = True
                except Exception:
                    bounds = None
                    success_bounds = False
            if is_main_process():
                if success_bounds:
                    Rminus1_cl = np.std(bounds, axis=0).T / np.sqrt(np.diag(mean_of_covs))
                    self.log.debug(" - normalized std's of bounds = %r", Rminus1_cl)
                    Rminus1_cl = np.max(Rminus1_cl)
                    self.progress.at[self.i_learn, "Rminus1_cl"] = Rminus1_cl
                    accpt_multi_str = (
                        " = sum(%r)" % Ns.astype(int).tolist()
                        if more_than_one_process()
                        else ""
                    )
                    self.log.info(
                        " - Convergence of bounds: R-1 = %f after %d accepted steps%s",
                        Rminus1_cl,
                        sum(Ns) if more_than_one_process() else self.n(),
                        accpt_multi_str,
                    )
                    if Rminus1_cl < self.Rminus1_cl_stop:
                        self.converged = True
                        self.log.info("The run has converged!")
                        self._Ns = Ns
                else:
                    self.log.info(
                        "Computation of the bounds was not possible. "
                        "Waiting until the next converge check."
                    )
        # Broadcast and save the convergence status and the last R-1 of means
        if success_means:
            self.Rminus1_last, self.converged = mpi.share(
                (Rminus1, self.converged) if is_main_process() else None
            )
            # Do we want to learn a better proposal pdf?
            if self.learn_proposal and not self.converged:
                good_Rminus1 = (
                    self.learn_proposal_Rminus1_max
                    > self.Rminus1_last
                    > self.learn_proposal_Rminus1_min
                )
                if not good_Rminus1:
                    self.mpi_info(
                        "Convergence less than requested for updates: "
                        "waiting until the next convergence check."
                    )
                    return
                mean_of_covs = mpi.share(mean_of_covs)
                try:
                    self.proposer.set_covariance(mean_of_covs)  # is already tempered
                    self.mpi_info(" - Updated covariance matrix of proposal pdf.")
                    self.mpi_debug("%r", mean_of_covs)
                except Exception:
                    self.mpi_debug(
                        "Updating covariance matrix failed unexpectedly. "
                        "waiting until next covmat learning attempt."
                    )
        # Save checkpoint info
        self.write_checkpoint()

    def do_output(self, date_time):
        """Writes/updates the output products of the chain."""
        self.collection.out_update()
        msg = "Progress @ %s : " % date_time.strftime("%Y-%m-%d %H:%M:%S")
        msg += "%d steps taken" % self.n_steps_raw
        if self.burn_in_left and self.burn_in:  # NB: burn_in_left = 1 even if no burn_in
            msg += " -- still burning in, %d accepted steps left." % self.burn_in_left
        else:
            msg += ", and %d accepted." % self.n()
        self.log.info(msg)

    def write_checkpoint(self):
        """Writes/updates the checkpoint file."""
        if is_main_process() and self.output:
            checkpoint_filename = self.checkpoint_filename()
            self.dump_covmat(
                remove_temperature_cov(self.proposer.get_covariance(), self.temperature)
            )
            checkpoint_info = {
                "sampler": {
                    self.get_name(): dict(
                        [
                            ("converged", self.converged),
                            ("Rminus1_last", self.Rminus1_last),
                            (
                                "burn_in",
                                (
                                    self.burn_in.value  # initial: repeat burn-in if not finished
                                    if not self.n() and self.burn_in_left
                                    else 0
                                ),
                            ),  # to avoid overweighting last point of prev. run
                            ("mpi_size", get_mpi_size()),
                        ]
                    )
                }
            }
            yaml_dump_file(checkpoint_filename, checkpoint_info, error_if_exists=False)
            if not self.progress.empty:
                with open(
                    self.progress_filename(), "a", encoding="utf-8"
                ) as progress_file:
                    fmts = {"N": "{:9f}".format}
                    progress_file.write(
                        self.progress.tail(1).to_string(
                            header=False, index=False, formatters=fmts
                        )
                        + "\n"
                    )
            self.log.debug("Dumped checkpoint and progress info, and current covmat.")

    def converge_info_changed(self, old_info, new_info):
        """Whether convergence parameters have changed between two inputs."""
        converge_params = [
            "Rminus1_stop",
            "Rminus1_cl_stop",
            "Rminus1_cl_level",
            "max_samples",
        ]
        return any(old_info.get(p) != new_info.get(p) for p in converge_params)

    # Finally: returning the computed products ###########################################

    def samples(
        self,
        combined: bool = False,
        skip_samples: float = 0,
        to_getdist: bool = False,
    ) -> Union[SampleCollection, "MCSamples"]:
        """
        Returns the sample of accepted steps.

        Parameters
        ----------
        combined: bool, default: False
            If ``True`` and running more than one MPI process, returns for all processes
            a single sample collection including all parallel chains concatenated, instead
            of the chain of the current process only. For this to work, this method needs
            to be called from all MPI processes simultaneously.
        skip_samples: int or float, default: 0
            Skips some amount of initial samples (if ``int``), or an initial fraction of
            them (if ``float < 1``). If concatenating (``combined=True``), skipping is
            applied before concatenation. Forces the return of a copy.
        to_getdist: bool, default: False
            If ``True``, returns a single :class:`getdist.MCSamples` instance, containing
            all samples, for all MPI processes (``combined`` is ignored).

        Returns
        -------
        SampleCollection, getdist.MCSamples
            The sample of accepted steps.
        """
        if self.temperature != 1 and not to_getdist:
            self.mpi_warning(
                "The MCMC chain(s) are stored with temperature != 1. "
                "Keep that in mind when operating on them, or detemper (in-place) with "
                "products()['sample'].reset_temperature()'."
            )
        collection = self.collection.skip_samples(skip_samples, inplace=False)
        if not (to_getdist or combined):
            return collection
        # In all the remaining cases, we'll concatenate the chains
        if not skip_samples:
            self.mpi_warning(
                "When combining chains, it is recommended to remove some "
                "initial fraction, e.g. 'skip_samples=0.3'"
            )
        collections = mpi.gather(collection)
        if is_main_process():
            if to_getdist:
                collection = collections[0].to_getdist(combine_with=collections[1:])
            else:
                for collection in collections[1:]:
                    collections[0]._append(collection)
                collection = collections[0]
        return mpi.share_mpi(collection)

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
            If ``True`` and running more than one MPI process, the ``sample`` key of the
            returned dictionary contains a sample including all parallel chains
            concatenated, instead of the chain of the current process only. For this to
            work, this method needs to be called from all MPI processes simultaneously.
        skip_samples: int or float, default: 0
            Skips some amount of initial samples (if ``int``), or an initial fraction of
            them (if ``float < 1``). If concatenating (``combined=True``), skipping is
            applied previously to concatenation. Forces the return of a copy.
        to_getdist: bool, default: False
            If ``True``, the ``sample`` key of the returned dictionary contains a single
            :class:`getdist.MCSamples` instance including all samples (``combined`` is
            ignored).

        Returns
        -------
        dict
            A dictionary containing the sample of accepted steps under ``sample`` (as
            :class:`cobaya.collection.SampleCollection` by default, or as
            :class:`getdist.MCSamples` if ``to_getdist=True``), and a progress report
            table under ``"progress"``.
        """
        return {
            "sample": self.samples(
                combined=combined, skip_samples=skip_samples, to_getdist=to_getdist
            ),
            "progress": share_mpi(getattr(self, "progress", None)),
        }

    # Class methods
    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        """Returns regexps for the output files created by this sampler."""
        regexps = [output.collection_regexp(name=None)]
        if minimal:
            return [(r, None) for r in regexps]
        regexps += [
            re.compile(output.prefix_regexp_str + re.escape(ext.lstrip(".")) + "$")
            for ext in [Extension.checkpoint, Extension.progress, Extension.covmat]
        ]
        return [(r, None) for r in regexps]

    @classmethod
    def get_version(cls):
        """
        Returns the version string of this samples (since it is built-in, that of Cobaya).
        """
        return get_version()

    @classmethod
    def _get_desc(cls, info=None):
        drag_string = r" using the fast-dragging procedure described in \cite{Neal:2005}"
        if info is None:
            # Unknown case (no info passed)
            string = " [(if drag: True)%s]" % drag_string
        else:
            string = drag_string if info.get("drag", cls.get_defaults()["drag"]) else ""
        return (
            "Adaptive, speed-hierarchy-aware MCMC sampler (adapted from CosmoMC) "
            r"\cite{Lewis:2002ah,Lewis:2013hha}" + string + "."
        )


# Plotting tool for chain progress #######################################################


def plot_progress(
    progress, ax=None, index=None, figure_kwargs=empty_dict, legend_kwargs=empty_dict
):
    """
    Plots progress of one or more MCMC runs: evolution of R-1
    (for means and c.l. intervals) and acceptance rate.

    Takes a ``progress`` instance (actually a ``pandas.DataFrame``,
    returned as part of the sampler ``products``),
    a chain ``output`` prefix, or a list of those
    for plotting progress of several chains at once.

    You can use ``figure_kwargs`` and ``legend_kwargs`` to pass arguments to
    ``matplotlib.pyplot.figure`` and ``matplotlib.pyplot.legend`` respectively.

    Returns a subplots axes array. Display with ``matplotlib.pyplot.show()``.

    """
    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=2, sharex=True, **figure_kwargs)
    if isinstance(progress, DataFrame):
        pass  # go on to plotting
    elif isinstance(progress, str):
        try:
            if not progress.endswith(Extension.progress):
                progress += Extension.progress
            progress = load_DataFrame(progress)
            # 1-based
            progress.index = np.arange(1, len(progress) + 1)
        except Exception as excpt:
            raise ValueError(
                f"Cannot load progress file {progress!r}: {str(excpt)}"
            ) from excpt
    elif hasattr(type(progress), "__iter__"):
        # Assume is a list of progress'es
        for i, p in enumerate(progress):
            plot_progress(p, ax=ax, index=i + 1)
        return ax
    else:
        raise ValueError("Cannot understand progress argument: %r" % progress)
    # Plot!
    tag_pre = "" if index is None else "%d : " % index
    p = ax[0].semilogy(progress.N, progress.Rminus1, "o-", label=tag_pre + "means")
    ax[0].semilogy(
        progress.N,
        progress.Rminus1_cl,
        "x:",
        c=p[0].get_color(),
        label=tag_pre + "bounds",
    )
    ax[0].set_ylabel(r"$R-1$")
    ax[0].legend(**legend_kwargs)
    ax[1].plot(progress.N, progress.acceptance_rate, "o-")
    ax[1].set_ylabel(r"acc. rate")
    return ax
