"""
.. module:: samplers.mcmc

:Synopsis: Blocked fast-slow Metropolis sampler (Lewis 1304.4473)
:Author: Antony Lewis (for the CosmoMC sampler, wrapped for cobaya by Jesus Torrado)
"""

# Global
from itertools import chain
import numpy as np
from pandas import DataFrame
import datetime
from typing import Sequence, Optional
import re
from copy import deepcopy

# Local
from cobaya import __version__
from cobaya.sampler import CovmatSampler
from cobaya.mpi import get_mpi_size, get_mpi_rank, get_mpi_comm, get_mpi, share_mpi
from cobaya.mpi import more_than_one_process, is_main_process, sync_processes
from cobaya.collection import Collection, OneSamplePoint
from cobaya.conventions import kinds, _weight, _minuslogpost, _covmat_extension
from cobaya.conventions import _line_width, _progress_extension, empty_dict
from cobaya.conventions import _checkpoint_extension
from cobaya.samplers.mcmc.proposal import BlockedProposer
from cobaya.log import LoggedError
from cobaya.tools import get_external_function, NumberWithUnits, load_DataFrame
from cobaya.yaml import yaml_dump_file

_error_tag = 99


class mcmc(CovmatSampler):
    _at_resume_prefer_new = CovmatSampler._at_resume_prefer_new + [
        "burn_in", "callback_function", "callback_every", "max_tries", "output_every",
        "learn_every", "learn_proposal_Rminus1_max", "learn_proposal_Rminus1_max_early",
        "learn_proposal_Rminus1_min", "max_samples", "Rminus1_stop", "Rminus1_cl_stop",
        "Rminus1_cl_level", "covmat", "covmat_params"]
    _at_resume_prefer_old = CovmatSampler._at_resume_prefer_new + [
        "proposal_scale", "blocking"]

    # instance variables from yaml
    burn_in: NumberWithUnits
    learn_every: NumberWithUnits
    output_every: NumberWithUnits
    callback_every: NumberWithUnits
    max_tries: NumberWithUnits
    max_samples: int
    drag: bool
    callback_function: Optional[callable]
    blocking: Optional[Sequence]
    proposal_scale: float
    learn_proposal: bool
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
        super().set_instance_defaults()
        # checkpoint variables
        self.converged = None
        self.mpi_size = None
        self.Rminus1_last = np.inf

    def initialize(self):
        """Initializes the sampler:
        creates the proposal distribution and draws the initial sample."""
        if not self.model.prior.d():
            raise LoggedError(self.log, "No parameters being varied for sampler")
        self.log.debug("Initializing")
        # MARKED FOR DEPRECATION IN v3.0
        if getattr(self, "oversample", None) is not None:
            self.log.warning("*DEPRECATION*: `oversample` will be deprecated in the "
                             "next version. Oversampling is now requested by setting "
                             "`oversample_power` > 0.")
        # END OF DEPRECATION BLOCK
        # MARKED FOR DEPRECATION IN v3.0
        if getattr(self, "check_every", None) is not None:
            self.log.warning("*DEPRECATION*: `check_every` will be deprecated in the "
                             "next version. Please use `learn_every` instead.")
            # BEHAVIOUR TO BE REPLACED BY ERROR:
            self.learn_every = getattr(self, "check_every")
        # END OF DEPRECATION BLOCK
        if self.callback_every is None:
            self.callback_every = self.learn_every
        self._quants_d_units = []
        for q in ["max_tries", "learn_every", "callback_every", "burn_in"]:
            number = NumberWithUnits(getattr(self, q), "d", dtype=int)
            self._quants_d_units.append(number)
            setattr(self, q, number)
        self.output_every = NumberWithUnits(self.output_every, "s", dtype=int)
        if is_main_process():
            if self.output.is_resuming() and (
                    max(self.mpi_size or 0, 1) != max(get_mpi_size(), 1)):
                raise LoggedError(
                    self.log,
                    "Cannot resume a run with a different number of chains: "
                    "was %d and now is %d.", max(self.mpi_size, 1),
                    max(get_mpi_size(), 1))
            if more_than_one_process():
                if get_mpi().Get_version()[0] < 3:
                    raise LoggedError(self.log, "MPI use requires MPI version 3.0 or "
                                                "higher to support IALLGATHER.")
        sync_processes()
        # One collection per MPI process: `name` is the MPI rank + 1
        name = str(1 + (lambda r: r if r is not None else 0)(get_mpi_rank()))
        self.collection = Collection(
            self.model, self.output, name=name, resuming=self.output.is_resuming())
        self.current_point = OneSamplePoint(self.model)
        # Use standard MH steps by default
        self.get_new_sample = self.get_new_sample_metropolis
        # Prepare callback function
        if self.callback_function is not None:
            self.callback_function_callable = (
                get_external_function(self.callback_function))
        # Useful for getting last points added inside callback function
        self.last_point_callback = 0
        # Monitoring/restore progress
        if is_main_process():
            cols = ["N", "timestamp", "acceptance_rate", "Rminus1", "Rminus1_cl"]
            self.progress = DataFrame(columns=cols)
            self.i_learn = 1
            if self.output and not self.output.is_resuming():
                with open(self.progress_filename(), "w",
                          encoding="utf-8") as progress_file:
                    progress_file.write("# " + " ".join(self.progress.columns) + "\n")
        # Get first point, to be discarded -- not possible to determine its weight
        # Still, we need to compute derived parameters, since, as the proposal "blocked",
        # we may be saving the initial state of some block.
        # NB: if resuming but nothing was written (burn-in not finished): re-start
        if self.output.is_resuming() and len(self.collection):
            initial_point = (self.collection[self.collection.sampled_params]
                .iloc[len(self.collection) - 1]).values.copy()
            logpost = -(self.collection[_minuslogpost]
                        .iloc[len(self.collection) - 1].copy())
            logpriors = -(self.collection[self.collection.minuslogprior_names]
                          .iloc[len(self.collection) - 1].copy())
            loglikes = -0.5 * (self.collection[self.collection.chi2_names]
                               .iloc[len(self.collection) - 1].copy())
            derived = (self.collection[self.collection.derived_params]
                       .iloc[len(self.collection) - 1].values.copy())
        else:
            # NB: max_tries adjusted to dim instead of #cycles (blocking not computed yet)
            self.max_tries.set_scale(self.model.prior.d())
            self.log.info("Getting initial point... (this may take a few seconds)")
            initial_point, logpost, logpriors, loglikes, derived = \
                self.model.get_valid_point(max_tries=self.max_tries.value)
            # If resuming but no existing chain, assume failed run and ignore blocking
            # if speeds measurement requested
            if self.output.is_resuming() and not len(self.collection) \
               and self.measure_speeds:
                self.blocking = None
            if self.measure_speeds and self.blocking:
                self.log.warning(
                    "Parameter blocking manually fixed: speeds will not be measured.")
            elif self.measure_speeds:
                n = None if self.measure_speeds is True else int(self.measure_speeds)
                self.model.measure_and_set_speeds(n=n, discard=0)
        self.set_proposer_blocking()
        self.set_proposer_covmat(load=True)
        self.current_point.add(initial_point, derived=derived, logpost=logpost,
                               logpriors=logpriors, loglikes=loglikes)
        self.log.info("Initial point: %s", self.current_point)
        # Max #(learn+convergence checks) to wait,
        # in case one process dies without sending MPI_ABORT
        self.been_waiting = 0
        self.max_waiting = max(50, self.max_tries.unit_value)
        # Burning-in countdown -- the +1 accounts for the initial point (always accepted)
        self.burn_in_left = self.burn_in.value * self.current_point.output_thin + 1
        # Initial dummy checkpoint
        # (needed when 1st "learn point" not reached in prev. run)
        self.write_checkpoint()

    @property
    def i_last_slow_block(self):
        if self.drag:
            return next(i for i, o in enumerate(self.oversampling_factors) if o != 1) - 1
        self.log.warning("`i_last_slow_block` is only well defined when dragging.")
        return 0

    @property
    def slow_blocks(self):
        return self.blocks[:1 + self.i_last_slow_block]

    @property
    def slow_params(self):
        return list(chain(*self.slow_blocks))

    @property
    def n_slow(self):
        return len(self.slow_params)

    @property
    def fast_blocks(self):
        return self.blocks[self.i_last_slow_block + 1:]

    @property
    def fast_params(self):
        return list(chain(*self.fast_blocks))

    @property
    def n_fast(self):
        return len(self.fast_params)

    @property
    def acceptance_rate(self):
        return self.n() / self.collection[_weight].sum()

    def set_proposer_blocking(self):
        if self.blocking:
            # Includes the case in which we are resuming
            self.blocks, self.oversampling_factors = \
                self.model.check_blocking(self.blocking)
        else:
            self.blocks, self.oversampling_factors = \
                self.model.get_param_blocking_for_sampler(
                    oversample_power=self.oversample_power, split_fast_slow=self.drag)
        # Turn off dragging if one block, or if speed differences < 2x, or no differences
        if self.drag:
            if len(self.blocks) == 1:
                self.drag = False
                self.log.warning(
                    "Dragging disabled: not possible if there is only one block.")
            if max(self.oversampling_factors) / min(self.oversampling_factors) < 2:
                self.drag = False
                self.log.warning(
                    "Dragging disabled: speed ratios < 2.")
        if self.drag:
            # The definition of oversample_power=1 as spending the same amount of time in
            # the slow and fast block would suggest a 1/2 factor here, but this additional
            # factor of 2 w.r.t. oversampling should produce an equivalent exploration
            # efficiency.
            self.drag_interp_steps = int(
                np.round(self.oversampling_factors[self.i_last_slow_block + 1] *
                         self.n_fast / self.n_slow))
            if self.drag_interp_steps < 2:
                self.drag = False
                self.log.warning(
                    "Dragging disabled: "
                    "speed ratio and fast-to-slow ratio not large enough.")
        # Define proposer and other blocking-related quantities
        if self.drag:
            # MARKED FOR DEPRECATION IN v3.0
            if getattr(self, "drag_limits", None) is not None:
                self.log.warning("*DEPRECATION*: 'drag_limits' has been deprecated. "
                                 "Use 'oversample_power' to control the amount of "
                                 "dragging steps.")
            # END OF DEPRECATION BLOCK
            self.get_new_sample = self.get_new_sample_dragging
            self.mpi_info("Dragging with number of interpolating steps:")
            max_width = len(str(self.drag_interp_steps))
            self.mpi_info("* %" + "%d" % max_width + "d : %r", 1, self.slow_blocks)
            self.mpi_info("* %" + "%d" % max_width + "d : %r",
                          self.drag_interp_steps, self.fast_blocks)
        elif np.any(np.array(self.oversampling_factors) > 1):
            self.mpi_info("Oversampling with factors:")
            max_width = len(str(max(self.oversampling_factors)))
            for f, b in zip(self.oversampling_factors, self.blocks):
                self.mpi_info("* %" + "%d" % max_width + "d : %r", f, b)
            if self.oversample_thin:
                self.current_point.output_thin = int(np.round(sum(
                    len(b) * o for b, o in zip(self.blocks, self.oversampling_factors)) /
                                                              self.model.prior.d()))

        # Save blocking in updated info, in case we want to resume
        self._updated_info["blocking"] = list(zip(self.oversampling_factors, self.blocks))
        sampled_params_list = list(self.model.parameterization.sampled_params())
        blocks_indices = [[sampled_params_list.index(p) for p in b] for b in self.blocks]
        self.proposer = BlockedProposer(
            blocks_indices, oversampling_factors=self.oversampling_factors,
            i_last_slow_block=(self.i_last_slow_block if self.drag else None),
            proposal_scale=self.proposal_scale)
        # Cycle length, taking into account oversampling/dragging
        if self.drag:
            self.cycle_length = self.n_slow
        else:
            self.cycle_length = sum(len(b) * o for b, o in
                                    zip(blocks_indices, self.oversampling_factors))
        self.log.debug(
            "Cycle length in steps: %r", self.cycle_length)
        for number in self._quants_d_units:
            number.set_scale(self.cycle_length // self.current_point.output_thin)

    def set_proposer_covmat(self, load=False):
        if load:
            # Build the initial covariance matrix of the proposal, or load from checkpoint
            self._covmat, where_nan = self._load_covmat(
                prefer_load_old=self.output.is_resuming())
            if np.any(where_nan) and self.learn_proposal:
                # We want to start learning the covmat earlier.
                self.mpi_info("Covariance matrix " +
                              ("not present" if np.all(where_nan) else "not complete") +
                              ". We will start learning the covariance of the proposal "
                              "earlier: R-1 = %g (would be %g if all params loaded).",
                              self.learn_proposal_Rminus1_max_early,
                              self.learn_proposal_Rminus1_max)
                self.learn_proposal_Rminus1_max = self.learn_proposal_Rminus1_max_early
            self.log.debug(
                "Sampling with covmat:\n%s",
                DataFrame(self._covmat,
                          columns=self.model.parameterization.sampled_params(),
                          index=self.model.parameterization.sampled_params()).to_string(
                    line_width=_line_width))
        self.proposer.set_covariance(self._covmat)

    def _get_last_nondragging_block(self, blocks, speeds):
        # blocks and speeds are already sorted
        log_differences = np.zeros(len(blocks) - 1)
        for i in range(len(blocks) - 1):
            log_differences[i] = (np.log(np.min(speeds[:i + 1])) -
                                  np.log(np.min(speeds[i + 1:])))
        i_max = np.argmin(log_differences)
        return i_max

    def _run(self):
        """
        Runs the sampler.
        """
        self.mpi_info(
            "Sampling!" +
            (" (NB: no accepted step will be saved until %d burn-in samples " %
             self.burn_in.value + "have been obtained)"
             if self.burn_in.value else ""))
        self.n_steps_raw = 0
        last_output = 0
        last_n = self.n()
        while last_n < self.max_samples and not self.converged:
            self.get_new_sample()
            self.n_steps_raw += 1
            if self.output_every.unit:
                # if output_every in sec, print some info and dump at fixed time intervals
                now = datetime.datetime.now()
                now_sec = now.timestamp()
                if now_sec >= last_output + self.output_every.value:
                    self.do_output(now)
                    last_output = now_sec
            if self.current_point.weight == 1:
                # have added new point
                # Callback function
                n = self.n()
                if n != last_n:
                    # and actually added
                    last_n = n
                    if (hasattr(self, "callback_function_callable") and
                            not (max(n, 1) % self.callback_every.value) and
                            self.current_point.weight == 1):
                        self.callback_function_callable(self)
                        self.last_point_callback = len(self.collection)
                    # Checking convergence and (optionally) learning
                    # the covmat of the proposal
                    if self.check_all_ready():
                        self.check_convergence_and_learn_proposal()
                        if is_main_process():
                            self.i_learn += 1
        if last_n == self.max_samples:
            self.log.info("Reached maximum number of accepted steps allowed. "
                          "Stopping.")
        # Make sure the last batch of samples ( < output_every (not in sec)) are written
        self.collection.out_update()
        if more_than_one_process():
            Ns = (lambda x: np.array(get_mpi_comm().gather(x)))(self.n())
            if not is_main_process():
                Ns = []
        else:
            Ns = [self.n()]
        self.mpi_info("Sampling complete after %d accepted steps.", sum(Ns))

    def n(self, burn_in=False):
        """
        Returns the total number of accepted steps taken, including or not burn-in steps
        depending on the value of the `burn_in` keyword.
        """
        return len(self.collection) + (0 if not burn_in
                                       else self.burn_in.value - self.burn_in_left //
                                            self.current_point.output_thin + 1)

    def get_new_sample_metropolis(self):
        """
        Draws a new trial point from the proposal pdf and checks whether it is accepted:
        if it is accepted, it saves the old one into the collection and sets the new one
        as the current state; if it is rejected increases the weight of the current state
        by 1.

        Returns:
           ``True`` for an accepted step, ``False`` for a rejected one.
        """
        trial = self.current_point.values.copy()
        self.proposer.get_proposal(trial)
        try:
            logpost_trial, logprior_trial, loglikes_trial, derived = \
                self.model.logposterior(trial)
        except:
            self.send_error_signal()
            raise
        accept = self.metropolis_accept(logpost_trial, self.current_point.logpost)
        self.process_accept_or_reject(accept, trial, derived,
                                      logpost_trial, logprior_trial, loglikes_trial)
        return accept

    def get_new_sample_dragging(self):
        """
        Draws a new trial point in the slow subspace, and gets the corresponding trial
        in the fast subspace by "dragging" the fast parameters.
        Finally, checks the acceptance of the total step using the "dragging" pdf:
        if it is accepted, it saves the old one into the collection and sets the new one
        as the current state; if it is rejected increases the weight of the current state
        by 1.

        Returns:
           ``True`` for an accepted step, ``False`` for a rejected one.
        """
        # Prepare starting and ending points *in the SLOW subspace*
        # "start_" and "end_" mean here the extremes in the SLOW subspace
        start_slow_point = self.current_point.values.copy()
        start_slow_logpost = self.current_point.logpost
        end_slow_point = start_slow_point.copy()
        self.proposer.get_proposal_slow(end_slow_point)
        self.log.debug("Proposed slow end-point: %r", end_slow_point)
        # Save derived parameters of delta_slow jump, in case I reject all the dragging
        # steps but accept the move in the slow direction only
        end_slow_logpost, end_slow_logprior, end_slow_loglikes, derived = (
            self.model.logposterior(end_slow_point))
        if end_slow_logpost == -np.inf:
            self.current_point.weight += 1
            return False
        # trackers of the dragging
        current_start_point = start_slow_point
        current_end_point = end_slow_point
        current_start_logpost = start_slow_logpost
        current_end_logpost = end_slow_logpost
        current_end_logprior = end_slow_logprior
        current_end_loglikes = end_slow_loglikes
        # accumulators for the "dragging" probabilities to be metropolis-tested
        # at the end of the interpolation
        start_drag_logpost_acc = start_slow_logpost
        end_drag_logpost_acc = end_slow_logpost
        # start dragging
        for i_step in range(1, 1 + self.drag_interp_steps):
            self.log.debug("Dragging step: %d", i_step)
            # take a step in the fast direction in both slow extremes
            delta_fast = np.zeros(len(current_start_point))
            self.proposer.get_proposal_fast(delta_fast)
            self.log.debug("Proposed fast step delta: %r", delta_fast)
            proposal_start_point = current_start_point + delta_fast
            proposal_end_point = current_end_point + delta_fast
            # get the new extremes for the interpolated probability
            # (reject if any of them = -inf; avoid evaluating both if just one fails)
            # Force the computation of the (slow blocks) derived params at the starting
            # point, but discard them, since they contain the starting point's fast ones,
            # not used later -- save the end point's ones.
            proposal_start_logpost = self.model.logposterior(proposal_start_point)[0]
            (proposal_end_logpost, proposal_end_logprior, proposal_end_loglikes,
             derived_proposal_end) = (self.model.logposterior(proposal_end_point)
                                      if proposal_start_logpost > -np.inf
                                      else (-np.inf, None, [], []))
            if proposal_start_logpost > -np.inf and proposal_end_logpost > -np.inf:
                # create the interpolated probability and do a Metropolis test
                frac = i_step / (1 + self.drag_interp_steps)
                proposal_interp_logpost = ((1 - frac) * proposal_start_logpost
                                           + frac * proposal_end_logpost)
                current_interp_logpost = ((1 - frac) * current_start_logpost
                                          + frac * current_end_logpost)
                accept_drag = self.metropolis_accept(proposal_interp_logpost,
                                                     current_interp_logpost)
            else:
                accept_drag = False
            self.log.debug("Dragging step: %s",
                           ("accepted" if accept_drag else "rejected"))
            # If the dragging step was accepted, do the drag
            if accept_drag:
                current_start_point = proposal_start_point
                current_start_logpost = proposal_start_logpost
                current_end_point = proposal_end_point
                current_end_logpost = proposal_end_logpost
                current_end_logprior = proposal_end_logprior
                current_end_loglikes = proposal_end_loglikes
                derived = derived_proposal_end
            # In any case, update the dragging probability for the final metropolis test
            start_drag_logpost_acc += current_start_logpost
            end_drag_logpost_acc += current_end_logpost
        # Test for the TOTAL step
        accept = self.metropolis_accept(end_drag_logpost_acc / self.drag_interp_steps,
                                        start_drag_logpost_acc / self.drag_interp_steps)
        self.process_accept_or_reject(
            accept, current_end_point, derived,
            current_end_logpost, current_end_logprior, current_end_loglikes)
        self.log.debug("TOTAL step: %s", ("accepted" if accept else "rejected"))
        return accept

    def metropolis_accept(self, logp_trial, logp_current):
        """
        Symmetric-proposal Metropolis-Hastings test.

        Returns:
           ``True`` or ``False``.
        """
        if logp_trial == -np.inf:
            return False
        elif logp_trial > logp_current:
            return True
        else:
            return np.random.exponential() > (logp_current - logp_trial)

    def process_accept_or_reject(self, accept_state, trial=None, derived=None,
                                 logpost_trial=None, logprior_trial=None,
                                 loglikes_trial=None):
        """Processes the acceptance/rejection of the new point."""
        if accept_state:
            # add the old point to the collection (if not burning or initial point)
            if self.burn_in_left <= 0:
                if self.current_point.add_to_collection(self.collection):
                    self.log.debug("New sample, #%d: \n   %s",
                                   self.n(), self.current_point)
                    # Update chain files, if output_every *not* in sec
                    if not self.output_every.unit:
                        if self.n() % self.output_every.value == 0:
                            self.collection.out_update()
            else:
                self.burn_in_left -= 1
                self.log.debug("Burn-in sample:\n   %s", self.current_point)
                if self.burn_in_left == 0 and self.burn_in:
                    self.log.info("Finished burn-in phase: discarded %d accepted steps.",
                                  self.burn_in.value)
            # set the new point as the current one, with weight one
            self.current_point.add(trial, derived=derived, logpost=logpost_trial,
                                   logpriors=logprior_trial, loglikes=loglikes_trial)
        else:  # not accepted
            self.current_point.weight += 1
            # Failure criterion: chain stuck! (but be more permissive during burn_in)
            max_tries_now = self.max_tries.value * \
                            (1 + (10 - 1) * np.sign(self.burn_in_left))
            if self.current_point.weight > max_tries_now:
                self.collection.out_update()
                self.send_error_signal()
                raise LoggedError(
                    self.log,
                    "The chain has been stuck for %d attempts. Stopping sampling. "
                    "If this has happened often, try improving your "
                    "reference point/distribution. Alternatively (though not advisable) "
                    "make 'max_tries: np.inf' (or 'max_tries: .inf' in yaml).\n"
                    "Current point: %s", max_tries_now, self.current_point)

    # Functions to check convergence and learn the covariance of the proposal distribution

    def check_all_ready(self):
        """
        Checks if the chain(s) is(/are) ready to check convergence and, if requested,
        learn a new covariance matrix for the proposal distribution.
        """
        msg_ready = ("Ready to check convergence" +
                     (" and learn a new proposal covmat"
                      if self.learn_proposal else ""))
        n = len(self.collection)
        # If *just* (weight==1) got ready to check+learn
        if not (n % self.learn_every.value) and n > 0:
            self.log.info("Learn + convergence test @ %d samples accepted.", n)
            if more_than_one_process():
                self.been_waiting += 1
                if self.been_waiting > self.max_waiting:
                    self.send_error_signal()
                    raise LoggedError(
                        self.log, "Waiting for too long for all chains to be ready. "
                                  "Maybe one of them is stuck or died unexpectedly?")
            self.model.dump_timing()
            # If not MPI size > 1, we are ready
            if not more_than_one_process():
                self.log.debug(msg_ready)
                return True
            # Error check in case any process already sent an error signal
            self.check_error_signal()
            # If MPI, tell the rest that we are ready -- we use a "gather"
            # ("reduce" was problematic), but we are in practice just pinging
            if not hasattr(self, "req"):  # just once!
                self.all_ready = np.empty(get_mpi_size())
                self.req = get_mpi_comm().Iallgather(
                    np.array([1.]), self.all_ready)
                self.log.info(msg_ready + " (waiting for the rest...)")
        # If all processes are ready to learn (= communication finished)
        if self.req.Test() if hasattr(self, "req") else False:
            # Sanity check: actually all processes have finished
            assert np.all(self.all_ready == 1), (
                "This should not happen! Notify the developers. (Got %r)", self.all_ready)
            if more_than_one_process() and is_main_process():
                self.log.info("All chains are r" + msg_ready[1:])
            delattr(self, "req")
            self.been_waiting = 0
            # Another error check, in case the error occurred after sending "ready" signal
            self.check_error_signal()
            # Just in case, a barrier here
            sync_processes()
            return True
        return False

    def check_convergence_and_learn_proposal(self):
        """
        Checks the convergence of the sampling process, and, if requested,
        learns a new covariance matrix for the proposal distribution from the covariance
        of the last samples.
        """
        if more_than_one_process():
            # Compute and gather means, covs and CL intervals of last half of chains
            use_first = int(self.n() / 2)
            mean = self.collection.mean(first=use_first)
            cov = self.collection.cov(first=use_first)
            mcsamples = self.collection._sampled_to_getdist_mcsamples(first=use_first)
            try:
                bound = np.array([[
                    mcsamples.confidence(i, limfrac=self.Rminus1_cl_level / 2.,
                                         upper=which)
                    for i in range(self.model.prior.d())] for which in [False, True]]).T
                success_bounds = True
            except:
                bound = None
                success_bounds = False
            Ns, means, covs, bounds, acceptance_rates = map(
                lambda x: np.array(get_mpi_comm().gather(x)),
                [self.n(), mean, cov, bound, self.acceptance_rate])
        else:
            # Compute and gather means, covs and CL intervals of last m-1 chain fractions
            m = 1 + self.Rminus1_single_split
            cut = int(len(self.collection) / m)
            try:
                Ns = (m - 1) * [cut]
                means = np.array(
                    [self.collection.mean(first=i * cut, last=(i + 1) * cut - 1) for i in
                     range(1, m)])
                covs = np.array(
                    [self.collection.cov(first=i * cut, last=(i + 1) * cut - 1) for i in
                     range(1, m)])
                mcsamples_list = [
                    self.collection._sampled_to_getdist_mcsamples(
                        first=i * cut, last=(i + 1) * cut - 1)
                    for i in range(1, m)]
            except:
                self.log.info("Not enough points in chain to check convergence. "
                              "Waiting for next checkpoint.")
                return
            acceptance_rates = self.acceptance_rate
            try:
                bounds = [np.array(
                    [[mcs.confidence(i, limfrac=self.Rminus1_cl_level / 2., upper=which)
                      for i in range(self.model.prior.d())] for which in [False, True]]).T
                          for mcs in mcsamples_list]
                success_bounds = True
            except:
                bounds = None
                success_bounds = False
        # Compute convergence diagnostics
        if is_main_process():
            self.progress.at[self.i_learn, "N"] = (
                sum(Ns) if more_than_one_process() else self.n())
            self.progress.at[self.i_learn, "timestamp"] = \
                datetime.datetime.now().isoformat()
            acceptance_rate = (
                np.average(acceptance_rates, weights=Ns)
                if more_than_one_process() else acceptance_rates)
            self.log.info(" - Acceptance rate: %.3f" +
                          (" = avg(%r)" % list(acceptance_rates)
                           if more_than_one_process() else ""),
                          acceptance_rate)
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
            # NB: disables warnings from numpy
            prev_err_state = deepcopy(np.geterr())
            np.seterr(divide="ignore")
            diagSinvsqrt = np.diag(np.power(np.diag(cov_of_means), -0.5))
            np.seterr(**prev_err_state)
            corr_of_means = diagSinvsqrt.dot(cov_of_means).dot(diagSinvsqrt)
            norm_mean_of_covs = diagSinvsqrt.dot(mean_of_covs).dot(diagSinvsqrt)
            success = False
            # Cholesky of (normalized) mean of covs and eigvals of Linv*cov_of_means*L
            try:
                L = np.linalg.cholesky(norm_mean_of_covs)
            except np.linalg.LinAlgError:
                self.log.warning(
                    "Negative covariance eigenvectors. "
                    "This may mean that the covariance of the samples does not "
                    "contain enough information at this point. "
                    "Skipping learning a new covmat for now.")
            else:
                Linv = np.linalg.inv(L)
                # Suppress numpy warnings (restored later in this function)
                error_handling = deepcopy(np.geterr())
                np.seterr(all="ignore")
                try:
                    eigvals = np.linalg.eigvalsh(Linv.dot(corr_of_means).dot(Linv.T))
                    success = True
                except np.linalg.LinAlgError:
                    self.log.warning("Could not compute eigenvalues. "
                                     "Skipping learning a new covmat for now.")
                else:
                    Rminus1 = max(np.abs(eigvals))
                    self.progress.at[self.i_learn, "Rminus1"] = Rminus1
                    # For real square matrices, a possible def of the cond number is:
                    condition_number = Rminus1 / min(np.abs(eigvals))
                    self.log.debug(" - Condition number = %g", condition_number)
                    self.log.debug(" - Eigenvalues = %r", eigvals)
                    self.log.info(
                        " - Convergence of means: R-1 = %f after %d accepted steps" % (
                            Rminus1, (sum(Ns) if more_than_one_process() else self.n())) +
                        (" = sum(%r)" % list(Ns) if more_than_one_process() else ""))
                    # Have we converged in means?
                    # (criterion must be fulfilled twice in a row)
                    if max(Rminus1, self.Rminus1_last) < self.Rminus1_stop:
                        # Check the convergence of the bounds of the confidence intervals
                        # Same as R-1, but with the rms deviation from the mean bound
                        # in units of the mean standard deviation of the chains
                        if success_bounds:
                            Rminus1_cl = (np.std(bounds, axis=0).T /
                                          np.sqrt(np.diag(mean_of_covs)))
                            self.log.debug(" - normalized std's of bounds = %r",
                                           Rminus1_cl)
                            Rminus1_cl = np.max(Rminus1_cl)
                            self.progress.at[self.i_learn, "Rminus1_cl"] = Rminus1_cl
                            self.log.info(
                                " - Convergence of bounds: R-1 = %f after %d " % (
                                    Rminus1_cl,
                                    (sum(Ns) if more_than_one_process() else self.n())) +
                                "accepted steps" +
                                (" = sum(%r)" % list(
                                    Ns) if more_than_one_process() else ""))
                            if Rminus1_cl < self.Rminus1_cl_stop:
                                self.converged = True
                                self.log.info("The run has converged!")
                            self._Ns = Ns
                        else:
                            self.log.info("Computation of the bounds was not possible. "
                                          "Waiting until the next converge check.")
                np.seterr(**error_handling)
        else:
            mean_of_covs = np.empty((self.model.prior.d(), self.model.prior.d()))
            success = None
            Rminus1 = None
        # Broadcast and save the convergence status and the last R-1 of means
        success = share_mpi(success)
        if success:
            self.Rminus1_last, self.converged = share_mpi(
                (Rminus1, self.converged) if is_main_process() else None)
            # Do we want to learn a better proposal pdf?
            if self.learn_proposal and not self.converged:
                good_Rminus1 = (self.learn_proposal_Rminus1_max >
                                self.Rminus1_last > self.learn_proposal_Rminus1_min)
                if not good_Rminus1:
                    self.mpi_info("Convergence less than requested for updates: "
                                  "waiting until the next convergence check.")
                    return
                if more_than_one_process():
                    get_mpi_comm().Bcast(mean_of_covs, root=0)
                else:
                    mean_of_covs = covs[0]
                try:
                    self.proposer.set_covariance(mean_of_covs)
                    if is_main_process():
                        self.log.info(" - Updated covariance matrix of proposal pdf.")
                        self.log.debug("%r", mean_of_covs)
                except:
                    if is_main_process():
                        self.log.debug("Updating covariance matrix failed unexpectedly. "
                                       "waiting until next covmat learning attempt.")
        # Save checkpoint info
        self.write_checkpoint()

    def send_error_signal(self):
        """
        Sends an error signal to the other MPI processes.
        """
        for i_rank in range(get_mpi_size()):
            if i_rank != get_mpi_rank():
                get_mpi_comm().isend(True, dest=i_rank, tag=_error_tag)

    def check_error_signal(self):
        """
        Checks if any of the other process has sent an error signal, and fails.

        NB: This behaviour only shows up when running this sampler inside a Python script,
            not when running with `cobaya run` (in that case, the process raising an error
            will call `MPI_ABORT` and kill the rest.
        """
        for i in range(get_mpi_size()):
            if i != get_mpi_rank():
                from mpi4py import MPI
                status = MPI.Status()
                get_mpi_comm().iprobe(i, status=status)
                if status.tag == _error_tag:
                    raise LoggedError(self.log, "Another process failed! Exiting.")

    def do_output(self, date_time):
        self.collection.out_update()
        msg = "Progress @ %s : " % date_time.strftime("%Y-%m-%d %H:%M:%S")
        msg += "%d steps taken" % self.n_steps_raw
        if self.burn_in_left and self.burn_in:  # NB: burn_in_left = 1 even if no burn_in
            msg += " -- still burning in, %d accepted steps left." % self.burn_in_left
        else:
            msg += ", and %d accepted." % self.n()
        self.log.info(msg)

    def write_checkpoint(self):
        if is_main_process() and self.output:
            checkpoint_filename = self.checkpoint_filename()
            self.dump_covmat(self.proposer.get_covariance())
            checkpoint_info = {kinds.sampler: {self.get_name(): dict([
                ("converged", bool(self.converged)),
                ("Rminus1_last", self.Rminus1_last),
                ("burn_in", (self.burn_in.value  # initial: repeat burn-in if not finished
                             if not self.n() and self.burn_in_left else
                             0)),  # to avoid overweighting last point of prev. run
                ("mpi_size", get_mpi_size())])}}
            yaml_dump_file(checkpoint_filename, checkpoint_info, error_if_exists=False)
            if not self.progress.empty:
                with open(self.progress_filename(), "a",
                          encoding="utf-8") as progress_file:
                    progress_file.write(
                        self.progress.tail(1).to_string(header=False, index=False) + "\n")
            self.log.debug("Dumped checkpoint and progress info, and current covmat.")

    # Finally: returning the computed products ###########################################

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the accepted steps.
        """
        products = {"sample": self.collection}
        if is_main_process():
            products["progress"] = self.progress
        return products

    # Class methods
    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        regexps = [output.collection_regexp(name=None)]
        if minimal:
            return [(r, None) for r in regexps]
        regexps += [
            re.compile(output.prefix_regexp_str + re.escape(ext.lstrip(".")) + "$")
            for ext in [_checkpoint_extension, _progress_extension, _covmat_extension]]
        return [(r, None) for r in regexps]

    @classmethod
    def get_version(cls):
        return __version__


# Plotting tool for chain progress #######################################################

def plot_progress(progress, ax=None, index=None,
                  figure_kwargs=empty_dict, legend_kwargs=empty_dict):
    """
    Plots progress of one or more MCMC runs: evolution of R-1
    (for means and c.l. intervals) and acceptance rate.

    Takes a ``progress`` instance (actually a ``pandas.DataFrame``,
    returned as part of the sampler ``products``),
    a chain ``output`` prefix, or a list of any of those
    for plotting progress of several chains at once.

    You can use ``figure_kwargs`` and ``legend_kwargs`` to pass arguments to
    ``matplotlib.pyplot.figure`` and ``matplotlib.pyplot.legend`` respectively.

    Return a subplots axes array. Display with ``matplotlib.pyplot.show()``.

    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2, sharex=True, **figure_kwargs)
    if isinstance(progress, DataFrame):
        pass  # go on to plotting
    elif isinstance(progress, str):
        try:
            if not progress.endswith(_progress_extension):
                progress += _progress_extension
            progress = load_DataFrame(progress)
            # 1-based
            progress.index = np.arange(1, len(progress) + 1)
        except:
            raise ValueError("Cannot load progress file %r" % progress)
    elif hasattr(type(progress), "__iter__"):
        # Assume is a list of progress'es
        for i, p in enumerate(progress):
            plot_progress(p, ax=ax, index=i + 1)
        return ax
    else:
        raise ValueError("Cannot understand progress argument: %r" % progress)
    # Plot!
    tag_pre = "" if index is None else "%d : " % index
    p = ax[0].semilogy(progress.N, progress.Rminus1,
                       "o-", label=tag_pre + "means")
    ax[0].semilogy(progress.N, progress.Rminus1_cl,
                   "x:", c=p[0].get_color(), label=tag_pre + "bounds")
    ax[0].set_ylabel(r"$R-1$")
    ax[0].legend(**legend_kwargs)
    ax[1].plot(progress.N, progress.acceptance_rate, "o-")
    ax[1].set_ylabel(r"acc. rate")
    return ax
