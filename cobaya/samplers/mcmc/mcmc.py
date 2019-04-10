"""
.. module:: samplers.mcmc

:Synopsis: Blocked fast-slow Metropolis sampler (Lewis 1304.4473)
:Author: Antony Lewis (for the CosmoMC sampler, wrapped for cobaya by Jesus Torrado)
"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
import six

# Global
import os
from copy import deepcopy
from itertools import chain
import numpy as np
from collections import OrderedDict as odict
import logging
from pandas import DataFrame

# Local
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi_size, get_mpi_rank, get_mpi_comm
from cobaya.mpi import more_than_one_process, am_single_or_primary_process, sync_processes
from cobaya.collection import Collection, OnePoint
from cobaya.conventions import _weight, _p_proposal, _p_renames, _sampler, _minuslogpost
from cobaya.conventions import _line_width, _path_install
from cobaya.samplers.mcmc.proposal import BlockedProposer
from cobaya.log import HandledException
from cobaya.tools import get_external_function, read_dnumber, compare_params_lists
from cobaya.tools import relative_to_int
from cobaya.yaml import yaml_dump_file
from cobaya.output import OutputDummy

ignore_at_resume = ["burn_in", "callback_function", "callback_every", "max_tries",
                    "check_every", "output_every", "learn_proposal_Rminus1_max",
                    "learn_proposal_Rminus1_max_early", "learn_proposal_Rminus1_min",
                    "max_samples", "Rminus1_stop", "Rminus1_cl_stop",
                    "Rminus1_cl_level", "covmat", "covmat_params", "proposal_scale",
                    "Rminus1_last", "converged"]


class mcmc(Sampler):

    def initialize(self):
        """Initializes the sampler:
        creates the proposal distribution and draws the initial sample."""
        self.log.debug("Initializing")
        for p in [
            "burn_in", "max_tries", "output_every", "check_every", "callback_every"]:
            setattr(
                self, p, read_dnumber(getattr(self, p), self.model.prior.d(), dtype=int))
        if self.callback_every is None:
            self.callback_every = self.check_every
        # Burning-in countdown -- the +1 accounts for the initial point (always accepted)
        self.burn_in_left = self.burn_in + 1
        # Max # checkpoints to wait, in case one process dies without sending MPI_ABORT
        self.been_waiting = 0
        self.max_waiting = max(50, self.max_tries / self.model.prior.d())
        if am_single_or_primary_process():
            if self.resuming and (max(self.mpi_size or 0, 1) != max(get_mpi_size(), 1)):
                self.log.error(
                    "Cannot resume a sample with a different number of chains: "
                    "was %d and now is %d.", max(self.mpi_size, 1), max(get_mpi_size(), 1))
                raise HandledException
        sync_processes()
        if not self.resuming and self.output:
            # Delete previous files (if not "forced", the run would have already failed)
            if ((os.path.abspath(self.covmat_filename()) !=
                 os.path.abspath(str(self.covmat)))):
                try:
                    os.remove(self.covmat_filename())
                except OSError:
                    pass
            # There may be more that chains than expected,
            # if #ranks was bigger in a previous run
            i = 0
            while True:
                i += 1
                collection_filename, _ = self.output.prepare_collection(str(i))
                try:
                    os.remove(collection_filename)
                except OSError:
                    break
        # One collection per MPI process: `name` is the MPI rank + 1
        name = str(1 + (lambda r: r if r is not None else 0)(get_mpi_rank()))
        self.collection = Collection(
            self.model, self.output, name=name, resuming=self.resuming)
        self.current_point = OnePoint(
            self.model, OutputDummy({}), name=name)
        # Use standard MH steps by default
        self.get_new_sample = self.get_new_sample_metropolis
        # Prepare oversampling / dragging if applicable
        self.effective_max_samples = self.max_samples
        if self.oversample and self.drag:
            self.log.error("Choose either oversampling or dragging, not both.")
            raise HandledException
        if self.blocking:
            try:
                speeds, blocks = zip(*list(self.blocking))
                speeds = np.array(speeds)
            except:
                raise HandledException(
                    "Manual blocking not understood. Check documentation.")
            check = compare_params_lists(
                list(chain(*blocks)), list(self.model.parameterization.sampled_params()))
            duplicate = check.pop("duplicate_A", None)
            missing = check.pop("B_but_not_A", None)
            unknown = check.pop("A_but_not_B", None)
            if duplicate:
                self.log.error("Manual blocking: repeated parameters: %r", duplicate)
                raise HandledException
            if missing:
                self.log.error("Manual blocking: missing parameters: %r", missing)
                raise HandledException
            if unknown:
                self.log.error("Manual blocking: unkown parameters: %r", unknown)
                raise HandledException
            int_speeds = self.oversample or self.drag
            if int_speeds:
                speeds = relative_to_int(speeds)
            if (speeds != np.sort(speeds)).all():
                self.log.warn("Manual blocking: speed-blocking non-optimal: sort by "
                              "ascending speed when possible")
        else:
            speeds, blocks = self.model.likelihood._speeds_of_params(
                int_speeds=self.oversample or self.drag, fast_slow=self.drag)
        if self.oversample:
            self.oversampling_factors = speeds
            self.log.info("Oversampling with factors:\n" + "\n".join([
                "   %d : %r" % (f, b) for f, b in zip(self.oversampling_factors, blocks)]))
            self.i_last_slow_block = None
            # No way right now to separate slow and fast
            slow_params = list(self.model.parameterization.sampled_params())
        elif self.drag:
            # For now, no blocking inside either fast or slow: just 2 blocks
            self.i_last_slow_block = 0
            if np.all(speeds == speeds[0]):
                self.log.error("All speeds are equal or too similar: cannot drag! "
                               "Make sure to define accurate likelihoods' speeds.")
                raise HandledException
            # Make the 1st factor 1:
            speeds = [1, speeds[1] / speeds[0]]
            # Target: dragging step taking as long as slow step
            self.drag_interp_steps = self.drag * speeds[1]
            # Per dragging step, the (fast) posterior is evaluated *twice*,
            self.drag_interp_steps /= 2
            self.drag_interp_steps = int(np.round(self.drag_interp_steps))
            fast_params = list(chain(*blocks[1 + self.i_last_slow_block:]))
            # Not too much or too little dragging
            drag_limits = [(int(l) * len(fast_params) if l is not None else l)
                           for l in self.drag_limits]
            if drag_limits[0] is not None and self.drag_interp_steps < drag_limits[0]:
                self.log.warning("Number of dragging steps clipped from below: was not "
                                 "enough to efficiently explore the fast directions -- "
                                 "avoid this limit by decreasing 'drag_limits[0]'.")
                self.drag_interp_steps = drag_limits[0]
            if drag_limits[1] is not None and self.drag_interp_steps > drag_limits[1]:
                self.log.warning("Number of dragging steps clipped from above: "
                                 "excessive, probably inefficient, exploration of the "
                                 "fast directions -- "
                                 "avoid this limit by increasing 'drag_limits[1]'.")
                self.drag_interp_steps = drag_limits[1]
            # Re-scale steps between checkpoint and callback to the slow dimensions only
            slow_params = list(chain(*blocks[:1 + self.i_last_slow_block]))
            self.n_slow = len(slow_params)
            for p in ["check_every", "callback_every"]:
                setattr(self, p, int(getattr(self, p) * self.n_slow / self.model.prior.d()))
            self.log.info(
                "Dragging with oversampling per step:\n" +
                "\n".join(["   %d : %r" % (f, b)
                           for f, b in zip([1, self.drag_interp_steps],
                                           [blocks[0], fast_params])]))
            self.get_new_sample = self.get_new_sample_dragging
        else:
            self.oversampling_factors = [1 for b in blocks]
            slow_params = list(self.model.parameterization.sampled_params())
            self.n_slow = len(slow_params)
        # Turn parameter names into indices
        self.blocks = [
            [list(self.model.parameterization.sampled_params()).index(p) for p in b]
            for b in blocks]
        self.proposer = BlockedProposer(
            self.blocks, oversampling_factors=self.oversampling_factors,
            i_last_slow_block=self.i_last_slow_block, proposal_scale=self.proposal_scale)
        # Build the initial covariance matrix of the proposal, or load from checkpoint
        if self.resuming:
            covmat = np.loadtxt(self.covmat_filename())
            self.log.info("Covariance matrix from checkpoint.")
        else:
            covmat = self.initial_proposal_covmat(slow_params=slow_params)
            self.log.info("Initial covariance matrix.")
        self.log.debug(
            "Sampling with covmat:\n%s",
            DataFrame(covmat, columns=self.model.parameterization.sampled_params(),
                      index=self.model.parameterization.sampled_params()).to_string(
                line_width=_line_width))
        self.proposer.set_covariance(covmat)
        # Prepare callback function
        if self.callback_function is not None:
            self.callback_function_callable = (
                get_external_function(self.callback_function))
        # Useful for getting last points added inside callback function
        self.last_point_callback = 0

    def initial_proposal_covmat(self, slow_params=None):
        """
        Build the initial covariance matrix, using the data provided, in descending order
        of priority:
        1. "covmat" field in the "mcmc" sampler block.
        2. "proposal" field for each parameter.
        3. variance of the reference pdf.
        4. variance of the prior pdf.

        The covariances between parameters when both are present in a covariance matrix
        provided through option 1 are preserved. All other covariances are assumed 0.
        """
        params_infos = self.model.parameterization.sampled_params_info()
        covmat = np.diag([np.nan] * len(params_infos))
        # Try to generate it automatically
        if isinstance(self.covmat, six.string_types) and self.covmat.lower() == "auto":
            slow_params_info = {
                p: info for p, info in params_infos.items() if p in slow_params}
            auto_covmat = self.model.likelihood._get_auto_covmat(slow_params_info)
            if auto_covmat:
                self.covmat = os.path.join(auto_covmat["folder"], auto_covmat["name"])
                self.log.info("Covariance matrix selected automatically: %s", self.covmat)
            else:
                self.covmat = None
                self.log.info("Could not automatically find a good covmat. "
                              "Will generate from parameter info (proposal and prior).")
        # If given, load and test the covariance matrix
        if isinstance(self.covmat, six.string_types):
            covmat_pre = "{%s}" % _path_install
            if self.covmat.startswith(covmat_pre):
                self.covmat = self.covmat.format(
                    **{_path_install: self.path_install}).replace("/", os.sep)
            try:
                with open(self.covmat, "r") as file_covmat:
                    header = file_covmat.readline()
                loaded_covmat = np.loadtxt(self.covmat)
            except TypeError:
                self.log.error("The property 'covmat' must be a file name,"
                               "but it's '%s'.", str(self.covmat))
                raise HandledException
            except IOError:
                self.log.error("Can't open covmat file '%s'.", self.covmat)
                raise HandledException
            if header[0] != "#":
                self.log.error(
                    "The first line of the covmat file '%s' "
                    "must be one list of parameter names separated by spaces "
                    "and staring with '#', and the rest must be a square matrix, "
                    "with one row per line.", self.covmat)
                raise HandledException
            loaded_params = header.strip("#").strip().split()
        elif hasattr(self.covmat, "__getitem__"):
            if not self.covmat_params:
                self.log.error(
                    "If a covariance matrix is passed as a numpy array, "
                    "you also need to pass the parameters it corresponds to "
                    "via 'covmat_params: [name1, name2, ...]'.")
                raise HandledException
            loaded_params = self.covmat_params
            loaded_covmat = self.covmat
        if self.covmat is not None:
            if len(loaded_params) != len(set(loaded_params)):
                self.log.error(
                    "There are duplicated parameters in the header of the "
                    "covmat file '%s' ", self.covmat)
                raise HandledException
            if len(loaded_params) != loaded_covmat.shape[0]:
                self.log.error(
                    "The number of parameters in the header of '%s' and the "
                    "dimensions of the matrix do not coincide.", self.covmat)
                raise HandledException
            if not (np.allclose(loaded_covmat.T, loaded_covmat) and
                    np.all(np.linalg.eigvals(loaded_covmat) > 0)):
                self.log.error(
                    "The covmat loaded from '%s' is not a positive-definite, "
                    "symmetric square matrix.", self.covmat)
                raise HandledException
            # Fill with parameters in the loaded covmat
            renames = [[p] + np.atleast_1d(v.get(_p_renames, [])).tolist()
                       for p, v in params_infos.items()]
            renames = odict([[a[0], a] for a in renames])
            indices_used, indices_sampler = zip(*[
                [loaded_params.index(p),
                 [list(params_infos).index(q) for q, a in renames.items() if p in a]]
                for p in loaded_params])
            if not any(indices_sampler):
                self.log.error(
                    "A proposal covariance matrix has been loaded, but none of its "
                    "parameters are actually sampled here. Maybe a mismatch between"
                    " parameter names in the covariance matrix and the input file?")
                raise HandledException
            indices_used, indices_sampler = zip(*[
                [i, j] for i, j in zip(indices_used, indices_sampler) if j])
            if any(len(j) - 1 for j in indices_sampler):
                first = next(j for j in indices_sampler if len(j) > 1)
                self.log.error(
                    "The parameters %s have duplicated aliases. Can't assign them an "
                    "element of the covariance matrix unambiguously.",
                    ", ".join([list(params_infos)[i] for i in first]))
                raise HandledException
            indices_sampler = list(chain(*indices_sampler))
            covmat[np.ix_(indices_sampler, indices_sampler)] = (
                loaded_covmat[np.ix_(indices_used, indices_used)])
            self.log.info(
                "Covariance matrix loaded for params %r",
                [list(params_infos)[i] for i in indices_sampler])
            missing_params = set(params_infos).difference(
                set([list(params_infos)[i] for i in indices_sampler]))
            if missing_params:
                self.log.info(
                    "Missing proposal covariance for params %r",
                    [p for p in self.model.parameterization.sampled_params()
                     if p in missing_params])
            else:
                self.log.info("All parameters' covariance loaded from given covmat.")
        # Fill gaps with "proposal" property, if present, otherwise ref (or prior)
        where_nan = np.isnan(covmat.diagonal())
        if np.any(where_nan):
            covmat[where_nan, where_nan] = np.array(
                [info.get(_p_proposal, np.nan) ** 2
                 for info in params_infos.values()])[where_nan]
            # we want to start learning the covmat earlier
            self.log.info("Covariance matrix " +
                          ("not present" if np.all(where_nan) else "not complete") + ". "
                                                                                     "We will start learning the covariance of the proposal earlier:"
                                                                                     " R-1 = %g (was %g).",
                          self.learn_proposal_Rminus1_max_early,
                          self.learn_proposal_Rminus1_max)
            self.learn_proposal_Rminus1_max = self.learn_proposal_Rminus1_max_early
        where_nan = np.isnan(covmat.diagonal())
        if np.any(where_nan):
            covmat[where_nan, where_nan] = (
                self.model.prior.reference_covmat().diagonal()[where_nan])
        assert not np.any(np.isnan(covmat))
        return covmat

    def run(self):
        """
        Runs the sampler.
        """
        # Get first point, to be discarded -- not possible to determine its weight
        # Still, we need to compute derived parameters, since, as the proposal "blocked",
        # we may be saving the initial state of some block.
        # NB: if resuming but nothing was written (burn-in not finished): re-start
        self.log.info("Initial point:")
        if self.resuming and self.collection.n():
            initial_point = (self.collection[self.collection.sampled_params]
                .ix[self.collection.n() - 1]).values.copy()
            logpost = -(self.collection[_minuslogpost]
                        .ix[self.collection.n() - 1].copy())
            logpriors = -(self.collection[self.collection.prior_names]
                          .ix[self.collection.n() - 1].copy())
            loglikes = -0.5 * (self.collection[self.collection.chi2_names]
                               .ix[self.collection.n() - 1].copy())
            derived = (self.collection[self.collection.derived_params]
                       .ix[self.collection.n() - 1].values.copy())
        else:
            initial_point = self.model.prior.reference(max_tries=self.max_tries)
            logpost, logpriors, loglikes, derived = self.model.logposterior(initial_point)
        self.current_point.add(initial_point, derived=derived, logpost=logpost,
                               logpriors=logpriors, loglikes=loglikes)
        self.log.info("\n%s", self.current_point.data.to_string(
            index=False, line_width=_line_width))
        # Initial dummy checkpoint (needed when 1st checkpoint not reached in prev. run)
        self.write_checkpoint()
        # Main loop!
        self.log.info("Sampling!" + (
            " (NB: nothing will be printed until %d burn-in samples " % self.burn_in +
            "have been obtained)" if self.burn_in else ""))
        while self.n() < self.effective_max_samples and not self.converged:
            self.get_new_sample()
            # Callback function
            if (hasattr(self, "callback_function_callable") and
                    not (max(self.n(), 1) % self.callback_every) and
                    self.current_point[_weight] == 1):
                self.callback_function_callable(self)
                self.last_point_callback = self.collection.n()
            # Checking convergence and (optionally) learning the covmat of the proposal
            if self.check_all_ready():
                self.check_convergence_and_learn_proposal()
            if self.n() == self.effective_max_samples:
                self.log.info("Reached maximum number of accepted steps allowed. "
                              "Stopping.")
        # Make sure the last batch of samples ( < output_every ) are written
        self.collection._out_update()
        if more_than_one_process():
            Ns = (lambda x: np.array(get_mpi_comm().gather(x)))(self.n())
        else:
            Ns = [self.n()]
        if am_single_or_primary_process():
            self.log.info("Sampling complete after %d accepted steps.", sum(Ns))

    def n(self, burn_in=False):
        """
        Returns the total number of steps taken, including or not burn-in steps depending
        on the value of the `burn_in` keyword.
        """
        return self.collection.n() + (
            0 if not burn_in else self.burn_in - self.burn_in_left + 1)

    def get_new_sample_metropolis(self):
        """
        Draws a new trial point from the proposal pdf and checks whether it is accepted:
        if it is accepted, it saves the old one into the collection and sets the new one
        as the current state; if it is rejected increases the weight of the current state
        by 1.

        Returns:
           ``True`` for an accepted step, ``False`` for a rejected one.
        """
        trial = deepcopy(self.current_point[self.model.parameterization._sampled])
        self.proposer.get_proposal(trial)
        logpost_trial, logprior_trial, loglikes_trial, derived = self.model.logposterior(trial)
        accept = self.metropolis_accept(logpost_trial,
                                        -self.current_point["minuslogpost"])
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
        start_slow_point = self.current_point[self.model.parameterization._sampled]
        start_slow_logpost = -self.current_point["minuslogpost"]
        end_slow_point = deepcopy(start_slow_point)
        self.proposer.get_proposal_slow(end_slow_point)
        self.log.debug("Proposed slow end-point: %r", end_slow_point)
        # Save derived parameters of delta_slow jump, in case I reject all the dragging
        # steps but accept the move in the slow direction only
        end_slow_logpost, end_slow_logprior, end_slow_loglikes, derived = (
            self.model.logposterior(end_slow_point))
        if end_slow_logpost == -np.inf:
            self.current_point.increase_weight(1)
            return False
        # trackers of the dragging
        current_start_point = start_slow_point
        current_end_point = end_slow_point
        current_start_logpost = start_slow_logpost
        current_end_logpost = end_slow_logpost
        current_end_logprior = end_slow_logprior
        current_end_loglikes = end_slow_loglikes
        # accumulators for the "dragging" probabilities to be metropolist-tested
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
            proposal_start_point = deepcopy(current_start_point)
            proposal_start_point += delta_fast
            proposal_end_point = deepcopy(current_end_point)
            proposal_end_point += delta_fast
            # get the new extremes for the interpolated probability
            # (reject if any of them = -inf; avoid evaluating both if just one fails)
            # Force the computation of the (slow blocks) derived params at the starting
            # point, but discard them, since they contain the starting point's fast ones,
            # not used later -- save the end point's ones.
            proposal_start_logpost = self.model.logposterior(proposal_start_point)[0]
            proposal_end_logpost, proposal_end_logprior, \
            proposal_end_loglikes, derived_proposal_end = (
                self.model.logposterior(proposal_end_point)
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
            self.log.debug("Dragging step: %s", ("accepted" if accept_drag else "rejected"))
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
                                 logpost_trial=None, logprior_trial=None, loglikes_trial=None):
        """Processes the acceptance/rejection of the new point."""
        if accept_state:
            # add the old point to the collection (if not burning or initial point)
            if self.burn_in_left <= 0:
                self.current_point.add_to_collection(self.collection)
                self.log.debug("New sample, #%d: \n   %r", self.n(), self.current_point)
                if self.n() % self.output_every == 0:
                    self.collection._out_update()
            else:
                self.burn_in_left -= 1
                self.log.debug("Burn-in sample:\n   %r", self.current_point)
                if self.burn_in_left == 0 and self.burn_in:
                    self.log.info("Finished burn-in phase: discarded %d accepted steps.",
                                  self.burn_in)
            # set the new point as the current one, with weight one
            self.current_point.add(trial, derived=derived, weight=1, logpost=logpost_trial,
                                   logpriors=logprior_trial, loglikes=loglikes_trial)
        else:  # not accepted
            self.current_point.increase_weight(1)
            # Failure criterion: chain stuck! (but be more permissive during burn_in)
            max_tries_now = self.max_tries * (1 + (10 - 1) * np.sign(self.burn_in_left))
            if self.current_point[_weight] > max_tries_now:
                self.collection._out_update()
                self.log.error(
                    "The chain has been stuck for %d attempts. Stopping sampling. "
                    "If this has happened often, try improving your "
                    "reference point/distribution. Alternatively (though not advisable) "
                    "make 'max_tries: np.inf' (or 'max_tries: .inf' in yaml)",
                    max_tries_now)
                raise HandledException

    # Functions to check convergence and learn the covariance of the proposal distribution

    def check_all_ready(self):
        """
        Checks if the chain(s) is(/are) ready to check convergence and, if requested,
        learn a new covariance matrix for the proposal distribution.
        """
        msg_ready = ("Ready to check convergence" +
                     (" and learn a new proposal covmat" if self.learn_proposal else ""))
        # If *just* (weight==1) got ready to check+learn
        if (self.n() > 0 and self.current_point[_weight] == 1 and
                not (self.n() % self.check_every)):
            self.log.info("Checkpoint: %d samples accepted.", self.n())
            if more_than_one_process():
                self.been_waiting += 1
                if self.been_waiting > self.max_waiting:
                    self.log.error(
                        "Waiting for too long for all chains to be ready. "
                        "Maybe one of them is stuck or died unexpectedly?")
                    raise HandledException
            self.model.dump_timing()
            # If not MPI size > 1, we are ready
            if not more_than_one_process():
                if msg_ready:
                    self.log.info(msg_ready)
                return True
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
            if more_than_one_process() and am_single_or_primary_process():
                self.log.info("All chains are r" + msg_ready[1:])
            delattr(self, "req")
            self.been_waiting = 0
            # Just in case, a barrier here
            sync_processes()
            return True
        return False

    def check_convergence_and_learn_proposal(self):
        """
        Checks the convergence of the sampling process (MPI only), and, if requested,
        learns a new covariance matrix for the proposal distribution from the covariance
        of the last samples.
        """
        if more_than_one_process():
            # Compute and gather means, covs and CL intervals of last half of chains
            mean = self.collection.mean(first=int(self.n() / 2))
            cov = self.collection.cov(first=int(self.n() / 2))
            mcsamples = self.collection._sampled_to_getdist_mcsamples(first=int(self.n() / 2))
            try:
                bound = np.array([[
                    mcsamples.confidence(i, limfrac=self.Rminus1_cl_level / 2., upper=which)
                    for i in range(self.model.prior.d())] for which in [False, True]]).T
                success_bounds = True
            except:
                bound = None
                success_bounds = False
            Ns, means, covs, bounds = map(
                lambda x: np.array(get_mpi_comm().gather(x)),
                [self.n(), mean, cov, bound])
        else:
            # Compute and gather means, covs and CL intervals of last m-1 chain fractions
            m = 1 + self.Rminus1_single_split
            cut = int(self.collection.n() / m)
            if cut <= 1:
                self.log.error("Not enough points in chain to check convergence. "
                               "Increase `check_every` or reduce `Rminus1_single_split`.")
                raise HandledException
            Ns = (m - 1) * [cut]
            means = np.array(
                [self.collection.mean(first=i * cut, last=(i + 1) * cut - 1) for i in range(1, m)])
            covs = np.array(
                [self.collection.cov(first=i * cut, last=(i + 1) * cut - 1) for i in range(1, m)])
            # No logging of warnings temporarily, so getdist won't complain unnecessarily
            logging.disable(logging.WARNING)
            mcsampleses = [
                self.collection._sampled_to_getdist_mcsamples(
                    first=i * cut, last=(i + 1) * cut - 1)
                for i in range(1, m)]
            logging.disable(logging.NOTSET)
            try:
                bounds = [np.array(
                    [[mcs.confidence(i, limfrac=self.Rminus1_cl_level / 2., upper=which)
                      for i in range(self.model.prior.d())] for which in [False, True]]).T
                          for mcs in mcsampleses]
                success_bounds = True
            except:
                bounds = None
                success_bounds = False
        # Compute convergence diagnostics
        if am_single_or_primary_process():
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
            diagSinvsqrt = np.diag(np.power(np.diag(cov_of_means), -0.5))
            corr_of_means = diagSinvsqrt.dot(cov_of_means).dot(diagSinvsqrt)
            norm_mean_of_covs = diagSinvsqrt.dot(mean_of_covs).dot(diagSinvsqrt)
            # Cholesky of (normalized) mean of covs and eigvals of Linv*cov_of_means*L
            try:
                L = np.linalg.cholesky(norm_mean_of_covs)
            except np.linalg.LinAlgError:
                self.log.warning(
                    "Negative covariance eigenvectors. "
                    "This may mean that the covariance of the samples does not "
                    "contain enough information at this point. "
                    "Skipping this checkpoint")
                success = False
            else:
                Linv = np.linalg.inv(L)
                try:
                    eigvals = np.linalg.eigvalsh(Linv.dot(corr_of_means).dot(Linv.T))
                    success = True
                except np.linalg.LinAlgError:
                    self.log.warning("Could not compute eigenvalues. "
                                     "Skipping this checkpoint.")
                    success = False
                if success:
                    Rminus1 = max(np.abs(eigvals))
                    # For real square matrices, a possible def of the cond number is:
                    condition_number = Rminus1 / min(np.abs(eigvals))
                    self.log.debug("Condition number = %g", condition_number)
                    self.log.debug("Eigenvalues = %r", eigvals)
                    self.log.info(
                        "Convergence of means: R-1 = %f after %d accepted steps" % (
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
                            self.log.debug("normalized std's of bounds = %r", Rminus1_cl)
                            self.log.info(
                                "Convergence of bounds: R-1 = %f after %d " % (
                                    np.max(Rminus1_cl),
                                    (sum(Ns) if more_than_one_process() else self.n())) +
                                "accepted steps" +
                                (" = sum(%r)" % list(Ns) if more_than_one_process() else ""))
                            if np.max(Rminus1_cl) < self.Rminus1_cl_stop:
                                self.converged = True
                                self.log.info("The run has converged!")
                            self._Ns = Ns
                        else:
                            self.log.info("Computation of the bounds was not possible. "
                                          "Waiting until the next checkpoint")
        if more_than_one_process():
            # Broadcast and save the convergence status and the last R-1 of means
            success = get_mpi_comm().bcast(
                (success if am_single_or_primary_process() else None), root=0)
            if success:
                self.Rminus1_last = get_mpi_comm().bcast(
                    (Rminus1 if am_single_or_primary_process() else None), root=0)
                self.converged = get_mpi_comm().bcast(
                    (self.converged if am_single_or_primary_process() else None), root=0)
        else:
            if success:
                self.Rminus1_last = Rminus1
        # Do we want to learn a better proposal pdf?
        if self.learn_proposal and not self.converged and success:
            good_Rminus1 = (self.learn_proposal_Rminus1_max >
                            self.Rminus1_last > self.learn_proposal_Rminus1_min)
            if not good_Rminus1:
                if am_single_or_primary_process():
                    self.log.info("Bad convergence statistics: "
                                  "waiting until the next checkpoint.")
                return
            if more_than_one_process():
                if not am_single_or_primary_process():
                    mean_of_covs = np.empty((self.model.prior.d(), self.model.prior.d()))
                get_mpi_comm().Bcast(mean_of_covs, root=0)
            else:
                mean_of_covs = covs[0]
            try:
                self.proposer.set_covariance(mean_of_covs)
            except:
                self.log.debug("Updating covariance matrix failed unexpectedly. "
                               "waiting until next checkpoint.")
            if am_single_or_primary_process():
                self.log.info("Updated covariance matrix of proposal pdf.")
                self.log.debug("%r", mean_of_covs)
        # Save checkpoint info
        self.write_checkpoint()

    def write_checkpoint(self):
        if am_single_or_primary_process() and self.output:
            checkpoint_filename = self.checkpoint_filename()
            covmat_filename = self.covmat_filename()
            np.savetxt(covmat_filename, self.proposer.get_covariance(), header=" ".join(
                list(self.model.parameterization.sampled_params())))
            checkpoint_info = {_sampler: {self.name: odict([
                ["converged", bool(self.converged)],
                ["Rminus1_last", self.Rminus1_last],
                ["proposal_scale", self.proposer.get_scale()],
                ["blocks", self.blocks],
                ["oversampling_factors", self.oversampling_factors],
                ["i_last_slow_block", self.i_last_slow_block],
                ["burn_in", (self.burn_in  # initial: repeat burn-in if not finished
                             if not self.n() and self.burn_in_left else
                             "d")],  # to avoid overweighting last point of prev. run
                ["mpi_size", get_mpi_size()]])}}
            yaml_dump_file(checkpoint_filename, checkpoint_info, error_if_exists=False)
            self.log.debug("Dumped checkpoint info and current covmat.")

    # Finally: returning the computed products ###########################################

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the accepted steps.
        """
        return {"sample": self.collection}
