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
from copy import deepcopy
from itertools import chain
import numpy as np
import logging

# Local
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi, get_mpi_size, get_mpi_rank, get_mpi_comm
from cobaya.collection import Collection, OnePoint
from cobaya.conventions import _weight, _p_proposal
from cobaya.samplers.mcmc.proposal import BlockedProposer
from cobaya.log import HandledException
from cobaya.tools import get_external_function


class mcmc(Sampler):

    def initialise(self):
        """Initialises the sampler:
        creates the proposal distribution and draws the initial sample."""
        self.log.info("Initializing")
        # Burning-in countdown -- the +1 accounts for the initial point (always accepted)
        self.burn_in_left = self.burn_in + 1
        # One collection per MPI process: `name` is the MPI rank + 1
        name = str(1 + (lambda r: r if r is not None else 0)(get_mpi_rank()))
        self.collection = Collection(
            self.parametrization, self.likelihood, self.output, name=name)
        self.current_point = OnePoint(
            self.parametrization, self.likelihood, self.output, name=name)
        # Use the standard steps by default
        self.get_new_sample = self.get_new_sample_metropolis
        # Prepare oversampling / fast-dragging if applicable
        self.effective_max_samples = self.max_samples
        if self.oversample and self.drag:
            self.log.error("Choose either oversampling or fast-dragging, not both.")
            raise HandledException
#        if (self.oversample or self.drag) and len(set(factors)) == 1:
#            self.log.error("All block speeds are similar: "
#                           "no dragging or oversampling possible.")
#            raise HandledException
        if self.oversample:
            factors, blocks = self.likelihood.speeds_of_params(oversampling_factors=True)
            self.oversampling_factors = factors
            # WIP: actually, we would have to re-normalise to the dimension of the blocks.
            self.log.info(
                "Oversampling with factors:\n" +
                "\n".join(["   %d : %r"%(f,b)
                           for f,b in zip(self.oversampling_factors, blocks)]))
            # WIP: useless until likelihoods have STATES!
            self.log.error("Sorry, oversampling is WIP")
            raise HandledException
        elif self.drag:
            # WIP: for now, can only separate between theory and likelihoods
            # until likelihoods have states
            if not self.likelihood.theory:
                self.log.error("WIP: dragging disabled for now when no theory code present.")
                raise HandledException
#            if self.max_speed_slow < min(speeds) or self.max_speed_slow >= max(speeds):
#                self.log.error("The maximum speed considered slow, `max_speed_slow`, must be "
#                          "%g <= `max_speed_slow < %g, and is %g",
#                          min(speeds), max(speeds), self.max_speed_slow)
#                raise HandledException
            speeds, blocks = self.likelihood.speeds_of_params(
                int_speeds=True, fast_slow=True)
            if np.all(speeds==speeds[0]):
                self.log.error("All speeds are equal: cannot drag! Make sure to define, "
                               "especially, the speed of the fastest likelihoods.")
            self.i_last_slow_block = 0  # just theory can be slow for now
            fast_params = list(chain(*blocks[1+self.i_last_slow_block:]))
            self.n_slow = sum(len(blocks[i]) for i in range(1+self.i_last_slow_block))
            from cobaya.conventions import _overhead
            self.drag_interp_steps = int(self.drag*np.round(min(speeds[1:])/speeds[0]))
            self.log.info(
                "Dragging with oversampling per step:\n" +
                "\n".join(["   %d : %r"%(f,b)
                           for f,b in zip([1, self.drag_interp_steps],
                                          [blocks[0], fast_params])]))
            self.get_new_sample = self.get_new_sample_dragging
        else:
            _, blocks = self.likelihood.speeds_of_params()
            self.oversampling_factors = [1 for b in blocks]
            self.n_slow = len(self.parametrization.sampled_params())
        # Turn parameter names into indices
        blocks = [[list(self.parametrization.sampled_params().keys()).index(p) for p in b]
                  for b in blocks]
        self.proposer = BlockedProposer(
            blocks, oversampling_factors=getattr(self, "oversampling_factors", None),
            i_last_slow_block=getattr(self, "i_last_slow_block", None),
            propose_scale=self.propose_scale)
        # Build the initial covariance matrix of the proposal
        covmat = self.initial_proposal_covmat()
        self.log.info("Sampling with covariance matrix:")
        self.log.info("%r", covmat)
        self.proposer.set_covariance(covmat)
        # Prepare callback function
        if self.callback_function is not None:
            self.callback_function_callable = (
                get_external_function(self.callback_function))

    def initial_proposal_covmat(self):
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
        params, params_infos = zip(*self.parametrization.sampled_params().items())
        covmat = np.diag([np.nan]*len(params))
        # If given, load and test the covariance matrix
        if isinstance(self.covmat, six.string_types):
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
            loaded_params_used = set(loaded_params).intersection(set(params))
            if not loaded_params_used:
                self.log.error(
                    "A proposal covariance matrix has been loaded, but none of its "
                    "parameters are actually sampled here. Maybe a mismatch between"
                    " parameter names in the covariance matrix and the input file?")
                raise HandledException
            indices_used, indices_sampler = np.array(
                [[loaded_params.index(p),params.index(p)]
                 for p in loaded_params if p in loaded_params_used]).T
            covmat[np.ix_(indices_sampler,indices_sampler)] = (
                loaded_covmat[np.ix_(indices_used,indices_used)])
            self.log.info(
                "Covariance matrix loaded for params %r",
                [p for p in self.parametrization.sampled_params()
                 if p in loaded_params_used])
            missing_params = set(params).difference(set(loaded_params))
            if missing_params:
                self.log.info(
                    "Missing proposal covarince for params %r",
                    [p for p in self.parametrization.sampled_params()
                     if p in missing_params])
            else:
                self.log.info("All parameters' covariance loaded from given covmat.")
        # Fill gaps with "proposal" property, if present, otherwise ref (or prior)
        where_nan = np.isnan(covmat.diagonal())
        if np.any(where_nan):
            covmat[where_nan, where_nan] = np.array(
                [info.get(_p_proposal, np.nan)**2 for info in params_infos])[where_nan]
            # we want to start learning the covmat earlier
            self.log.info("Covariance matrix " +
                     ("not present" if np.all(where_nan) else "not complete") + ". "
                     "We will start learning the covariance of the proposal earlier: "
                     "R-1 = %g (was %g).", self.learn_proposal_Rminus1_max_early,
                     self.learn_proposal_Rminus1_max)
            self.learn_proposal_Rminus1_max = self.learn_proposal_Rminus1_max_early
        where_nan = np.isnan(covmat.diagonal())
        if np.any(where_nan):
            covmat[where_nan, where_nan] = (
                self.prior.reference_covmat().diagonal()[where_nan])
        assert not np.any(np.isnan(covmat))
        return covmat

    def run(self):
        """
        Runs the sampler.
        """
        # Get first point, to be discarded -- not possible to determine its weight
        # Still, we need to compute derived parameters, since, as the proposal "blocked",
        # we may be saving the initial state of some block.
        initial_point = self.prior.reference(max_tries=self.max_tries)
        logpost, _, _, derived = self.logposterior(initial_point)
        self.current_point.add(initial_point, derived=derived, logpost=logpost)
        self.log.info("Initial point:\n %r ",self.current_point)
        # Main loop!
        self.converged = False
        self.log.info("Sampling!" + (
            "(NB: nothing will be printed until %d burn-in samples "%self.burn_in +
            "have been obtained)" if self.burn_in else ""))
        while self.n() < self.effective_max_samples and not self.converged:
            self.get_new_sample()
            # Callback function
            if (hasattr(self, "callback_function_callable") and
                    not(max(self.n(),1)%self.callback_every) and
                    self.current_point[_weight] == 1):
                self.callback_function_callable(self)
            # Checking convergence and (optionally) learning the covmat of the proposal
            if self.check_all_ready():
                self.check_convergence_and_learn_proposal()
        # Make sure the last batch of samples ( < output_every ) are written
        self.collection.out_update()
        if not get_mpi_rank():
            self.log.info("Sampling complete after %d accepted steps.", self.n())

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
        trial = deepcopy(self.current_point[self.parametrization.sampled_params()])
        self.proposer.get_proposal(trial)
        logpost_trial, logprior_trial, logliks_trial, derived = self.logposterior(trial)
        accept = self.metropolis_accept(logpost_trial,
                                        -self.current_point["minuslogpost"])
        self.process_accept_or_reject(accept, trial, derived,
                                      logpost_trial, logprior_trial, logliks_trial)
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
        start_slow_point = self.current_point[self.parametrization.sampled_params()]
        start_slow_logpost = -self.current_point["minuslogpost"]
        end_slow_point = deepcopy(start_slow_point)
        self.proposer.get_proposal_slow(end_slow_point)
        self.log.debug("Proposed slow end-point: %r", end_slow_point)
        # Save derived paramters of delta_slow jump, in case I reject all the dragging
        # steps but accept the move in the slow direction only
        end_slow_logpost, end_slow_logprior, end_slow_logliks, derived = (
            self.logposterior(end_slow_point))
        if end_slow_logpost == -np.inf:
            self.current_point.increase_weight(1)
            return False
        # trackers of the dragging
        current_start_point = start_slow_point
        current_end_point   = end_slow_point
        current_start_logpost = start_slow_logpost
        current_end_logpost   = end_slow_logpost
        current_end_logprior  = end_slow_logprior
        current_end_logliks   = end_slow_logliks
        # accumulators for the "dragging" probabilities to be metropolist-tested
        # at the end of the interpolation
        start_drag_logpost_acc = start_slow_logpost
        end_drag_logpost_acc = end_slow_logpost
        # start dragging
        for i_step in range(1, 1+self.drag_interp_steps):
            self.log.debug("Dragging step: %d", i_step)
            # take a step in the fast direction in both slow extremes
            delta_fast = np.zeros(len(current_start_point))
            self.proposer.get_proposal_fast(delta_fast)
            self.log.debug("Proposed fast step delta: %r", delta_fast)
            proposal_start_point  = deepcopy(current_start_point)
            proposal_start_point += delta_fast
            proposal_end_point    = deepcopy(current_end_point)
            proposal_end_point   += delta_fast
            # get the new extremes for the interpolated probability
            # (reject if any of them = -inf; avoid evaluating both if just one fails)
            # Force the computation of the (slow blocks) derived params at the starting
            # point, but discard them, since they contain the starting point's fast ones,
            # not used later -- save the end point's ones.
            proposal_start_logpost = self.logposterior(proposal_start_point)[0]
            proposal_end_logpost, proposal_end_logprior, \
                proposal_end_logliks, derived_proposal_end = (
                    self.logposterior(proposal_end_point)
                    if proposal_start_logpost > -np.inf
                    else (-np.inf, None, [], []))
            if proposal_start_logpost > -np.inf and proposal_end_logpost > -np.inf:
                # create the interpolated probability and do a Metropolis test
                frac = i_step / (1 + self.drag_interp_steps)
                proposal_interp_logpost = ((1-frac)*proposal_start_logpost
                                             +frac *proposal_end_logpost)
                current_interp_logpost  = ((1-frac)*current_start_logpost
                                             +frac *current_end_logpost)
                accept_drag = self.metropolis_accept(proposal_interp_logpost,
                                                     current_interp_logpost)
            else:
                accept_drag = False
            self.log.debug("Dragging step: %s", ("accepted" if accept_drag else "rejected"))
            # If the dragging step was accepted, do the drag
            if accept_drag:
                current_start_point   = proposal_start_point
                current_start_logpost = proposal_start_logpost
                current_end_point     = proposal_end_point
                current_end_logpost   = proposal_end_logpost
                current_end_logprior  = proposal_end_logprior
                current_end_logliks   = proposal_end_logliks
                derived = derived_proposal_end
            # In any case, update the dragging probability for the final metropolis test
            start_drag_logpost_acc += current_start_logpost
            end_drag_logpost_acc   += current_end_logpost
        # Test for the TOTAL step
        accept = self.metropolis_accept(end_drag_logpost_acc/self.drag_interp_steps,
                                        start_drag_logpost_acc/self.drag_interp_steps)
        self.process_accept_or_reject(
            accept, current_end_point, derived,
            current_end_logpost, current_end_logprior, current_end_logliks)
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
                                 logpost_trial=None, logprior_trial=None, logliks_trial=None):
        """Processes the acceptance/rejection of the new point."""
        if accept_state:
            # add the old point to the collection (if not burning or initial point)
            if self.burn_in_left <= 0:
                self.current_point.add_to_collection(self.collection)
                self.log.debug("New sample, #%d: \n   %r", self.n(), self.current_point)
                if self.n()%self.output_every == 0:
                    self.collection.out_update()
            else:
                self.burn_in_left -= 1
                self.log.debug("Burn-in sample:\n   %r", self.current_point)
                if self.burn_in_left == 0:
                    self.log.info("Finished burn-in phase: discarded %d accepted steps.",
                                  self.burn_in)
            # set the new point as the current one, with weight one
            self.current_point.add(trial, derived=derived, weight=1, logpost=logpost_trial,
                                   logprior=logprior_trial, logliks=logliks_trial)
        else:  # not accepted
            self.current_point.increase_weight(1)
            # Failure criterion: chain stuck!
            if self.current_point[_weight] > self.max_tries:
                self.collection.out_update()
                self.log.error(
                    "The chain has been stuck for %d attempts. "
                    "Stopping sampling. If this has happened often, try improving your"
                    " reference point/distribution.", self.max_tries)
                raise HandledException

    # Functions to check convergence and learn the covariance of the proposal distribution

    def check_all_ready(self):
        """
        Checks if the chain(s) is(/are) ready to check convergence and, if requested,
        learn a new covariance matrix for the proposal distribution.
        """
        msg_ready = (("Ready to" if get_mpi() or self.learn_proposal else "") +
                     (" check convergence" if get_mpi() else "") +
                     (" and" if get_mpi() and self.learn_proposal else "") +
                     (" learn a new proposal covmat" if self.learn_proposal else ""))
        # If *just* (weight==1) got ready to check+learn
        if (    self.n() > 0 and self.current_point[_weight] == 1 and
                not (self.n()%(self.check_every_dimension_times*self.n_slow))):
            self.log.info("Checkpoint: %d samples accepted.", self.n())
            # If not MPI, we are ready
            if not get_mpi():
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
            if get_mpi_rank() == 0:
                self.log.info("All chains are r"+msg_ready[1:])
            delattr(self, "req")
            # Just in case, a barrier here
            get_mpi_comm().barrier()
            return True
        return False

    def check_convergence_and_learn_proposal(self):
        """
        Checks the convergence of the sampling process (MPI only), and, if requested,
        learns a new covariance matrix for the proposal distribution from the covariance
        of the last samples.
        """
        # Compute and gather means, covs and CL intervals of last half of chains
        mean = self.collection.mean(first=int(self.n()/2))
        cov = self.collection.cov(first=int(self.n()/2))
        # No logging of warnings temporarily, so getdist won't complain innecessarily
        logging.disable(logging.WARNING)
        mcsamples = self.collection.sampled_to_getdist_mcsamples(first=int(self.n()/2))
        logging.disable(logging.NOTSET)
        bound = np.array(
            [[mcsamples.confidence(i, limfrac=self.Rminus1_cl_level/2., upper=which)
              for i in range(self.prior.d())] for which in [False, True]]).T
        Ns, means, covs, bounds = map(
            lambda x: np.array((get_mpi_comm().gather(x) if get_mpi() else [x])),
            [self.n(), mean, cov, bound])
        # Compute convergence diagnostics
        if get_mpi():
            if get_mpi_rank() == 0:
                # "Within" or "W" term -- our "units" for assessing convergence
                # and our prospective new covariance matrix
                mean_of_covs = np.average(covs, weights=Ns, axis=0)
                # "Between" or "B" term
                # We don't weight with the number of samples in the chains here:
                # shorter chains will likely be outliers, and we want to notice them
                cov_of_means = np.cov(means.T)  # , fweights=Ns)
                # For numerical stability, we turn mean_of_covs into correlation matrix:
                #   rho = (diag(Sigma))^(-1/2) * Sigma * (diag(Sigma))^(-1/2)
                # and apply the same transformation to the mean of covs (same eigenvals!)
                diagSinvsqrt = np.diag(np.power(np.diag(cov_of_means), -0.5))
                corr_of_means     = diagSinvsqrt.dot(cov_of_means).dot(diagSinvsqrt)
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
                    eigvals = np.linalg.eigvalsh(Linv.dot(corr_of_means).dot(Linv.T))
                    Rminus1 = max(np.abs(eigvals))
                    # For real square matrices, a possible def of the cond number is:
                    condition_number = Rminus1/min(np.abs(eigvals))
                    self.log.debug("Condition number = %g", condition_number)
                    self.log.debug("Eigenvalues = %r", eigvals)
                    self.log.info("Convergence of means: R-1 = %f after %d samples",
                                  Rminus1, self.n())
                    success = True
                    # Have we converged in means?
                    # (criterion must be fulfilled twice in a row)
                    if (max(Rminus1,
                            getattr(self, "Rminus1_last", np.inf)) < self.Rminus1_stop):
                        # Check the convergence of the bounds of the confidence intervals
                        # Same as R-1, but with the rms deviation from the mean bound
                        # in units of the mean standard deviation of the chains
                        Rminus1_cl = (np.std(bounds, axis=0).T /
                                      np.sqrt(np.diag(mean_of_covs)))
                        self.log.debug("normalized std's of bounds = %r", Rminus1_cl)
                        self.log.info("Convergence of bounds: R-1 = %f after %d samples",
                                      np.max(Rminus1_cl), self.n())
                        if np.max(Rminus1_cl) < self.Rminus1_cl_stop:
                            self.converged = True
                            self.log.info("The run has converged!")
            # Broadcast and save the convergence status and the last R-1 of means
            success = get_mpi_comm().bcast(
                (success if not get_mpi_rank() else None), root=0)
            if success:
                self.Rminus1_last = get_mpi_comm().bcast(
                    (Rminus1 if not get_mpi_rank() else None), root=0)
                self.converged = get_mpi_comm().bcast(
                    (self.converged if not get_mpi_rank() else None), root=0)
        else:  # No MPI
            pass
        # Do we want to learn a better proposal pdf?
        if self.learn_proposal and not self.converged:
            # update iff (not MPI, or MPI and "good" Rminus1)
            if get_mpi():
                good_Rminus1 = (self.learn_proposal_Rminus1_max >
                                self.Rminus1_last > self.learn_proposal_Rminus1_min)
                if not good_Rminus1:
                    if not get_mpi_rank():
                        self.log.info("Bad convergence statistics: "
                                      "waiting until the next checkpoint.")
                    return
            if get_mpi():
                if get_mpi_rank():
                    mean_of_covs = np.empty((self.prior.d(),self.prior.d()))
                get_mpi_comm().Bcast(mean_of_covs, root=0)
            elif not get_mpi():
                mean_of_covs = covs[0]
            self.proposer.set_covariance(mean_of_covs)
            if not get_mpi_rank():
                self.log.info("Updated covariance matrix of proposal pdf.")
                self.log.debug("%r", mean_of_covs)

    # Finally: returning the computed products ###########################################

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the accepted steps.
        """
        return {"sample": self.collection}
