"""
.. module:: samplers.mcmc

:Synopsis: Blocked fast-slow Metropolis sampler (Lewis 1304.4473)
:Author: Antony Lewis (for the CosmoMC sampler), Jesus Torrado (for parts of the wrapper only)

.. |br| raw:: html

   <br />

.. note::
   **If you use this sampler, please cite it as:**
   |br|
   `A. Lewis and S. Bridle, "Cosmological parameters from CMB and other data: A Monte Carlo approach"
   (arXiv:astro-ph/0205436) <https://arxiv.org/abs/astro-ph/0205436>`_
   |br|
   `A. Lewis, "Efficient sampling of fast and slow cosmological parameters"
   (arXiv:1304.4473) <https://arxiv.org/abs/1304.4473>`_
   |br|
   If you use *fast-dragging*, you should also cite
   |br|
   `R.M. Neal, "Taking Bigger Metropolis Steps by Dragging Fast Variables"
   (arXiv:math/0502099) <https://arxiv.org/abs/math/0502099>`_


This is the Markov Chain Monte Carlo Metropolis sampler used by CosmoMC, and described in 
`Lewis, "Efficient sampling of fast and slow cosmological parameters" (arXiv:1304.4473) 
<https://arxiv.org/abs/1304.4473>`_.

The proposal pdf is a gaussian mixed with an exponential pdf in random directions, which is
more robust to misestimation of the width of the proposal than a pure gaussian. The scale width
of the proposal can be specified per parameter with the property ``proposal`` (it defaults to the
standard deviation of the reference pdf, if defined, or the prior's one, if not). However,
initial performance will be much better if you provide a covariance matrix, which overrides
the default proposal scale width set for each parameter.

A callback function can be specified through the ``callback_function`` option. In it, the
sampler instance is accessible as ``sampler_instance``, which has ``prior``, ``likelihood``
and (sample) ``collection`` as attributes.

Initial point and covariance of the proposal pdf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The initial points for the chains are sampled from the *reference* pdf
(see :doc:`params_prior`). The reference pdf can be a fixed point, and in that case the
chain starts always from that same point. If there is no reference pdf defined for a
parameter, the initial sample is drawn from the prior instead.

Example *parameters* block:

.. code-block:: yaml
   :emphasize-lines: 10,17

   params:
     a:
      ref:
        min: -1
        max:  1
      prior:
        min: -2
        max:  2
      latex: \\alpha
      proposal: 0.5
     b:
      ref: 2
      prior:
        min: -1
        max:  4
      latex: \\beta
      proposal: 0.25
     c:
      ref:
        dist: norm
        loc: 0
        scale: 0.2        
      prior:
        min: -1
        max:  1
      latex: \\gamma

+ ``a`` -- the initial point of the chain is drawn from an uniform pdf between -1 and 1,
  and its proposal width is 0.5.
+ ``b`` -- the initial point of the chain is always 2,
  and its proposal width is 0.25.
+ ``c`` -- the initial point of the chain is drawn from a gaussian centred at 0
  with standard deviation 0.2; its proposal width is not specified, so it is taken to be 
  that of the reference pdf, 0.2.

A good initial covariance matrix for the proposal is critical for convergence.
It can be specified either with the property ``proposal`` of each parameter, as shown
above, or thorugh ``mcmc``'s property ``covmat``, as a file name (including path,
if not located at the invocation folder).
The first line of the ``covamt`` file must start with ``#``, followed by a list of parameter
names, separated by a space. The rest of the file must contain the covariance matrix,
one row per line. It does not need to contain the same parameters as the sampled ones: 
it overrides the ``proposal``'s (and adds covariances) for the sampled parameters,
and ignores the non-sampled ones.

An example for the case above::

   # a     b
     0.1   0.01
     0.01  0.2

In this case, internally, the final covariance matrix of the proposal would be::

   # a     b     c
     0.1   0.01  0
     0.01  0.2   0
     0     0     0.04

If the option `learn_proposal` is set to ``True``, the covariance matrix will be updated 
once in a while to accelerate convergence
(nb. convergence testing is only implemented for paralel chains right now).

If you are not sure that your posterior has one single mode, or if its shape is very
irregular, you should probably set ``learn_proposal: False``.

If you don't know how good your initial guess for starting point and covariance of the 
proposal are, it is a good idea to allow for a number of initial *burn in* samples,
e.g. 10 per dimension. This can be specified with the parameter ``burn_in``.
These samples will be ignored for all purposes (output, covergence, proposal learning...)


.. _mcmc_speed_hierarchy:

Taking advantage of a speed hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The proposal pdf is *blocked* by speeds, i.e. it allows for efficient sampling of a mixture
of *fast* and *slow* parameters, such that we can avoid recomputing the parts of the
likelihood determined by slow parameters when sampling along the fast directions only.

Two different sampling schemes are available to take additional advantage from a speed
hierarchy:

- **Dragging the fast parameters:** implies a number of intermediate steps when
  jumping between fast+slow combinations, such that the jump in the fast parameters is
  optimised with respect to the jump in the slow parameters to explore any possible
  degeneracy betwen them.

- **Oversampling the fast parameters:** consists simply of taking a larger proportion of
  steps in the fast directions, since exploring their conditional distributions is cheap.

In general, the *dragging* method is the recommended one.
For a thorough description of both methods, see
`A. Lewis, "Efficient sampling of fast and slow cosmological parameters" (arXiv:1304.4473) <https://arxiv.org/abs/1304.4473>`_.

The relative speeds can be specified per likelihood/theory, with the option ``speed``.
It is given as factors of the speed of the slowest part, which is not specified.
If two or more likelihoods with different speeds share a parameter, said parameter is
assigned the slowest of their speeds.

For example:

.. code-block:: yaml

   theory:
     theory_code:
       speed: 2

   likelihood:
     lik_a:
     lik_b:
       speed: 4

Here, evaluating the likelihood ``lik_a`` is the slowest step, while the theory code and
the likelihood ``lik_b`` are both faster, with ``lik_b`` being faster than the theory code
(the absolute values of the speeds are ignored, only the relative ranking matters:
giving speeds 1,2,3 would have the same effect).

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
import six

# Global
import os
import sys
from copy import deepcopy
from itertools import chain
import numpy as np

# Local
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi, get_mpi_size, get_mpi_rank, get_mpi_comm
from cobaya.collection import Collection, OnePoint
from cobaya.conventions import _weight, _p_proposal, _p_ref
from cobaya.samplers.mcmc.proposal import BlockedProposer
from cobaya.log import HandledException
from cobaya.tools import get_external_function

# Logger
import logging
log = logging.getLogger(__name__)


class mcmc(Sampler):

    def initialise(self):
        """Initialises the sampler:
        creates the proposal distribution and draws the initial sample."""
        log.info("Initialising")
        # Burning-in countdown -- the +1 accounts for the initial point (always accepted)
        self.burn_in_left = self.burn_in + 1
        # One collection per MPI process: `name` is the MPI rank + 1
        name = str(1 + (lambda r: r if r!=None else 0)(get_mpi_rank()))
        self.collection = Collection(self.parametrisation, self.likelihood,
                                     self.output, name=name)
        self.current_point = OnePoint(self.parametrisation, self.likelihood,
                                     self.output, name=name)
        # No simultaneous fast-dragging and oversampling:
        self.oversample_fast = (1 if not self.oversample_fast else self.oversample_fast)
        if ((self.oversample_fast > 1) and
            (self.drag_nfast_times or self.drag_interp_steps)):
            log.error(
                "Not possible to do oversampling and fast-dragging at the same time.")
            raise HandledException
        # Create proposer
        # If externally, defined, use and ignore the rest
        ##########################################################
        if getattr(self, "external", None):
            proposal = self.external
            class proposer():
                def get_proposal(self, trial):
                    proposal(trial)
            self.proposer = proposer()
            self.get_new_sample = self.get_new_sample_metropolis
            return
            # THIS IS WRONG, JUST TESTING CODE!!!!!!
        ##########################################################
        # Prepare speed hierarchy
        speeds, blocks = zip(*self.likelihood.speeds_of_params().items())
        self.max_speed_slow = max(self.max_speed_slow, min(speeds))
        # Turn parameter names into indices
        blocks = [[self.parametrisation.sampled_params().keys().index(p) for p in b] for b in blocks]
        try:
            i_last_slow_block = (i for i,speed in enumerate(list(speeds))
                                 if speed > self.max_speed_slow).next() - 1
        except StopIteration:
            i_last_slow_block = len(speeds) - 1
            if self.max_speed_slow > max(speeds):
                log.error("`max_speed_slow` is equal or higher than the fastest speed, "
                          "so no fast-dragging or oversampling.")
                raise HandledException
            elif (self.oversample_fast > 1 or
                  (self.drag_nfast_times or self.drag_interp_steps)):
                log.error(
                    "All blocks have the same speed, so no fast-dragging or oversampling.")
                raise HandledException
        self.proposer = BlockedProposer(blocks, oversampling_factors=self.oversample_fast,
                i_last_slow_block=i_last_slow_block, propose_scale=self.propose_scale)
        # Save the number of slow parameters
        # -- it's the effective number of parameters for counting steps
        self.n_slow = sum(len(blocks[i]) for i in range(1+i_last_slow_block))
        # Build the initial covariance matrix of the proposal
        covmat = self.initial_proposal_covmat()
        log.info("Sampling with covariance matrix:")
        log.info("%r", covmat)
        self.proposer.set_covariance(covmat)
        # Using fast dragging?
        if self.drag_nfast_times and self.drag_interp_steps:
            log.error(
                "To specify the number of dragging interpolating steps, "
                "use *either* `drag_nfast_times` or `drag_interp_steps`, not both.")
            raise HandledException
        if self.drag_nfast_times:
            self.drag_interp_steps = (
                max(2, int(self.drag_nfast_times * self.proposer.fast.n) + 1))
        if self.drag_interp_steps:
            self.get_new_sample = self.get_new_sample_dragging
            fast_params = [self.parametrisation.sampled_params().keys()[i]
                           for i in chain(*blocks[1+i_last_slow_block:])]
            log.info("Using fast dragging over %d slow parameters, "
                     "with %d interpolating steps on fast parameters %r",
                     self.n_slow, self.drag_interp_steps, fast_params)
        else:
            self.get_new_sample = self.get_new_sample_metropolis
        # Prepare callback function
        if self.callback_function != None:
            self.callback_function_callable = get_external_function(self.callback_function)

    def initial_proposal_covmat(self):
        """
        Build the initial covariance matrix, using the data provided, in descending order
        of priority:
        1. "covmat" field in the "mcmc" sampler block.
        2. "proposal" field for each parameter.
        3. variance of the reference pdf.
        4. variance of the prior pdf.

        The covariances between parameters when both are present in a covariance matrix
        provided through option 1 are preserved. All other covariances are assumed to be 0.
        """
        params, params_infos = zip(*self.parametrisation.sampled_params().items())
        covmat = np.diag([np.nan]*len(params))
        # If given, load and test the covariance matrix
        if isinstance(self.covmat, six.string_types):
            try:
                with open(self.covmat, "r") as file_covmat:
                    header = file_covmat.readline()
                loaded_covmat = np.loadtxt(self.covmat)
            except TypeError:
                log.error("The property 'covmat' must be a file name,"
                          "but it's '%s'.", str(self.covmat))
                raise HandledException
            except IOError:
                log.error("Can't open covmat file '%s'.", self.covmat)
                raise HandledException
            if header[0] != "#":
                log.error("The first line of the covmat file '%s' "
                          "must be one list of parameter names separated by spaces "
                          "and staring with '#', and the rest must be a square matrix, "
                          "with one row per line.", self.covmat)
                raise HandledException
            loaded_params = header.strip("#").strip().split()
        elif hasattr(self.covmat, "__getitem__"):
            if not self.covmat_params:
                log.error("If a covariance matrix is passed as a numpy array, "
                          "you also need to pass the parameters it corresponds to "
                          "via 'covmat_params: [name1, name2, ...]'.")
                raise HandledException
            loaded_params = self.covmat_params
            loaded_covmat = self.covmat
        if self.covmat is not None:
            if len(loaded_params) != len(set(loaded_params)):
                log.error("There are duplicated parameters in the header of the "
                          "covmat file '%s' ", self.covmat)
                raise HandledException
            if len(loaded_params) != loaded_covmat.shape[0]:
                log.error("The number of parameters in the header of '%s' and the "
                          "dimensions of the matrix do not coincide.", self.covmat)
                raise HandledException
            if not (np.allclose(loaded_covmat.T, loaded_covmat) and
                    np.all(np.linalg.eigvals(loaded_covmat) > 0)):
                log.error("The covmat loaded from '%s' is not a positive-definite, "
                          "symmetric square matrix.", self.covmat)
                raise HandledException
            # Fill with parameters in the loaded covmat
            loaded_params_used = set(loaded_params).intersection(set(params))
            if not loaded_params_used:
                log.error("A proposal covariance matrix has been loaded, but none of its "
                          "parameters are actually sampled here. Maybe a mismatch between"
                          " parameter names in the covariance matrix and the input file?")
                raise HandledException
            indices_used, indices_sampler = np.array(
                [[loaded_params.index(p),params.index(p)]
                 for p in loaded_params if p in loaded_params_used]).T
            covmat[np.ix_(indices_sampler,indices_sampler)] = (
                loaded_covmat[np.ix_(indices_used,indices_used)])
            log.info("Covariance matrix loaded for params %r", list(loaded_params_used))
        # Fill gaps with "proposal" property, if present, otherwise ref (or prior)
        where_nan = np.isnan(covmat.diagonal())
        if np.any(where_nan):
            covmat[where_nan, where_nan] = np.array(
                [info.get(_p_proposal, np.nan)**2 for info in params_infos])[where_nan]
            # we want to start learning the covmat earlier
            log.info("Covariance matrix "
                     +("not present" if np.all(where_nan) else "not complete")+". "
                     "We will start learning the covariance of the proposal earlier: "
                     "R-1 = %g (was %g).", self.learn_proposal_Rminus1_max_early,
                     self.learn_proposal_Rminus1_max)
            self.learn_proposal_Rminus1_max = self.learn_proposal_Rminus1_max_early
        where_nan = np.isnan(covmat.diagonal())
        if np.any(where_nan):
            covmat[where_nan, where_nan] = self.prior.reference_covmat().diagonal()[where_nan]
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
        log.info("Initial point:\n %r ",self.current_point)
        # Main loop!
        self.converged = False
        log.info("Sampling!")
        while self.n() < (self.max_samples*self.oversample_fast) and not(self.converged):
            self.get_new_sample()
            # Callback function
            if (hasattr(self, "callback_function_callable") and
                not(max(self.n(),1)%self.callback_every) and
                self.current_point["weight"]==1):
                self.callback_function_callable(self)
            # Checking convergence and (optionally) learning the covmat of the proposal
            if self.check_all_ready():
                self.check_convergence_and_learn_proposal()
        # Make sure the last batch of samples ( < output_every ) are written
        self.collection.out_update()
        if not get_mpi_rank():
            log.info("Sampling complete after %d accepted steps.", self.n())
        if self.oversample_fast > 1:
            log.info("(Including an oversampling factor of '%d'.)", self.oversample_fast)

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
        trial = deepcopy(self.current_point[self.parametrisation.sampled_params()])
        self.proposer.get_proposal(trial)
        logpost_trial, logprior_trial, logliks_trial, derived = self.logposterior(trial)
        accept = self.metropolis_accept(logpost_trial, -self.current_point["minuslogpost"])
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
        start_slow_point   = self.current_point[self.parametrisation.sampled_params()]
        start_slow_logpost = -self.current_point["minuslogpost"]
        end_slow_point = deepcopy(start_slow_point)
        self.proposer.get_proposal_slow(end_slow_point)
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
        end_drag_logpost_acc   = end_slow_logpost
        # start dragging
        for i_step in range(1, 1+self.drag_interp_steps):
            log.debug("Dragging step: %d", i_step)
            # take a step in the fast direction in both slow extremes
            delta_fast = np.zeros(len(current_start_point))
            self.proposer.get_proposal_fast(delta_fast)
            proposal_start_point = deepcopy(current_start_point)
            proposal_start_point += delta_fast
            proposal_end_point = deepcopy(current_end_point)
            proposal_end_point += delta_fast
            # get the new extremes for the interpolated probability
            # (reject if any of them = -inf; avoid evaluating both if just one fails)
            # Force the computation of the (slow blocks) derived params at the starting
            # point, but discard them, since they contain the starting point's fast ones,
            # not used later -- save the end point's ones.
            proposal_start_logpost = self.logposterior(proposal_start_point)[0]
            proposal_end_logpost, proposal_end_logprior, proposal_end_logliks, derived_proposal_end = (
                self.logposterior(proposal_end_point)
                if proposal_start_logpost > -np.inf
                else (-np.inf, None, [], []))
            if proposal_start_logpost > -np.inf and proposal_end_logpost > -np.inf:
                # create the interpolated probability and do a Metropolis test
                frac = i_step / self.drag_interp_steps
                proposal_interp_logpost = ((1-frac)*proposal_start_logpost
                                             +frac *proposal_end_logpost)
                current_interp_logpost  = ((1-frac)*current_start_logpost
                                             +frac *current_end_logpost)
                accept_drag = self.metropolis_accept(proposal_interp_logpost,
                                                     current_interp_logpost)
            else:
                accept_drag = False
            log.debug("Dragging step: %s", ("accepted" if accept_drag else "rejected"))
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
        log.debug("TOTAL step: %s", ("accepted" if accept else "rejected"))
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
                log.debug("New sample, #%d: \n   %r", self.n(), self.current_point)
                if self.n()%self.output_every == 0:
                    self.collection.out_update()
            else:
                self.burn_in_left -= 1
                log.debug("Burn-in sample:\n   %r", self.current_point)
                if self.burn_in_left == 0:
                    log.info("Finished burn-in phase: discarded %d accepted steps.",
                             self.burn_in)
            # set the new point as the current one, with weight one
            self.current_point.add(trial, derived=derived, weight=1, logpost=logpost_trial,
                                   logprior=logprior_trial, logliks=logliks_trial)
        else: # not accepted
            self.current_point.increase_weight(1)
            # Failure criterion: chain stuck!
            if self.current_point["weight"] > self.max_tries:
                self.collection.out_update()
                log.error(
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
        if (self.n() > 0 and self.current_point["weight"]==1 and
            not (self.n()%(self.check_every_dimension_times*self.n_slow))):
            log.info("Checkpoint: %d samples accepted.", self.n())
            # If not MPI, we are ready
            if not get_mpi():
                if msg_ready:
                    log.info(msg_ready)
                return True
            # If MPI, tell the rest that we are ready -- we use a "gather"
            # ("reduce" was problematic), but we are in practice just pinging
            if not hasattr(self, "req"): # just once!
                self.all_ready = np.empty(get_mpi_size())
                self.req = get_mpi_comm().Iallgather(
                    np.array([1.]), self.all_ready)
                log.info(msg_ready + " (waiting for the rest...)")
        # If all processes are ready to learn (= communication finished)
        if (self.req.Test() if hasattr(self, "req") else False):
            # Sanity check: actually all processes have finished
            assert np.all(self.all_ready == 1), (
                "This should not happen! Notify the developers. (Got %r)", self.all_ready)
            if get_mpi_rank() == 0:
                log.info("All chains are r"+msg_ready[1:])
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
        # Compute and gather means, covs and cl limits of last half of chains
        mean = self.collection.mean(first=int(self.n()/2))
        cov  = self.collection.cov (first=int(self.n()/2))
        # Turn off logging of warnings momentarily, so getdist won't complain innecessarily
        logging.disable(logging.WARNING)
        mcsamples = self.collection.as_getdist_mcsamples(
            derived=False, prior_and_lik=False, first=int(self.n()/2))
        logging.disable(logging.NOTSET)
        limit = np.array([[mcsamples.confidence(i, limfrac=self.Rminus1_cl_level/2., upper=lim)
                            for i in range(self.prior.d())] for lim in [False, True]]).T
        Ns, means, covs, limits = map(
            lambda x: np.array((get_mpi_comm().gather(x) if get_mpi() else [x])),
            [self.n(), mean, cov, limit])
        # Compute convergence diagnostics
        if get_mpi():
            if get_mpi_rank() == 0:
                # "Within" or "W" term -- our "units" for assessing convergence
                # and our prospective new covariance matrix
                mean_of_covs = np.average(covs, weights=Ns, axis=0)
                # "Between" or "B" term
                # We don't weight with the number of samples in the chains here:
                # shorter chains will likely be outliers, and we want to notice them
                cov_of_means = np.cov(means.T) #, fweights=Ns)
                # For numerical stability, we turn mean_of_covs into correlation matrix:
                #   rho = (diag(Sigma))^(-1/2) * Sigma * (diag(Sigma))^(-1/2)
                # and apply the same transformation to the mean of covs (same eigenvals!)
                diagSinvsqrt = np.diag(np.power(np.diag(cov_of_means), -0.5))
                corr_of_means     = diagSinvsqrt.dot(cov_of_means).dot(diagSinvsqrt)
                norm_mean_of_covs = diagSinvsqrt.dot(mean_of_covs).dot(diagSinvsqrt)
                # Cholesky of (normalised) mean of covs and eigvals of Linv*cov_of_means*L
                L = np.linalg.cholesky(norm_mean_of_covs)
                Linv = np.linalg.inv(L)
                eigvals = np.linalg.eigvalsh(Linv.dot(corr_of_means).dot(Linv.T))                
                Rminus1 = max(np.abs(eigvals))
                # For real square matrices, a possible def of the cond number is:
                condition_number = Rminus1/min(np.abs(eigvals))
                log.debug("Condition number = %g", condition_number)
                log.debug("Eigenvalues = %r", eigvals)
                log.info("Convergence of means: R-1 = %f after %d samples", Rminus1, self.n())
#                assert np.all(eigvals >= 0), (
#                    "Negative eigenvectors. This should not happen. "
#                    "Check the condition number of the diagonalised matrix.")
                # Have we converged in means? (criterion must be fulfilled twice in a row)
                if (max(Rminus1, getattr(self, "Rminus1_last", np.inf)) < self.Rminus1_stop):
                    # Check the convergence of the limits of the confidence intervals
                    # Same as R-1, but with the rms deviation from the mean limit
                    # in units of the mean standard deviation of the chains
                    Rminus1_cl = np.std(limits, axis=0).T/np.sqrt(np.diag(mean_of_covs))
                    log.debug("Normalised std's of limits = %r", Rminus1_cl)
                    log.info("Convergence of limits: R-1 = %f after %d samples", 
                             np.max(Rminus1_cl), self.n())
                    if np.max(Rminus1_cl) < self.Rminus1_cl_stop:
                        self.converged=True
                        log.info("The run has converged!")
            # Broadcast and save the convergence status and the last R-1 of means
            self.Rminus1_last = get_mpi_comm().bcast(
                (Rminus1 if not get_mpi_rank() else None), root=0)
            self.converged = get_mpi_comm().bcast(
                (self.converged if not get_mpi_rank() else None), root=0)
        else: # No MPI
            pass
        # Do we want to learn a better proposal pdf?
        if self.learn_proposal and not self.converged:
            # update iff (not MPI, or MPI and "good" Rminus1)
            if get_mpi():
                good_Rminus1 = (self.Rminus1_last < self.learn_proposal_Rminus1_max and
                                self.Rminus1_last > self.learn_proposal_Rminus1_min)
                if not good_Rminus1:
                    if not get_mpi_rank():
                        log.info("Bad convergence statistics: waiting until the next checkpoint.")
                    return
            if get_mpi():
                if get_mpi_rank():
                    mean_of_covs = np.empty((self.prior.d(),self.prior.d()))
                get_mpi_comm().Bcast(mean_of_covs, root=0)
            elif not get_mpi():
                mean_of_covs = covs[0]
            self.proposer.set_covariance(mean_of_covs)
            if not get_mpi_rank():
                log.info("Updated covariance matrix of proposal pdf.")
                log.debug("%r", mean_of_covs)

    # Finally: returning the computed products ############################################

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``Collection`` containing the accepted steps.
        """
        return {"sample": self.collection}
