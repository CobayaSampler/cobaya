"""
.. module:: samplers.mcmc.proposal

:Synopsis: proposal distributions
:Author: Antony Lewis (from CosmoMC)

Using the covariance matrix to give the proposal directions typically
significantly increases the acceptance rate and gives faster movement
around parameter space.

We generate a random basis in the eigenvectors, then cycle through them
proposing changes to each, then generate a new random basis.
The distance proposal in the random direction is given by a two-D Gaussian
radial function mixed with an exponential, which is quite robust to wrong width estimates

See http://cosmologist.info/notes/CosmoMC.pdf
"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import numpy as np
import scipy
import numpy

# Local
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)


class IndexCycler(object):
    def __init__(self, n):
        self.n = n
        self.loopix = -1


class CyclicIndexRandomizer(IndexCycler):
    def next(self):
        """
        Get the next random index

        :return: index
        """
        self.loopix = (self.loopix + 1) % self.n
        if self.loopix == 0:
            self.indices = np.random.permutation(self.n)
        return self.indices[self.loopix]


class RandDirectionProposer(IndexCycler):
    def propose_vec(self, scale):
        """
        propose a random n-dimension vector

        :param scale: units for the distance
        :return: array with vector
        """
        self.loopix = (self.loopix + 1) % self.n
        if self.loopix == 0:
            if self.n > 1:
                self.R = scipy.stats.special_ortho_group.rvs(self.n)
            else:
                self.R = np.eye(1)*numpy.random.choice((-1,1))
        return self.R[:, self.loopix] * self.propose_r() * scale

    def propose_r(self):
        """
        Radial proposal. By default a mixture of an exponential and 2D Gaussian radial proposal
        (to make wider tails and more mass near zero, so more robust to scale misestimation)
        :return: random distance (unit scale)
        """
        if np.random.uniform() < 0.33:
            return np.random.exponential()
        else:
            return np.linalg.norm(np.random.normal(size=min(self.n, 2)))


class BlockProposer(RandDirectionProposer):
    def update_params(self, params, vec):
        params[self.params_changed] += self.mapping_matrix.dot(vec)


class CovmatProposer(object):
    def set_covariance(self):
        pass


class BlockedProposer(CovmatProposer):
    def __init__(self, parameter_blocks, params_used, slow_block_max,
                 oversample_fast=1, propose_scale=2.4):

        """
        Proposal density for fast and slow parameters, where parameters are
        grouped into blocks which are changed at the same time.

        :param parameter_blocks: list of arrays of parameter indices in each block
        :param params_used: array of indices of actual parameters that are varied
        :param slow_block_max: index of block which is the last which is slow
        :param oversample_fast: factor by which to oversample fast parameters
        :param propose_scale: overal scale for the proposal
        """
        self.oversample_fast = oversample_fast
        self.propose_scale = propose_scale
        self.fast_ix = 0
        # total number and number of slow parameters
        n_all = 0
        n_slow = 0
        used_blocks_indices = []
        for i, block in enumerate(parameter_blocks):
            n_block = len(block)
            if n_block:
                used_blocks_indices.append(i)
                n_all += n_block
                if i <= slow_block_max:
                    n_slow += n_block
        self.all  = CyclicIndexRandomizer(n_all)
        self.slow = CyclicIndexRandomizer(n_slow)
        self.fast = CyclicIndexRandomizer(n_all - n_slow)
        # Creating the proposers
        self.proposer = []
        self.proposer_for_index = range(n_all)
        self.used_param_indices = np.zeros(n_all, dtype=int)
        # parameter index
        ix = 0
        for i_block in used_blocks_indices:
            pars = parameter_blocks[i_block]
            bp = BlockProposer(len(pars))
            bp.block_start = ix
            bp.used_param_indices = pars[:] # this is a copy
            self.proposer.append(bp)
            self.proposer_for_index[ix:ix + bp.n] = [bp for _ in range(bp.n)]
            self.used_param_indices[ix:ix + bp.n] = pars
            ix += bp.n
        for i_block, bp in enumerate(self.proposer):
            bp.used_params_changed = self.used_param_indices[bp.block_start:]
            # ensuring params_used is an array, to be able to select with list of indices
            bp.params_changed = np.array(params_used)[bp.used_params_changed]

    def set_covariance(self, propose_matrix):
        """
        Take covariance of used parameters (propose_matrix), and construct orthonormal parameters
        where orthonormal parameters are grouped in blocks by speed, so changes in slowest block
        changes slow and fast parameters, but changes in the fastest block only changes fast parameters

        :param propose_matrix: covariance matrix for used parameters
        """
        if not (np.allclose(propose_matrix.T,propose_matrix) and
                np.all(np.linalg.eigvals(propose_matrix) > 0)):
            log.error("The given covmat is not a positive-definite, "
                      "symmetric square matrix.")
            raise HandledException
        self.propose_matrix = propose_matrix.copy()
        sigmas = np.sqrt(np.diag(propose_matrix))
        corr = propose_matrix[np.ix_(self.used_param_indices, self.used_param_indices)]
        # before: "in range(self.size)", UNDEFINED!!!
        for i in range(len(self.used_param_indices)):
            s = np.sqrt(corr[i, i])
            corr[i, :] /= s
            corr[:, i] /= s
        L = np.linalg.cholesky(corr)
        for i, bp in enumerate(self.proposer):
            bp.mapping_matrix = np.empty((len(bp.used_params_changed), bp.n))
            for j, par in enumerate(bp.used_params_changed):
                bp.mapping_matrix[j, :] = (sigmas[par] *
                                           L[bp.block_start + j, bp.block_start:bp.block_start+bp.n])

    def get_covariance(self):
        return self.propose_matrix.copy()

    def get_block_proposal(self, P, bp):
        bp.update_params(P, bp.propose_vec(self.propose_scale))

    def get_proposal(self, P):
        if self.fast_ix:
            self.get_proposal_fast(P)
            self.fast_ix -= 1
        else:
            if self.all.next() > (self.slow.n - 1):
                self.get_proposal_fast(P)
                self.fast_ix = self.oversample_fast - 1
            else:
                self.get_proposal_slow(P)

    def get_proposal_slow(self, P):
        self.get_block_proposal(P, self.proposer_for_index[self.slow.next()])

    def get_proposal_fast(self, P):
        self.get_block_proposal(P, self.proposer_for_index[self.slow.n + self.fast.next()])

    def get_proposal_fast_delta(self,P):
        P[:]=0
        self.get_proposal_fast(P)
