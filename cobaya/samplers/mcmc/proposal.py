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

See https://cosmologist.info/notes/CosmoMC.pdf
"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import numpy as np
import scipy
import numpy
from itertools import chain

# Local
from cobaya.log import HandledException

# Logger
import logging

log = logging.getLogger(__name__.split(".")[-1])


class IndexCycler(object):
    def __init__(self, n):
        self.n = n
        self.loop_index = -1


class CyclicIndexRandomizer(IndexCycler):
    def next(self):
        """
        Get the next random index

        :return: index
        """
        self.loop_index = (self.loop_index + 1) % self.n
        if self.loop_index == 0:
            self.indices = np.random.permutation(self.n)
        return self.indices[self.loop_index]


class RandDirectionProposer(IndexCycler):
    def propose_vec(self, scale=1):
        """
        propose a random n-dimension vector

        :param scale: units for the distance
        :return: array with vector
        """
        self.loop_index = (self.loop_index + 1) % self.n
        if self.loop_index == 0:
            if self.n > 1:
                self.R = scipy.stats.special_ortho_group.rvs(self.n)
            else:
                self.R = np.eye(1) * numpy.random.choice((-1, 1))
        return self.R[:, self.loop_index] * self.propose_r() * scale

    def propose_r(self):
        """
        Radial proposal. By default a mixture of an exponential and 2D Gaussian radial
        proposal (to make wider tails and more mass near zero, so more robust to scale
        misestimation)

        :return: random distance (unit scale)
        """
        if np.random.uniform() < 0.33:
            return np.random.exponential()
        else:
            return np.linalg.norm(np.random.normal(size=min(self.n, 2)))


class BlockedProposer(object):
    def __init__(self, parameter_blocks, oversampling_factors=None,
                 i_last_slow_block=None, proposal_scale=2.4):
        """
        Proposal density for fast and slow parameters, where parameters are
        grouped into blocks which are changed at the same time.

        :param parameter_blocks: list of arrays of parameter indices in each block,
            with blocks sorted by ascending speed.
        :param oversampling_factors: list of *int* oversampling factors *per parameter*,
            i.e. a factor of n for a block of dimension d would mean n*d jumps for that
            block per full cycle, whereas a factor of 1 for all blocks (default) means
            that all *directions* are treated equally (but the proposals are still
            block-wise).
        :param i_last_slow_block: index of the last block considered slow.
            By default, all blocks are considered slow.
        :param proposal_scale: overall scale for the proposal.
        """
        self.proposal_scale = proposal_scale
        if oversampling_factors is None:
            self.oversampling_factors = np.array([1 for _ in parameter_blocks], dtype=int)
        else:
            if len(oversampling_factors) != len(parameter_blocks):
                log.error("List of oversampling factors has a different length that "
                          "list of blocks: %d vs %d.",
                          len(oversampling_factors), len(parameter_blocks))
                raise HandledException
            if np.any(oversampling_factors != np.floor(np.array(oversampling_factors))):
                log.error("Oversampling factors must be integer! Got %r.",
                          oversampling_factors)
                raise HandledException
            self.oversampling_factors = np.array(oversampling_factors, dtype=int)
        # Turn it into per-block: multiply by number of params in block
        self.oversampling_factors *= np.array([len(b) for b in parameter_blocks])
        if i_last_slow_block is None:
            i_last_slow_block = len(parameter_blocks) - 1
        else:
            if i_last_slow_block > len(parameter_blocks) - 1:
                log.error("The index given for the last slow block, %d, is not valid: "
                          "there are only %d blocks.",
                          i_last_slow_block, len(parameter_blocks))
                raise HandledException
        n_all = sum([len(b) for b in parameter_blocks])
        n_slow = sum([len(b) for b in parameter_blocks[:1 + i_last_slow_block]])
        if set(list(chain(*parameter_blocks))) != set(range(n_all)):
            log.error("The blocks do not contain all the parameter indices.")
            raise HandledException
        # Mapping between internal indices, sampler parameter indices and blocks:
        # let i=0,1,... be the indices of the parameters for the sampler,
        # and j=0,1,... be the indices of the parameters *as given to the proposal*
        # i.e. if passed blocks=[[1,2],[0]] (those are the i's),
        # then the j's are [0 (for 1), 1 (for 2), 2 (for 0)].
        # iblock is the index of the blocks
        self.i_of_j = np.array(list(chain(*parameter_blocks)))
        self.iblock_of_j = list(
            chain(*[[iblock] * len(b) for iblock, b in enumerate(parameter_blocks)]))
        # Creating the blocked proposers
        self.proposer = [RandDirectionProposer(len(b)) for b in parameter_blocks]
        # Starting j index of each block
        self.j_start = [len(list(chain(*parameter_blocks[:iblock])))
                        for iblock, b in enumerate(parameter_blocks)]
        # Parameter cyclers, cycling over the j's
        self.cycler_all = CyclicIndexRandomizer(n_all)
        # These ones are used by fast dragging only
        self.cycler_slow = CyclicIndexRandomizer(n_slow)
        self.cycler_fast = CyclicIndexRandomizer(n_all - n_slow)
        # Samples left to draw from the current block
        self.samples_left = 0

    def d(self):
        return len(self.i_of_j)

    def get_proposal(self, P):
        # if a block has been chosen
        if self.samples_left:
            self.get_block_proposal(P, self.current_iblock)
            self.samples_left -= 1
        # otherwise, choose a block
        else:
            self.current_iblock = self.iblock_of_j[self.cycler_all.next()]
            self.samples_left = self.oversampling_factors[self.current_iblock]
            self.get_proposal(P)

    def get_proposal_slow(self, P):
        current_iblock_slow = self.iblock_of_j[self.cycler_slow.next()]
        self.get_block_proposal(P, current_iblock_slow)

    def get_proposal_fast(self, P):
        current_iblock_fast = self.iblock_of_j[
            self.cycler_slow.n + self.cycler_fast.next()]
        self.get_block_proposal(P, current_iblock_fast)

    def get_block_proposal(self, P, iblock):
        vec_standarized = self.proposer[iblock].propose_vec(self.proposal_scale)
        P[self.i_of_j[self.j_start[iblock]:]] += (self.transform[iblock]
                                                  .dot(vec_standarized))

    def set_covariance(self, propose_matrix):
        """
        Take covariance of sampled parameters (propose_matrix), and construct orthonormal
        parameters where orthonormal parameters are grouped in blocks by speed, so changes
        in slowest block changes slow and fast parameters, but changes in the fastest
        block only changes fast parameters

        :param propose_matrix: covariance matrix for the sampled parameters.
        """
        if propose_matrix.shape[0] != self.d():
            log.error("The covariance matrix does not have the correct dimension: "
                      "it's %d, but it should be %d.",
                      propose_matrix.shape[0], self.d())
            raise HandledException
        if not (np.allclose(propose_matrix.T, propose_matrix) and
                np.all(np.linalg.eigvals(propose_matrix) > 0)):
            log.error("The given covmat is not a positive-definite, "
                      "symmetric square matrix.")
            raise HandledException
        self.propose_matrix = propose_matrix.copy()
        propose_matrix_j_sorted = self.propose_matrix[np.ix_(self.i_of_j, self.i_of_j)]
        sigmas_diag = np.diag(np.sqrt(np.diag(propose_matrix_j_sorted)))
        invsigmas_diag = np.linalg.inv(sigmas_diag)
        corr = invsigmas_diag.dot(propose_matrix_j_sorted).dot(invsigmas_diag)
        L = np.linalg.cholesky(corr)
        # Store the basis as transformation matrices
        self.transform = []
        for iblock, bp in enumerate(self.proposer):
            j_start = self.j_start[iblock]
            j_end = j_start + bp.n
            self.transform += [sigmas_diag[j_start:, j_start:]
                                   .dot(L[j_start:, j_start:j_end])]

    def get_covariance(self):
        return self.propose_matrix.copy()

    def get_scale(self):
        return self.proposal_scale
