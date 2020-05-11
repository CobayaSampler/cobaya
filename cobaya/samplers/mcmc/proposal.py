"""
.. module:: samplers.mcmc.proposal

:Synopsis: proposal distributions
:Author: Antony Lewis and Jesus Torrado

Using the covariance matrix to give the proposal directions typically
significantly increases the acceptance rate and gives faster movement
around parameter space.

We generate a random basis in the eigenvectors, then cycle through them
proposing changes to each, then generate a new random basis.
The distance proposal in the random direction is given by a two-D Gaussian
radial function mixed with an exponential, which is quite robust to wrong width estimates

See https://arxiv.org/abs/1304.4473
"""

# Global
import numpy as np
from itertools import chain

# Local
from cobaya.tools import choleskyL
from cobaya.log import LoggedError, HasLogger


class IndexCycler:
    def __init__(self, n):
        self.n = n
        self.loop_index = -1


class CyclicIndexRandomizer(IndexCycler):
    def __init__(self, n):
        if isinstance(n, int):
            self.sorted_indices = list(range(n))
        else:
            self.sorted_indices = n
            n = len(n)
        super().__init__(n)
        if n <= 2:
            self.indices = list(range(n))

    def next(self):
        """
        Get the next random index, or alternate for two or less.

        :return: index
        """
        self.loop_index = (self.loop_index + 1) % self.n
        if self.loop_index == 0 and self.n > 2:
            self.indices = np.random.permutation(self.sorted_indices)
        return self.indices[self.loop_index]


try:
    import numba
except ImportError:
    from scipy.stats import special_ortho_group

    random_SO_N = special_ortho_group.rvs
    numba = None
else:
    from numpy.random import normal
    import warnings


    def random_SO_N(dim):
        """
        Draw random samples from SO(N).
        Equivalent to scipy function but about 10x faster
        Parameters
        ----------
        dim : integer
            Dimension of rotation space (N).
        Returns
        -------
        rvs : Random size N-dimensional matrices, dimension (dim, dim)

        """
        dim = np.int64(dim)
        H = np.eye(dim)
        xx = normal(size=(dim + 2) * (dim - 1) // 2)
        _rvs(dim, xx, H)
        return H


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")


        @numba.njit("void(int64,float64[::1],float64[:,::1])")
        def _rvs(dim, xx, H):
            D = np.empty((dim,))
            ix = 0
            for n in range(dim - 1):
                x = xx[ix:ix + dim - n]
                ix += dim - n
                norm2 = np.dot(x, x)
                x0 = x[0].item()
                D[n] = np.sign(x[0]) if x[0] != 0 else 1
                x[0] += D[n] * np.sqrt(norm2)
                x /= np.sqrt((norm2 - x0 ** 2 + x[0] ** 2) / 2.)
                # Householder transformation
                tmp = np.dot(H[:, n:], x)
                H[:, n:] -= np.outer(tmp, x)
            D[-1] = (-1) ** (dim - 1) * D[:-1].prod()
            H[:, :] = (D * H.T).T


class RandDirectionProposer(IndexCycler):
    def propose_vec(self, scale: float = 1):
        """
        propose a random n-dimension vector for n>1

        :param scale: units for the distance
        :return: array with vector
        """
        self.loop_index = (self.loop_index + 1) % self.n
        if self.loop_index == 0:
            self.R = random_SO_N(self.n)
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


class RandProposer1D(RandDirectionProposer):
    def propose_vec(self, scale: float = 1):
        return np.array([self.propose_r() * scale if np.random.randint(2)
                         else -self.propose_r() * scale])


class BlockedProposer(HasLogger):
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
        self.set_logger(lowercase=True)
        self.proposal_scale = proposal_scale
        if oversampling_factors is None:
            self.oversampling_factors = np.ones(len(parameter_blocks), dtype=int)
        else:
            if len(oversampling_factors) != len(parameter_blocks):
                raise LoggedError(
                    self.log, "List of oversampling factors has a different length that "
                              "list of blocks: %d vs %d.",
                    len(oversampling_factors), len(parameter_blocks))
            if np.any(oversampling_factors != np.floor(np.array(oversampling_factors))):
                raise LoggedError(
                    self.log, "Oversampling factors must be integer! Got %r.",
                    oversampling_factors)
            self.oversampling_factors = np.array(oversampling_factors, dtype=int)
        # Binary fast-slow split
        self.i_last_slow_block = i_last_slow_block
        if self.i_last_slow_block is None:
            self.i_last_slow_block = len(parameter_blocks) - 1
        else:
            if self.i_last_slow_block > len(parameter_blocks) - 1:
                raise LoggedError(
                    self.log,
                    "The index given for the last slow block, %d, is not valid: "
                    "there are only %d blocks.",
                    self.i_last_slow_block, len(parameter_blocks))
        n_all = sum(len(b) for b in parameter_blocks)
        n_slow = sum(len(b) for b in parameter_blocks[:1 + self.i_last_slow_block])
        self.nsamples_slow = 0
        self.nsamples_fast = 0
        if set(chain(*parameter_blocks)) != set(range(n_all)):
            raise LoggedError(self.log,
                              "The blocks do not contain all the parameter indices.")
        # Prepare indices for the cycler, repeated if there is oversampling
        self.n_block = np.array([len(b) for b in parameter_blocks])
        indices_repeated = list(chain(
            *[list(chain(*[[p] * o for p in block]))
              for block, o in zip(parameter_blocks, oversampling_factors)]))
        # Mapping between internal indices, sampler parameter indices and blocks:
        # let i=0,1,... be the indices of the parameters for the sampler,
        # and j=0,1,... be the indices of the parameters as the proposer manages them
        # i.e. if passed blocks=[[1,2],[0]] (those are the i's),
        # then the j's are [0 (for 1), 1 (for 2), 2 (for 0)].
        # iblock is the index of the blocks, which in term of j indices is simply
        # [0,0,1] in this example
        self.i_of_j = np.array(list(chain(*parameter_blocks)))
        self.iblock_of_j = list(
            chain(*[[iblock] * len(b) for iblock, b in enumerate(parameter_blocks)]))
        # Creating the blocked proposers
        self.proposer = [(RandDirectionProposer(len(b)) if len(b) > 1
                          else RandProposer1D(1)) for b in parameter_blocks]
        # Starting j index of each block
        self.j_start = [len(list(chain(*parameter_blocks[:iblock])))
                        for iblock, b in enumerate(parameter_blocks)]
        # Parameter cyclers, cycling over the j's
        self.parameter_cycler = CyclicIndexRandomizer(indices_repeated)
        # These ones are used by fast dragging only
        self.parameter_cycler_slow = CyclicIndexRandomizer(n_slow)
        self.parameter_cycler_fast = CyclicIndexRandomizer(n_all - n_slow)

    def d(self):
        return len(self.i_of_j)

    def get_proposal(self, P):
        self.current_iblock = self.iblock_of_j[self.parameter_cycler.next()]
        if self.current_iblock <= self.i_last_slow_block:
            self.nsamples_slow += 1
        else:
            self.nsamples_fast += 1
        self.get_block_proposal(P, self.current_iblock)

    def get_proposal_slow(self, P):
        current_iblock_slow = self.iblock_of_j[self.parameter_cycler_slow.next()]
        self.nsamples_slow += 1
        self.get_block_proposal(P, current_iblock_slow)

    def get_proposal_fast(self, P):
        self.nsamples_fast += 1
        current_iblock_fast = self.iblock_of_j[self.parameter_cycler_slow.n
                                               + self.parameter_cycler_fast.next()]
        self.get_block_proposal(P, current_iblock_fast)

    def get_block_proposal(self, P, iblock):
        vec_standardized = self.proposer[iblock].propose_vec(self.proposal_scale)
        P[self.i_of_j[self.j_start[iblock]:]] += (self.transform[iblock]
                                                  .dot(vec_standardized))

    def set_covariance(self, propose_matrix):
        """
        Take covariance of sampled parameters (propose_matrix), and construct orthonormal
        parameters where orthonormal parameters are grouped in blocks by speed, so changes
        in slowest block changes slow and fast parameters, but changes in the fastest
        block only changes fast parameters

        :param propose_matrix: covariance matrix for the sampled parameters.
        """
        if propose_matrix.shape[0] != self.d():
            raise LoggedError(
                self.log, "The covariance matrix does not have the correct dimension: "
                          "it's %d, but it should be %d.", propose_matrix.shape[0],
                self.d())
        if not (np.allclose(propose_matrix.T, propose_matrix) and
                np.all(np.linalg.eigvals(propose_matrix) > 0)):
            raise LoggedError(self.log, "The given covmat is not a positive-definite, "
                                        "symmetric square matrix.")
        self.propose_matrix = propose_matrix.copy()
        propose_matrix_j_sorted = self.propose_matrix[np.ix_(self.i_of_j, self.i_of_j)]
        sigmas_diag, L = choleskyL(propose_matrix_j_sorted, return_scale_free=True)
        # Store the basis as transformation matrices
        self.transform = []
        for j_start, bp in zip(self.j_start, self.proposer):
            j_end = j_start + bp.n
            self.transform += [sigmas_diag[j_start:, j_start:].dot(L[j_start:,
                                                                   j_start:j_end])]

    def get_covariance(self):
        return self.propose_matrix.copy()

    def get_scale(self):
        return self.proposal_scale
