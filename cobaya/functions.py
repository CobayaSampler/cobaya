import numpy as np
import logging

try:
    import numba
except (ImportError, SystemError):
    # SystemError caused usually by incompatible numpy version

    numba = None
    logging.debug("Numba not available, install it for better performance.")

    from scipy.stats import special_ortho_group

    random_SO_N = special_ortho_group.rvs

    _fast_chi_squared = None
    _sym_chi_squared = None

else:
    import warnings


    def random_SO_N(dim, random_state):
        """
        Draw random samples from SO(N).
        Equivalent to scipy function but about 10x faster
        Parameters
        ----------
        dim : integer
            Dimension of rotation space (N).
        random_state: generator
        Returns
        -------
        rvs : Random size N-dimensional matrices, dimension (dim, dim)

        """
        dim = np.int64(dim)
        H = np.eye(dim)
        xx = random_state.standard_normal(size=(dim + 2) * (dim - 1) // 2)
        _rvs(dim, xx, H)
        return H


    logging.getLogger('numba').setLevel(logging.ERROR)
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


        @numba.njit(parallel=True)
        def _fast_chi_squared(c_inv, delta):
            """
            Calculate chi-squared from inverse matrix and delta vector,
            using symmetry. Note parallel is slower for small sizes.

            """
            n = delta.shape[0]
            chi2 = 0.0

            for j in numba.prange(n):
                z_temp = np.dot(c_inv[j, j + 1:], delta[j + 1:])
                chi2 += (2 * z_temp + c_inv[j, j] * delta[j]) * delta[j]

            return chi2


def chi_squared(c_inv, delta):
    if len(delta) < 1500 or not _fast_chi_squared:
        return c_inv.dot(delta).dot(delta)
    else:
        return _fast_chi_squared(c_inv, delta)
