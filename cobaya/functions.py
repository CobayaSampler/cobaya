import logging

import numpy as np
import scipy

try:
    import numba
except (ImportError, SystemError):
    # SystemError caused usually by incompatible numpy version

    numba = None
    logging.debug("Numba not available, install it for better performance.")

    from scipy.stats import special_ortho_group

    random_SO_N = special_ortho_group.rvs

else:
    import warnings

    def random_SO_N(dim, *, random_state=None):
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

    logging.getLogger("numba").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        @numba.njit("void(int64,float64[::1],float64[:,::1])")
        def _rvs(dim, xx, H):
            D = np.empty((dim,))
            ix = 0
            for n in range(dim - 1):
                x = xx[ix : ix + dim - n]
                ix += dim - n
                norm2 = np.dot(x, x)
                x0 = x[0].item()
                D[n] = np.sign(x[0]) if x[0] != 0 else 1
                x[0] += D[n] * np.sqrt(norm2)
                x /= np.sqrt((norm2 - x0**2 + x[0] ** 2) / 2.0)
                # Householder transformation
                tmp = np.dot(H[:, n:], x)
                H[:, n:] -= np.outer(tmp, x)
            D[-1] = (-1) ** (dim - 1) * D[:-1].prod()
            H[:, :] = (D * H.T).T


def chi_squared(c_inv, delta):
    """
    Compute chi squared, i.e. delta.T @ c_inv @ delta

    :param c_inv: symmetric positive definite inverse covariance matrix
    :param delta: 1D array
    :return: delta.T @ c_inv @ delta
    """
    if len(delta) < 1500:
        return c_inv.dot(delta).dot(delta)
    else:
        # use symmetry
        return scipy.linalg.blas.dsymv(
            alpha=1.0, a=c_inv if np.isfortran(c_inv) else c_inv.T, x=delta, lower=0
        ).dot(delta)


def inverse_cholesky(cov):
    """
    Get inverse of Cholesky decomposition

    :param cov: symmetric positive definite matrix
    :return: L^{-1} where cov = L L^T
    """
    cholesky = np.linalg.cholesky(cov)
    return scipy.linalg.lapack.dtrtri(cholesky, lower=True)[0]
