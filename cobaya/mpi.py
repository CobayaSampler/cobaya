"""
.. module:: mpi

:Synopsis: Manages MPI parallelization transparently
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
import os
# Local
from cobaya.conventions import _package

# Vars to keep track of MPI parameters
_mpi = -1
_mpi_size = -1
_mpi_comm = -1
_mpi_rank = -1

disabled = os.environ.get('COBAYA_NOMPI', False)


def get_mpi():
    """
    Import and returns the MPI object, or None if not running with MPI.

    Can be used as a boolean test if MPI is present.
    """
    global _mpi
    if _mpi == -1:
        if disabled:
            _mpi = None
            return None
        try:
            from mpi4py import MPI
            _mpi = MPI
        except ImportError:
            _mpi = None
    return _mpi


def get_mpi_size():
    """
    Returns the number of MPI processes that have been invoked,
    or 0 if not running with MPI.
    """
    global _mpi_size
    if _mpi_size == -1:
        _mpi_size = getattr(get_mpi_comm(), "Get_size", lambda: 0)()
    return _mpi_size


def get_mpi_comm():
    """
    Returns the MPI communicator, or `None` if not running with MPI.
    """
    global _mpi_comm
    if _mpi_comm == -1:
        _mpi_comm = getattr(get_mpi(), "COMM_WORLD", None)
    return _mpi_comm


def get_mpi_rank():
    """
    Returns the rank of the current MPI process:
        * None: not running with MPI
        * Z>=0: process rank, when running with MPI

    Can be used as a boolean that returns `False` for both the root process,
    if running with MPI, or always for a single process; thus, everything under
    `if not(get_mpi_rank()):` is run only *once*.
    """
    global _mpi_rank
    if _mpi_rank == -1:
        _mpi_rank = getattr(get_mpi_comm(), "Get_rank", lambda: None)()
    return _mpi_rank


# Aliases for simpler use
def am_single_or_primary_process(no_mpi=False):
    """Returns true if primary process or MPI not available.

    Use the no_mpi keyword to avoid checking for MPI via import, 
    which can die ungracefully (e.g. on head nodes at NERSC).
    """
    if not no_mpi:
        return not bool(get_mpi_rank())
    else:
        return True


def more_than_one_process(no_mpi=False):
    if not no_mpi:
        return bool(max(get_mpi_size(), 1) - 1)
    else:
        return False


def sync_processes():
    if get_mpi_size() > 1:
        get_mpi_comm().barrier()


def import_MPI(module, target):
    """Import helper for MPI wrappers."""
    from importlib import import_module
    target_name = target
    if get_mpi_rank() is not None:
        target_name = target + "_MPI"
    return getattr(import_module(module, package=_package), target_name)
