"""
.. module:: mpi

:Synopsis: Manages MPI parallelization transparently
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import os

# Local
from cobaya.conventions import package

# Vars to keep track of MPI parameters
_mpi = -1
_mpi_size = -1
_mpi_comm = -1
_mpi_rank = -1


def get_mpi():
    """
    Import and returns the MPI object, or None of not running with MPI.

    Can be used as a boolean test if MPI is present.
    """
    global _mpi
    if _mpi == -1:
        if (os.environ.get("OMPI_COMM_WORLD_SIZE") or  # OpenMPI
            os.environ.get("MPIR_CVAR_CH3_INTERFACE_HOSTNAME")):  # MPICH
            from mpi4py import MPI
            _mpi = MPI
        else:
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


def import_MPI(module, target):
    """Import helper for MPI wrappers."""
    from importlib import import_module
    target_name = target
    if get_mpi_rank() is not None:
        target_name = target+"_MPI"
    return getattr(import_module(module, package=package), target_name)
