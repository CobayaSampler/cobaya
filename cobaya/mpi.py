"""
.. module:: mpi

:Synopsis: Manages MPI parallelization transparently
:Author: Jesus Torrado

"""

import os
import functools

# Vars to keep track of MPI parameters
_mpi = None if os.environ.get('COBAYA_NOMPI', False) else -1
_mpi_size = -1
_mpi_comm = -1
_mpi_rank = -1


def set_mpi_disabled(disabled=True):
    """
    Disable MPI, e.g. for use on cluster head nodes where mpi4py may be installed
    but no MPI functions will work.
    """
    global _mpi, _mpi_size, _mpi_rank, _mpi_comm
    if disabled:
        _mpi = None
        _mpi_size = 0
        _mpi_comm = None
        _mpi_rank = None
    else:
        _mpi = -1
        _mpi_size = -1
        _mpi_comm = -1
        _mpi_rank = -1


# noinspection PyUnresolvedReferences
def get_mpi():
    """
    Import and returns the MPI object, or None if not running with MPI.

    Can be used as a boolean test if MPI is present.
    """
    global _mpi
    if _mpi == -1:
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
def is_main_process():
    """
    Returns true if primary process or MPI not available.
    """
    return not bool(get_mpi_rank())


def more_than_one_process():
    return bool(max(get_mpi_size(), 1) - 1)


def sync_processes():
    if get_mpi_size() > 1:
        get_mpi_comm().barrier()


def share_mpi(data=None, root=0):
    comm = get_mpi_comm()
    if comm and more_than_one_process():
        return comm.bcast(data, root=root)
    else:
        return data


def size() -> int:
    return get_mpi_size() or 1


def rank() -> int:
    return get_mpi_rank() or 0


def gather(data, root=0) -> list:
    comm = get_mpi_comm()
    if comm and more_than_one_process():
        return comm.gather(data, root=root)
    else:
        return [data]


def allgather(data) -> list:
    comm = get_mpi_comm()
    if comm and more_than_one_process():
        return comm.allgather(data)
    else:
        return [data]


def abort_if_mpi():
    """Closes all MPI process, if more than one present."""
    if get_mpi_size() > 1:
        get_mpi_comm().Abort(1)


# decorators to generalize functions/methods for mpi sharing

def root_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


def from_root(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return share_mpi(func(*args, **kwargs)
                         if is_main_process() else None)

    return wrapper


def set_from_root(attributes):
    atts = [attributes] if isinstance(attributes, str) else attributes

    def set_method(method):

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if is_main_process():
                result = method(self, *args, **kwargs)
                share_mpi([result] + [getattr(self, var, None) for var in atts])
            else:
                values = share_mpi()
                for name, var in zip(atts, values[1:]):
                    setattr(self, name, var)
                result = values[0]
            return result

        return wrapper

    return set_method
