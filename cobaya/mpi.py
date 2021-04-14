"""
.. module:: mpi

:Synopsis: Manages MPI parallelization transparently
:Author: Jesus Torrado

"""

import os
import functools
from typing import List, Iterable
import numpy as np
from typing import Any

# Vars to keep track of MPI parameters
_mpi: Any = None if os.environ.get('COBAYA_NOMPI', False) else -1
_mpi_size = -1
_mpi_comm: Any = -1
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


def is_disabled():
    return _mpi is None


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


share = share_mpi


def size() -> int:
    return get_mpi_size() or 1


def rank() -> int:
    return get_mpi_rank() or 0


def gather(data, root=0) -> list:
    comm = get_mpi_comm()
    if comm and more_than_one_process():
        return comm.gather(data, root=root) or []
    else:
        return [data]


def allgather(data) -> list:
    comm = get_mpi_comm()
    if comm and more_than_one_process():
        return comm.allgather(data)
    else:
        return [data]


def zip_gather(list_of_data, root=0) -> Iterable[tuple]:
    """
    Takes a list of items and returns a iterable of lists of items from each process
    e.g. for root node
    [(a_1, a_2),(b_1,b_2),...] = zip_gather([a,b,...])
    """
    comm = get_mpi_comm()
    if comm and more_than_one_process():
        return zip(*(comm.gather(list_of_data, root=root) or [list_of_data]))
    else:
        return ((item,) for item in list_of_data)


def array_gather(list_of_data, root=0) -> List[np.array]:
    return [np.array(i) for i in zip_gather(list_of_data, root=root)]


def abort_if_mpi(log=None, msg=None):
    """Closes all MPI process, if more than one present."""
    if get_mpi_size() > 1:
        if log and msg:
            log.error(msg)
        get_mpi_comm().Abort(1)


class OtherProcessError(Exception):
    pass


_error_tag = 99


def send_error_signal(tag=_error_tag):
    """
    Sends an error signal to the other MPI processes.
    """
    for i_rank in range(size()):
        if i_rank != rank():
            get_mpi_comm().isend(True, dest=i_rank, tag=tag).Test()


def check_error_signal(log, tag=_error_tag, msg="Another process failed! Exiting."):
    """
    Checks if any of the other process has sent an error signal, and raises an error.

    """
    if more_than_one_process() and _mpi_comm.iprobe(source=_mpi.ANY_SOURCE, tag=tag):
        clear_error_signal(tag)
        raise OtherProcessError("[%s: %s] %s" % (rank(), log.name, msg))


def clear_error_signal(tag=_error_tag):
    if more_than_one_process():
        while _mpi_comm.iprobe(source=_mpi.ANY_SOURCE, tag=tag):
            _mpi_comm.recv(source=_mpi.ANY_SOURCE, tag=tag)


def time_out_barrier(time_out_seconds=4):
    import time
    req = _mpi_comm.Ibarrier()
    time_start = time.time()
    while not req.Test():
        time.sleep(0.01)
        if time.time() - time_start > time_out_seconds:
            return False
    return True


# decorators to generalize functions/methods for mpi sharing

def root_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


def more_than_one(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if more_than_one_process():
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


def synch_errors(func):
    err = 'Another process raised an error in %s' % func.__name__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception:
            allgather(True)
            raise
        else:
            if any(allgather(False)):
                raise OtherProcessError(err)
            return result

    return wrapper
