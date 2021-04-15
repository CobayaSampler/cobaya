"""
.. module:: mpi

:Synopsis: Manages MPI parallelization transparently
:Author: Jesus Torrado

"""

import os
import sys
import functools
from typing import List, Iterable
import numpy as np
from typing import Any
import logging
import time

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
            log.critical(msg)
        get_mpi_comm().Abort(1)


class OtherProcessError(Exception):
    pass


_error_tag = 99
_synch_tag = _error_tag + 1

default_error_timeout_seconds = 5


def send_signal(tag=_error_tag, value=False):
    """
    Sends an error signal to the other MPI processes.
    """
    for i_rank in range(size()):
        if i_rank != rank():
            get_mpi_comm().isend(value, dest=i_rank, tag=tag).Test()


def check_error_signal(log, tag=_error_tag, msg="Another process failed! Exiting."):
    """
    Checks if any of the other process has sent an error signal, and raises an error.

    """
    if more_than_one_process() and _mpi_comm.iprobe(source=_mpi.ANY_SOURCE, tag=tag):
        clear_signal(tag)
        raise OtherProcessError(
            "[%s: %s] %s" % (rank(), log if isinstance(log, str) else log.name, msg))


def clear_signal(tag=_error_tag):
    if more_than_one_process():
        while _mpi_comm.iprobe(source=_mpi.ANY_SOURCE, tag=tag):
            _mpi_comm.recv(source=_mpi.ANY_SOURCE, tag=tag)


def wait_for_request(req, time_out_seconds=default_error_timeout_seconds, interval=0.01):
    time_start = time.time()
    while not req.Test():
        time.sleep(interval)
        if time.time() - time_start > time_out_seconds:
            return False
    return True


def time_out_barrier(time_out_seconds=default_error_timeout_seconds):
    if more_than_one_process():
        return wait_for_request(_mpi_comm.Ibarrier(), time_out_seconds)
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
        if is_main_process():
            try:
                result = func(*args, **kwargs)
            except Exception:
                share_mpi()
                raise
            else:
                share_mpi([result])
                return result
        else:
            result = share_mpi()
            if result is None:
                raise OtherProcessError('Root errored in %s' % func.__name__)
            return result[0]

    return wrapper


def set_from_root(attributes):
    atts = [attributes] if isinstance(attributes, str) else attributes

    def set_method(method):

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if is_main_process():
                try:
                    result = method(self, *args, **kwargs)
                except Exception:
                    share_mpi()
                    raise
                else:
                    share_mpi([result] + [getattr(self, var, None) for var in atts])
            else:
                values = share_mpi()
                if values is None:
                    raise OtherProcessError('Root errored in %s' % method.__name__)
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


def abort_if_test(log, exc_info):
    if "PYTEST_CURRENT_TEST" in os.environ and more_than_one_process():
        # in pytest, never gets to the system hook to kill mpi so do it here
        # (mpi.abort_if_mpi is replaced by conftest.py::mpi_handling session fixture)
        from cobaya.log import get_traceback_text
        abort_if_mpi(log, get_traceback_text(exc_info))


# Wrapper for main functions. Traps MPI deadlock via timeout MPI_ABORT if needed,

def synch_error_signal(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        clear_signal()
        clear_signal(_synch_tag)
        log = logging.getLogger(func.__name__)
        try:
            result = func(*args, **kwargs)
            check_error_signal(func.__name__)
        except Exception as e:
            if more_than_one_process():
                if not isinstance(e, OtherProcessError):
                    send_signal()
                send_signal(tag=_synch_tag)
                if not time_out_barrier() and not isinstance(e, OtherProcessError):
                    # Handling errors here will also work in pytest which doesn't get
                    # to global handler
                    from cobaya.log import get_traceback_text, LoggedError
                    abort_if_mpi(log,
                                 "Aborting MPI deadlock (original error above)"
                                 if isinstance(e, LoggedError) else get_traceback_text(
                                     sys.exc_info()))
                clear_signal()
                clear_signal(_synch_tag)
                raise
        else:
            if more_than_one_process():
                send_signal(value=True, tag=_synch_tag)
                for i in range(size()):
                    if i != rank():
                        status = _mpi_comm.recv(source=i, tag=_synch_tag)
                        if not status:
                            time_out_barrier()
                            clear_signal(_synch_tag)
                            raise OtherProcessError('Another process failed - exiting.')

            return result

    return wrapper
