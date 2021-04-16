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
from enum import IntEnum

default_error_timeout_seconds = 5

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
    return get_mpi_size() > 1


def sync_processes():
    if get_mpi_size() > 1:
        error_signal.check()
        get_mpi_comm().barrier()


def share_mpi(data=None, root=0):
    if get_mpi_size() > 1:
        return get_mpi_comm().bcast(data, root=root)
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
    if get_mpi_size() > 1:
        return get_mpi_comm().allgather(data)
    else:
        return [data]


def zip_gather(list_of_data, root=0) -> Iterable[tuple]:
    """
    Takes a list of items and returns a iterable of lists of items from each process
    e.g. for root node
    [(a_1, a_2),(b_1,b_2),...] = zip_gather([a,b,...])
    """
    if get_mpi_size() > 1:
        return zip(*(get_mpi_comm().gather(list_of_data, root=root) or [list_of_data]))
    else:
        return ((item,) for item in list_of_data)


def array_gather(list_of_data, root=0) -> List[np.array]:
    return [np.array(i) for i in zip_gather(list_of_data, root=root)]


# set if being run from pytest
capture_manager: Any = None


def abort_if_mpi(log=None, msg=None):
    """Closes all MPI process, if more than one present."""
    if get_mpi_size() > 1:
        if log and msg:
            log.critical(msg)
        if capture_manager:
            capture_manager.stop_global_capturing()
        get_mpi_comm().Abort(1)


_other_process_msg = "Another process failed - exiting."


class OtherProcessError(Exception):
    pass


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


def sync_errors(func):
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

def sync_error_signal(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not more_than_one_process():
            return func(*args, **kwargs)

        with Signal(func.__name__) as error, Signal(func.__name__, "sync") as sync:
            try:
                result = func(*args, **kwargs)
                error.check()
            except Exception as e:
                if not isinstance(e, OtherProcessError):
                    error.send()
                sync.send()
                if not time_out_barrier() and not isinstance(e, OtherProcessError):
                    # Handling errors here will also work in pytest which doesn't get
                    # to global handler
                    from cobaya.log import get_traceback_text, LoggedError
                    abort_if_mpi(logging.getLogger(func.__name__),
                                 "Aborting MPI deadlock (original error above)"
                                 if isinstance(e, LoggedError) else get_traceback_text(
                                     sys.exc_info()))
                raise
            else:
                sync.send(value=True)
                for i in range(size()):
                    if i != rank():
                        status = _mpi_comm.recv(source=i, tag=sync.tag)
                        if not status:
                            time_out_barrier()
                            raise OtherProcessError(_other_process_msg)
                return result

    return wrapper


# signalling

class State(IntEnum):
    END = 0
    READY = 1
    ERROR = 2


class SignalError(OtherProcessError):
    pass


_tags = []


class Signal:

    def __init__(self, owner='error_signal', name='error'):
        self.owner = owner
        self.name = name
        self.req = None
        if name not in _tags:
            _tags.append(name)
        self.tag = _tags.index(name) + 1
        self.requests = []

    @more_than_one
    def send(self, value=False):
        """
        Sends an error signal to the other MPI processes.
        """
        for i_rank in range(size()):
            if i_rank != rank():
                self.requests.append(
                    get_mpi_comm().isend(value, dest=i_rank, tag=self.tag).Test())

    @more_than_one
    def check(self):
        """
        Checks if any of the other process has sent an error signal, and raises an error.

        """
        if _mpi_comm.iprobe(source=_mpi.ANY_SOURCE, tag=self.tag):
            self.clear()
            self.fire()

    @more_than_one
    def clear(self):
        while _mpi_comm.iprobe(source=_mpi.ANY_SOURCE, tag=self.tag):
            _mpi_comm.recv(source=_mpi.ANY_SOURCE, tag=self.tag)
        # Unclear if this is needed or safe...
        # for req in self.requests:
        #    req.Cancel()
        self.requests = []

    def fire(self, cls=OtherProcessError, msg=_other_process_msg):
        raise cls("[%s: %s] %s" % (rank(), self.owner, msg))

    def __enter__(self):
        self.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()


error_signal = Signal()


class SyncError(SignalError):
    pass


# For testing and synchronizing processing when ready or ended/errored

class SyncState:

    def __init__(self, owner):
        self.statuses = np.empty(size(), dtype=int)
        self.owner = owner
        self.req = None
        self.error = Signal(owner)
        self.ending = False

    def set_ready(self, status=State.READY) -> bool:
        if self.req is None and not self.ending:
            self.owner.log.debug('Set process status: %s', status)
            self.req = get_mpi_comm().Iallgather(np.array([status], dtype=int),
                                                 self.statuses)
            return True
        return False

    @more_than_one
    def check(self, status=State.READY, wait=True) -> bool:
        # if have signalled ready, make sure don't leave others waiting fpr nothing
        if self.ending:
            return False
        was_set = self.set_ready(status)
        if wait or was_set:
            if status == State.ERROR:
                if not wait_for_request(self.req):
                    return False
            else:
                self.req.Wait()
        self.error.clear()
        self.req = None
        self.owner.log.debug('Got process statuses: %s', self.statuses)
        self.ending = any(self.statuses != State.READY)
        if status != State.ERROR and any(self.statuses == State.ERROR):
            self.error.fire(SyncError)
        if not was_set and not self.ending:
            # always need to sent at least one END or ERROR
            self.check(status)
        return not self.ending

    def all_ready(self) -> bool:
        return self.req is not None and self.req.Test() and self.check(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.check(State.END)
        elif exc_type and exc_type is not SyncError:
            self.error.send()
            self.check(State.ERROR)
        self.error.clear()
        self.req = None
