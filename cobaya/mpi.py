"""
.. module:: mpi

:Synopsis: Manages MPI parallelization transparently
:Author: Jesus Torrado

"""

import functools
import os
import sys
import time
from collections.abc import Callable, Iterable
from enum import IntEnum
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")

default_error_timeout_seconds = 5

# Vars to keep track of MPI parameters
_mpi: Any = None if os.environ.get("COBAYA_NOMPI", False) else -1
_mpi_size = -1
_mpi_comm: Any = -1
_mpi_rank: int | None = -1


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
        else:
            if more_than_one_process():
                try:
                    import dill
                except ImportError:
                    pass
                else:
                    _mpi.pickle.__init__(dill.dumps, dill.loads)

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


def check_errors():
    if process_state:
        process_state.check_error()


def sync_processes():
    if get_mpi_size() > 1:
        check_errors()
        get_mpi_comm().barrier()


def share_mpi(data: T = None, root=0) -> T:
    if get_mpi_size() > 1:
        return get_mpi_comm().bcast(data, root=root)
    else:
        return data


share = share_mpi


def scatter(data=None, root=0):
    if get_mpi_size() > 1:
        return get_mpi_comm().scatter(data, root=root)
    else:
        return data[0]


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
    Takes a list of items and returns an iterable of lists of items from each process
    e.g. for root node
    [(a_1, a_2),(b_1,b_2),...] = zip_gather([a,b,...])
    """
    if get_mpi_size() > 1:
        return zip(*(get_mpi_comm().gather(list_of_data, root=root) or [list_of_data]))
    else:
        return ((item,) for item in list_of_data)


def array_gather(list_of_data, root=0) -> list[np.ndarray]:
    return [np.array(i) for i in zip_gather(list_of_data, root=root)]


# set if being run from pytest
capture_manager: Any = None


def abort_if_mpi():
    """Closes all MPI process, if more than one present."""
    if get_mpi_size() > 1:
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
                raise OtherProcessError("Root errored in %s" % func.__name__)
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
                    raise OtherProcessError("Root errored in %s" % method.__name__)
                for name, var in zip(atts, values[1:]):
                    setattr(self, name, var)
                result = values[0]
            return result

        return wrapper

    return set_method


def sync_errors(func):
    err = "Another process raised an error in %s" % func.__name__

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


# Wrapper for main functions. Traps MPI deadlock via timeout MPI_ABORT if needed,


def sync_state(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not more_than_one_process():
            return func(*args, **kwargs)

        with ProcessState(func.__name__):
            return func(*args, **kwargs)

    return wrapper


# Synchronization between processes, generating OtherProcessError in other threads if
# any one process raises an exception


class State(IntEnum):
    NONE = 0
    READY = 1
    END = 2
    ERROR = 3


class SyncError(OtherProcessError):
    pass


# log = logging.getLogger('state')
log: Any = None


class ProcessState:
    _id: int = 1

    def __init__(
        self,
        name="error",
        time_out_seconds: float | int = default_error_timeout_seconds,
        sleep_interval=0.01,
        timeout_abort_proc: Callable = abort_if_mpi,
    ):
        self.name = str(name)
        ProcessState._id += 1
        self.tag = ProcessState._id
        self.states = np.empty(size(), dtype=int)
        self.sleep_interval = sleep_interval
        self.time_out_seconds = time_out_seconds
        self._recv_state = np.empty(1, dtype=int)
        self._rank = rank()
        self._others = [i for i in range(size()) if i != self._rank]
        self.timeout_abort_proc = timeout_abort_proc

    def set(self, value: State):
        """
        Sends an error signal to the other MPI processes.
        """
        if self.states[self._rank] != value:
            if log:
                log.info("SET %s %s %s", self.name, self.tag, value)
            self.states[self._rank] = value
            for i_rank in self._others:
                _mpi_comm.Isend(
                    self.states[self._rank : self._rank + 1], dest=i_rank, tag=self.tag
                ).Test()
            return True
        else:
            return False

    @more_than_one
    def sync(self, check_error=False):
        """
        Gets any messages from other processes without waiting, and optionally
        raises error if any others are in error state.
        """
        while _mpi_comm.iprobe(source=_mpi.ANY_SOURCE, tag=self.tag):
            status = _mpi.Status()
            _mpi_comm.Recv(
                self._recv_state, source=_mpi.ANY_SOURCE, tag=self.tag, status=status
            )
            state = self._recv_state[0]
            self.states[status.Get_source()] = state
            if check_error and state == State.ERROR:
                self.fire_error(SyncError)
            if log:
                log.info("SYNC %s %s %s", self.name, self.tag, self.states)

    def check_error(self):
        """
        Raises error if any other processes in error state
        """
        self.sync(check_error=True)

    def fire_error(self, cls=OtherProcessError, msg=_other_process_msg):
        raise cls(f"[{rank()}: {self.name}] {msg}")

    def wait_all_ended(self, timeout=False):
        """
        Wait until all processes in ERROR or END state.
        """
        self.sync()
        time_start = time.time()
        while any(self.states < State.END):
            time.sleep(self.sleep_interval)
            if timeout and time.time() - time_start > self.time_out_seconds:
                return False
            self.sync()
        return True

    def all_ready(self) -> bool:
        """
        Test if all processes in READY state (and if they are reset to NONE).
        """
        self.sync(check_error=True)
        if all_ready := bool(np.all(self.states == State.READY)):
            self.states[:] = State.NONE
        return all_ready

    def __enter__(self):
        if more_than_one_process():
            self.tag = share(self.tag)
            self.last_process_state = process_state
            set_current_process_state(self)
        self.states[:] = State.NONE
        return self

    @more_than_one
    def __exit__(self, exc_type, exc_val, exc_tb):
        if log:
            log.info("END %s %s", self.name, self.tag)
        if exc_type:
            self.set(State.ERROR)
            if not self.wait_all_ended(
                timeout=not issubclass(exc_type, OtherProcessError)
            ):
                from cobaya.log import LoggedError, get_logger, get_traceback_text

                get_logger(self.name).critical(
                    "Aborting MPI due to error"
                    if issubclass(exc_type, LoggedError)
                    else get_traceback_text(sys.exc_info())
                )
                self.timeout_abort_proc()
                self.wait_all_ended()  # if didn't actually MPI abort
        else:
            self.set(State.END)
            self.wait_all_ended()
        set_current_process_state(self.last_process_state)
        if not exc_type and State.ERROR in self.states:
            self.fire_error()


process_state: ProcessState | None = None


def set_current_process_state(state: ProcessState | None):
    global process_state
    process_state = state
