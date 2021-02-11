"""
.. module:: log

:Synopsis: Manages logging and error handling
:Author: Jesus Torrado

"""

# Global
import sys
import logging
import traceback
from copy import deepcopy

# Local
from cobaya.conventions import _debug, _debug_file
from cobaya.mpi import get_mpi_rank, get_mpi_size, get_mpi_comm, \
    more_than_one_process, is_main_process


class LoggedError(Exception):
    """
    Dummy exception, to be raised when the originating exception
    has been cleanly handled and logged.
    """

    def __init__(self, logger, *args, **kwargs):
        if args:
            logger.error(*args, **kwargs)
        msg = args[0] if len(args) else ""
        if msg and len(args) > 1:
            msg = msg % args[1:]
        super().__init__(msg)


# Exceptions that will never be ignored when a component's calculation fails
always_stop_exceptions = (LoggedError, KeyboardInterrupt, SystemExit, NameError,
                          SyntaxError, AttributeError, KeyError)


def safe_exit():
    """Closes all MPI process, if more than one present."""
    if get_mpi_size() > 1:
        get_mpi_comm().Abort(1)


def exception_handler(exception_type, exception_instance, trace_back):
    # Do not print traceback if the exception has been handled and logged
    if exception_type == LoggedError:
        safe_exit()
        return  # no traceback printed
    _logger_name = "exception handler"
    log = logging.getLogger(_logger_name)
    line = "-------------------------------------------------------------\n"
    log.critical(line[len(_logger_name) + 5:] + "\n" +
                 "".join(traceback.format_exception(
                     exception_type, exception_instance, trace_back)) +
                 line)
    if exception_type == KeyboardInterrupt:
        log.critical("Interrupted by the user.")
    else:
        log.critical(
            "Some unexpected ERROR occurred. "
            "You can see the exception information above.\n"
            "We recommend trying to reproduce this error with '%s:True' in the input.\n"
            "If you cannot solve it yourself and need to report it, "
            "include the debug output,\n"
            "which you can send it to a file setting '%s:[some_file_name]'.",
            _debug, _debug_file)
    # Exit all MPI processes
    safe_exit()


def logger_setup(debug=None, debug_file=None):
    """
    Configuring the root logger, for its children to inherit level, format and handlers.

    Level: if debug=True, take DEBUG. If numerical, use "logging"'s corresponding level.
    Default: INFO
    """
    if debug is True:
        level = logging.DEBUG
    elif debug in (False, None):
        level = logging.INFO
    else:
        level = int(debug)
    # Set the default level, to make sure the handlers have a higher one
    logging.root.setLevel(level)

    # Custom formatter
    class MyFormatter(logging.Formatter):
        def format(self, record):
            fmt = ((" %(asctime)s " if debug else "") +
                   "[" + ("%d : " % get_mpi_rank() if more_than_one_process() else "") +
                   "%(name)s" + "] " +
                   {logging.ERROR: "*ERROR* ",
                    logging.WARNING: "*WARNING* "}.get(record.levelno, "") +
                   "%(message)s")
            self._style._fmt = fmt
            return super().format(record)

    # Configure stdout handler
    handle_stdout = logging.StreamHandler(sys.stdout)
    handle_stdout.setLevel(level)
    handle_stdout.setFormatter(MyFormatter())
    # log file? Create and reduce stdout level to INFO
    if debug_file:
        file_stdout = logging.FileHandler(debug_file, mode="w")
        file_stdout.setLevel(level)
        handle_stdout.setLevel(logging.INFO)
        file_stdout.setFormatter(MyFormatter())
        logging.root.addHandler(file_stdout)
    # Add stdout handler only once!
    # noinspection PyUnresolvedReferences
    try:
        stdout_handler = next(
            h for h in logging.root.handlers if getattr(h, "stream", None) == sys.stdout)
        # If there is one, update it's logging level and formatter
        stdout_handler.setLevel(handle_stdout.level)
    except StopIteration:
        # If there is none, add it!
        logging.root.addHandler(handle_stdout)
    # Configure the logger to manage exceptions
    sys.excepthook = exception_handler


class HasLogger:
    """
    Class having a logger with its name (or an alternative one).

    Has magic methods to ignore the logger at (de)serialization.
    """

    def set_logger(self, lowercase=True, name=None):
        name = name or self.__class__.__name__
        self.log = logging.getLogger(name.lower() if lowercase else name)

    # Copying and pickling
    def __deepcopy__(self, memo=None):
        new = (lambda cls: cls.__new__(cls))(self.__class__)
        new.__dict__ = {k: deepcopy(v) for k, v in self.__dict__.items() if k != "log"}
        return new

    def __getstate__(self):
        return deepcopy(self).__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        self.set_logger()

    def mpi_warning(self, msg, *args, **kwargs):
        if is_main_process():
            self.log.warning(msg, *args, **kwargs)

    def mpi_info(self, msg, *args, **kwargs):
        if is_main_process():
            self.log.info(msg, *args, **kwargs)
