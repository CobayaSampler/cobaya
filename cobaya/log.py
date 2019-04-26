"""
.. module:: log

:Synopsis: Manages logging and error handling
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import sys
import six
import logging
import traceback

# Local
from cobaya.conventions import _debug, _debug_file
from cobaya.mpi import get_mpi_rank, get_mpi_size, get_mpi_comm, more_than_one_process


class HandledException(Exception):
    """
    Dummy exception, to be raised when the originating exception
    has been cleanly handled and logged.
    """


def safe_exit():
    """Closes all MPI process, if more than one present."""
    if get_mpi_size() > 1:
        get_mpi_comm().Abort()


def exception_handler(exception_type, value, trace_back):
    # Do not print traceback if the exception has been handled and logged
    if exception_type == HandledException:
        safe_exit()
        return  # no traceback printed
    _logger_name = "exception handler"
    log = logging.getLogger(_logger_name)
    line = "-------------------------------------------------------------\n"
    log.critical(line[len(_logger_name)+5:] + "\n" +
                 "".join(traceback.format_exception(exception_type, value, trace_back)) +
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
            if six.PY3:
                self._style._fmt = fmt
            else:
                self._fmt = fmt
            return super(MyFormatter, self).format(record)

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
    if not any(h.stream == sys.stdout for h in logging.root.handlers):
        logging.root.addHandler(handle_stdout)
    # Configure the logger to manage exceptions
    sys.excepthook = exception_handler
