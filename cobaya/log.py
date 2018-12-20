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
import logging
import traceback

# Local
from cobaya.conventions import _debug, _debug_file
from cobaya.mpi import get_mpi_rank, get_mpi_comm, more_than_one_process


class HandledException(Exception):
    """
    Dummy exception, to be raised when the originating exception
    has been cleanly handled and logged.
    """


def exception_handler(exception_type, value, trace_back):
    # Do nothing (just exit) if the exception has been handled and logged
    if exception_type == HandledException:
        # Exit all MPI processes
        getattr(get_mpi_comm(), "Abort", lambda: None)()
        return  # so that no traceback is printed
    log = logging.getLogger("exception handler")
    line = "------------------------------------------------\n"
    log.critical(line[6:] + "\n" +
                 "".join(traceback.format_exception(exception_type, value, trace_back)) +
                 line)
    if exception_type == KeyboardInterrupt:
        log.critical("Interrupted by the user.")
        return
    log.critical(
        "Some unexpected ERROR occurred. You can see the exception information above.\n"
        "We recommend trying to reproduce this error with '%s:True' in the input.\n"
        "If you cannot solve it yourself and need to report it, include the debug output,"
        "\nwhich you can send it to a file setting '%s:[some_file_name]'.",
        _debug, _debug_file)
    # Exit all MPI processes
    getattr(get_mpi_comm(), "Abort", lambda: None)()


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
            self._fmt = (
                    "[" + ("%d : " % get_mpi_rank() if more_than_one_process() else "") +
                    "%(name)s" + (" %(asctime)s " if debug else "") + "] " +
                    {logging.ERROR: "*ERROR* ",
                     logging.WARNING: "*WARNING* "}.get(record.levelno, "") +
                    "%(message)s")
            return logging.Formatter.format(self, record)

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
