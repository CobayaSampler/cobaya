"""
.. module:: log

:Synopsis: Manages logging and error handling
:Author: Jesus Torrado

"""

# Global
import os
import sys
import logging
import traceback
from copy import deepcopy
import functools

# Local
from cobaya import mpi


class LoggedError(Exception):
    """
    Dummy exception, to be raised when the originating exception
    has been cleanly handled and logged.
    """

    def __init__(self, logger, *args, **kwargs):
        if isinstance(logger, str):
            logger = get_logger(logger)
        if not isinstance(logger, logging.Logger):
            raise SyntaxError("The first argument of %s must be a logger "
                              "instance or name." %
                              self.__class__.__name__)
        if args:
            logger.error(*args, **kwargs)
        msg = args[0] if len(args) else ""
        if msg and len(args) > 1:
            msg = msg % args[1:]
        super().__init__(msg)


# Exceptions that will never be ignored when a component's calculation fails
always_stop_exceptions = (LoggedError, KeyboardInterrupt, SystemExit, NameError,
                          SyntaxError, AttributeError, KeyError, ImportError, TypeError)


def is_debug(log=None):
    log = log or logging.root
    return log.getEffectiveLevel() <= logging.DEBUG


def get_logger(name):
    if name.startswith('cobaya.'):
        name = name.split('.')[-1]
    return logging.getLogger(name)


def abstract(method):
    # abstract method decorator for base class HasLogger methods

    # If an @abstract method is called dynamically from another function,
    # you get a logged error that it's not implemented.
    # An @abstract methods also will not be picked up by the dependency analyser, so
    # a class with only an @abstract method implementation of X will not be assigned to
    # provide X. Descendants can of course override @abstract methods to implement them.

    @functools.wraps(method)
    def not_implemented(self, *args, **kwargs):
        if getattr(getattr(self, method.__name__, None), '_is_abstract', None):
            # OK to call if called via super, but not if not over-ridden
            raise LoggedError(self.log, "%s NotImplemented in %s", method.__name__,
                              self.__class__.__name__)
        else:
            return method(self, *args, **kwargs)

    not_implemented._is_abstract = True  # type: ignore

    return not_implemented


class NoLogging:
    def __init__(self, level=logging.WARNING):
        self._level = level

    def __enter__(self):
        if self._level:
            logging.disable(self._level)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        if self._level:
            logging.disable(logging.NOTSET)


def exception_handler(exception_type, exception_instance, trace_back):
    # Do not print traceback if the exception has been handled and logged by LoggedError
    # MPI abort if all processes don't raise exception in short timeframe (e.g. deadlock).
    want_abort = not mpi.time_out_barrier()
    _logger_name = "exception handler"
    log = logging.getLogger(_logger_name)

    if exception_type == LoggedError:
        # make show error easily visible at end of log
        if mpi.more_than_one_process():
            log.error(str(exception_instance))
        if want_abort:
            mpi.abort_if_mpi()
        if is_debug(log):
            return  # no traceback printed
    elif exception_type == mpi.OtherProcessError:
        log.info(str(exception_instance))
        if is_debug(log):
            return  # no traceback printed

    line = "-------------------------------------------------------------\n"
    log.critical(line[len(_logger_name) + 5:] + "\n" +
                 "".join(traceback.format_exception(
                     exception_type, exception_instance, trace_back)) +
                 line)
    if exception_type == KeyboardInterrupt:
        log.critical("Interrupted by the user.")
    elif is_debug(log):
        log.critical(
            "Some unexpected ERROR occurred. "
            "You can see the exception information above.\n"
            "We recommend trying to reproduce this error with '%s:True' in the input.\n"
            "If you cannot solve it yourself and need to report it, "
            "include the debug output,\n"
            "which you can send it to a file setting '%s:[some_file_name]'.",
            "debug", "debug_file")
    # Exit all MPI processes
    if want_abort:
        mpi.abort_if_mpi()


def logger_setup(debug=None, debug_file=None):
    """
    Configuring the root logger, for its children to inherit level, format and handlers.

    Level: if debug=True, take DEBUG. If numerical, use ""logging""'s corresponding level.
    Default: INFO
    """
    if debug is True or os.getenv('COBAYA_DEBUG'):
        level = logging.DEBUG
    elif debug in (False, None):
        level = logging.INFO
    else:
        level = debug
    # Set the default level, to make sure the handlers have a higher one
    logging.root.setLevel(level)
    debug = is_debug(logging.root)

    # Custom formatter
    class MyFormatter(logging.Formatter):
        def format(self, record):
            fmt = ((" %(asctime)s " if debug else "") +
                   "[" + ("%d : " % mpi.get_mpi_rank()
                          if mpi.more_than_one_process() else "") +
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


def get_traceback_text(exec_info):
    return "".join(["-"] * 20 + ["\n\n"] +
                   list(traceback.format_exception(*exec_info)) +
                   ["\n"] + ["-"] * 37)


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

    def is_debug(self):
        return is_debug(self.log)

    @mpi.root_only
    def mpi_warning(self, msg, *args, **kwargs):
        self.log.warning(msg, *args, **kwargs)

    @mpi.root_only
    def mpi_info(self, msg, *args, **kwargs):
        self.log.info(msg, *args, **kwargs)

    @mpi.root_only
    def mpi_debug(self, msg, *args, **kwargs):
        self.log.debug(msg, *args, **kwargs)
