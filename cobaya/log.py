"""
.. module:: log

:Synopsis: Manages logging and error handling
:Author: Jesus Torrado

"""

import functools
import logging
import os
import platform
import sys
import traceback
from random import choice, shuffle

import numpy as np

from cobaya import mpi


class LoggedError(Exception):
    """
    Dummy exception, to be raised when the originating exception
    has been cleanly handled and logged.

    Prints the error message even if caught.
    """

    def __init__(self, logger, *args, **kwargs):
        if isinstance(logger, str):
            logger = get_logger(logger)
        if not isinstance(logger, logging.Logger):
            raise SyntaxError(
                "The first argument of %s must be a logger "
                "instance or name." % self.__class__.__name__
            )
        if args:
            # If the exception is going to be caught, we may not want to print the msg
            # at logger.error level, but e.g. debug level.
            level = kwargs.pop("level", "error") or "error"
            getattr(logger, level)(*args, **kwargs)
        msg = args[0] if len(args) else ""
        if msg and len(args) > 1:
            msg = msg % args[1:]
        super().__init__(msg)


# Exceptions that will never be ignored when a component's calculation fails
always_stop_exceptions = (
    LoggedError,
    KeyboardInterrupt,
    SystemExit,
    NameError,
    SyntaxError,
    AttributeError,
    KeyError,
    ImportError,
    TypeError,
)


def is_debug(log=None):
    log = log or logging.root
    return log.getEffectiveLevel() <= logging.DEBUG


def get_logger(name):
    if name.startswith("cobaya."):
        name = name.split(".")[-1]
    return logging.getLogger(add_color_to_name(name))


# Some legible color combinations
color_strs = {
    "red_bold": "\x1b[31;1m",
    "green_bold": "\x1b[32;1m",
    "yellow_bold": "\x1b[33;1m",
    "blue_bold": "\x1b[34;1m",
    "magenta_bold": "\x1b[35;1m",
    "cyan_bold": "\x1b[36;1m",
    "light_red_bold": "\x1b[91;1m",
    "light_green_bold": "\x1b[92;1m",
    "light_yellow_bold": "\x1b[93;1m",
    "light_blue_bold": "\x1b[94;1m",
    "light_magenta_bold": "\x1b[95;1m",
    "light_cyan_bold": "\x1b[96;1m",
    # With background
    "light_grey_on_red_bold": "\x1b[37;1;41m",
    "light_grey_on_green_bold": "\x1b[37;1;42m",
    "light_grey_on_yellow_bold": "\x1b[37;1;43m",
    "light_grey_on_blue_bold": "\x1b[37;1;44m",
    "light_grey_on_magenta_bold": "\x1b[37;1;45m",
    "light_grey_on_cyan_bold": "\x1b[37;1;46m",
    "blue_on_light_green_bold": "\x1b[34;1;102m",
    "blue_on_light_yellow_bold": "\x1b[34;1;103m",
    "light_yellow_on_blue_bold": "\x1b[93;1;44m",
    "blue_on_light_cyan": "\x1b[34;1;106m",
    "red_on_white_bold": "\x1b[31;1;107m",
    "blue_on_white_bold": "\x1b[34;1;107m",
    "magenta_on_white_bold": "\x1b[35;1;107m",
}

reset_str = "\x1b[0m"

current_color_pool = []


def add_color_to_name(name, color=None):
    if not os.getenv("COBAYA_COLOR"):
        return name
    # TODO: implement for Windows, see
    # https://docs.microsoft.com/en-us/windows/console/console-virtual-terminal-sequences
    if platform.system() == "Windows":
        return name
    if color is None:
        # Choose one at random (ensure no repetition, unless too many requested)
        global current_color_pool
        if not current_color_pool:
            current_color_pool = list(color_strs.values())
            shuffle(current_color_pool)
        i = choice(list(range(len(current_color_pool))))
        color_str = current_color_pool.pop(i)
    else:
        color_str = color_strs.get(color)
    return color_str + name + reset_str


def abstract(method):
    # abstract method decorator for base class HasLogger methods

    # If an @abstract method is called dynamically from another function,
    # you get a logged error that it's not implemented.
    # An @abstract methods also will not be picked up by the dependency analyser, so
    # a class with only an @abstract method implementation of X will not be assigned to
    # provide X. Descendants can of course override @abstract methods to implement them.

    @functools.wraps(method)
    def not_implemented(self, *args, **kwargs):
        if getattr(getattr(self, method.__name__, None), "_is_abstract", None):
            # OK to call if called via super, but not if not over-ridden
            raise LoggedError(
                self.log,
                "%s NotImplemented in %s",
                method.__name__,
                self.__class__.__name__,
            )
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
    elif issubclass(exception_type, mpi.OtherProcessError):
        log.info(str(exception_instance))
        if is_debug(log):
            return  # no traceback printed

    line = "-------------------------------------------------------------\n"
    log.critical(
        line[len(_logger_name) + 5 :]
        + "\n"
        + "".join(
            traceback.format_exception(exception_type, exception_instance, trace_back)
        )
        + line
    )
    if exception_type is KeyboardInterrupt:
        log.critical("Interrupted by the user.")
    elif is_debug(log):
        log.critical(
            "Some unexpected ERROR occurred. "
            "You can see the exception information above.\n"
            "We recommend trying to reproduce this error with '%s:True' in the input.\n"
            "If you cannot solve it yourself and need to report it, "
            "include the debug output,\n"
            "which you can send it to a file setting '%s:[some_file_name]'.",
            "debug",
            "debug",
        )
    # Exit all MPI processes
    if want_abort:
        mpi.abort_if_mpi()


def logger_setup(debug=None):
    """
    Configuring the root logger, for its children to inherit level, format and handlers.

    Level: if debug=True, take DEBUG. If numerical, use ""logging""'s corresponding level.
    If string, set debug level and use it as output file.
    Default: INFO
    """
    debug_file = None
    if debug is True or os.getenv("COBAYA_DEBUG"):
        level = logging.DEBUG
    elif debug in (False, None):
        level = logging.INFO
    elif isinstance(debug, int):
        level = debug
    elif isinstance(debug, str):
        level = logging.DEBUG
        debug_file = debug
    else:
        raise ValueError(
            f"Bad value for debug: {debug}. Set to bool|str(file)|int(level)."
        )
    # Set the default level, to make sure the handlers have a higher one
    logging.root.setLevel(level)
    debug = is_debug(logging.root)

    # Custom formatter
    class MyFormatter(logging.Formatter):
        def format(self, record):
            fmt = (
                (" %(asctime)s " if debug else "")
                + "["
                + ("%d : " % mpi.get_mpi_rank() if mpi.more_than_one_process() else "")
                + "%(name)s"
                + "] "
                + {logging.ERROR: "*ERROR* ", logging.WARNING: "*WARNING* "}.get(
                    record.levelno, ""
                )
                + "%(message)s"
            )
            self._style._fmt = fmt
            return super().format(record)

    # Configure stdout handler
    handle_stdout = logging.StreamHandler(sys.stdout)
    handle_stdout.setLevel(level)
    handle_stdout.setFormatter(MyFormatter())
    # log file? Create and reduce stdout level to INFO
    if debug_file is not None:
        file_stdout = logging.FileHandler(debug_file, mode="w")
        file_stdout.setLevel(level)
        handle_stdout.setLevel(logging.INFO)
        file_stdout.setFormatter(MyFormatter())
        logging.root.addHandler(file_stdout)
    # Add stdout handler only once!
    try:
        stdout_handler = next(
            h for h in logging.root.handlers if getattr(h, "stream", None) == sys.stdout
        )
        # If there is one, update it's logging level and formatter
        stdout_handler.setLevel(handle_stdout.level)
    except StopIteration:
        # If there is none, add it!
        logging.root.addHandler(handle_stdout)
    # Configure the logger to manage exceptions
    sys.excepthook = exception_handler


def get_traceback_text(exec_info):
    return "".join(
        ["-"] * 20
        + ["\n\n"]
        + list(traceback.format_exception(*exec_info))
        + ["\n"]
        + ["-"] * 37
    )


class HasLogger:
    """
    Class having a logger with its name (or an alternative one).

    Has magic methods to ignore the logger at (de)serialization.
    """

    def set_logger(self, lowercase=True, name=None):
        name = name or self.__class__.__name__
        if lowercase:
            name = name.lower()
        self.log = logging.getLogger(add_color_to_name(name))

    # Copying and pickling
    def __getstate__(self):
        """Returns the current state, removing the logger (not picklable)."""
        return {k: v for k, v in self.__dict__.items() if k != "log"}

    def __setstate__(self, d):
        self.__dict__ = d
        self.set_logger()

    def is_debug(self):
        return is_debug(self.log)

    def is_debug_and_mpi_root(self):
        return is_debug(self.log) and mpi.is_main_process()

    @mpi.root_only
    def mpi_warning(self, msg, *args, **kwargs):
        self.log.warning(msg, *args, **kwargs)

    @mpi.root_only
    def mpi_info(self, msg, *args, **kwargs):
        self.log.info(msg, *args, **kwargs)

    @mpi.root_only
    def mpi_debug(self, msg, *args, **kwargs):
        self.log.debug(msg, *args, **kwargs)

    def param_dict_debug(self, msg, dic: dict):
        """Removes numpy2 np.float64 for consistent output"""
        if self.log.getEffectiveLevel() <= logging.DEBUG:
            self.log.debug(
                msg,
                {k: float(v) if isinstance(v, np.number) else v for k, v in dic.items()},
            )
