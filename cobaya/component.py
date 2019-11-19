from cobaya.log import HasLogger
from cobaya.input import HasDefaults
import numpy as np
import time
from collections import OrderedDict


class Timer(object):
    def __init__(self):
        self.n = 0
        self.time_avg = 0
        self.time_sqsum = 0
        self.time_std = np.inf  # not actually used?
        self._start = None
        self._time_func = getattr(time, "perf_counter", time.time)

    def start(self):
        self._start = self._time_func()

    def increment(self, logger=None):
        delta_time = self._time_func() - self._start
        self.n += 1
        # TODO: Protect against caching in first call by discarding first time
        # if too different from second one (maybe check difference in log10 > 1)
        # In that case, take into account that #timed_evals is now (self.n - 1)
        if logger and self.n == 2:
            if delta_time:
                log10diff = np.log10(self.time_avg / delta_time)
                if log10diff > 1:
                    logger.warning(
                        "It seems the first call has done some caching (difference "
                        " of a factor %g). Average timing will not be reliable "
                        "unless many evaluations are carried out.", 10 ** log10diff)
            else:
                logger.warning("Timing returning zero, may be inaccurate. First call may "
                               "have done some caching.")
        self.time_avg = (delta_time + self.time_avg * (self.n - 1)) / self.n
        self.time_sqsum += delta_time ** 2
        if self.n > 1:
            self.time_std = np.sqrt(
                (self.time_sqsum - self.n * self.time_avg ** 2) / (self.n - 1))
        if logger:
            logger.debug("Average evaluation time: %g s", self.time_avg)


class CobayaComponent(HasLogger, HasDefaults):
    """
    Base class for a theory, likelihood or sampler with associated .yaml parameter
    that can set attributes.
    """

    class_options = {}

    def __init__(self, info={}, name=None, timing=None, path_install=None):
        self._name = name or self.get_qualified_class_name()
        self.path_install = path_install
        for k, value in self.class_options.items():
            setattr(self, k, value)
        # set attributes from the info (usually from yaml file)
        for k, value in info.items():
            setattr(self, k, value)
        self.set_logger(name=self._name)
        if timing:
            self.timer = Timer()
        else:
            self.timer = None

    def get_name(self):
        return self._name

    def __repr__(self):
        return self.get_name()

    # Optional
    def close(self):
        """Finalizes the class, if something needs to be cleaned up."""
        pass

    # Optional
    def initialize(self):
        """
        Initializes the class.
        """
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        if self.timer:
            self.log.info("Average evaluation time for %s: %g s  (%d evaluations)" % (
                self.get_name(), self.timer.time_avg, self.timer.n))
        self.close()


class ComponentCollection(OrderedDict, HasLogger):
    """
    Base class for an ordered dictionary of components (e.g. likelihoods or theories)
    """

    def __init__(self):
        super(ComponentCollection, self).__init__()

    def dump_timing(self):
        timers = [component for component in self.values() if component.timer]
        if timers:
            sep = "\n   "
            self.log.info(
                "Average computation time:" + sep + sep.join(
                    ["%s : %g s (%d evaluations)" % (
                        component.get_name(), component.timer.time_avg, component.timer.n)
                     for component in timers]))

    # Python magic for the "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for component in self.values():
            component.__exit__(exception_type, exception_value, traceback)
