from cobaya.log import HasLogger
from cobaya.input import HasDefaults
import numpy as np
import time


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
                logger.warning("Timing returning zero, may be inaccurate. First call may have done some caching.")
        self.time_avg = (delta_time + self.time_avg * (self.n - 1)) / self.n
        self.time_sqsum += delta_time ** 2
        if self.n > 1:
            self.time_std = np.sqrt((self.time_sqsum - self.n * self.time_avg ** 2) / (self.n - 1))
        if logger:
            logger.debug("Average evaluation time: %g s", self.time_avg)


class CobayaComponent(HasLogger, HasDefaults):

    def __init__(self, info={}, name=None, timing=None, path_install=None):
        self.name = name or self.get_qualified_class_name()
        self.path_install = path_install
        # Load info of the sampler
        for k in info:
            setattr(self, k, info[k])
        self.set_logger(name=self.name)
        if timing:
            self.timer = Timer()
        else:
            self.timer = None

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
                self.name, self.timer.time_avg, self.timer.n))
        self.close()
