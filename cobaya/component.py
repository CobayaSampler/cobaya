import time
from packaging import version
from typing import Optional, Union, List

from cobaya.log import HasLogger, LoggedError
from cobaya.input import HasDefaults
from cobaya.typing import InfoDict, InfoDictIn, empty_dict
from cobaya.tools import resolve_packages_path
from cobaya.conventions import packages_path_input


class Timer:
    def __init__(self):
        self.n = 0
        self.time_sum = 0.
        self._start = None
        self._time_func = getattr(time, "perf_counter", time.time)
        self._first_time = None

    def start(self):
        self._start = self._time_func()

    def time_from_start(self):
        return self._time_func() - self._start

    def n_avg(self):
        return self.n - 1 if self.n > 1 else self.n

    def get_time_avg(self):
        if self.n > 1:
            return self.time_sum / (self.n - 1)
        else:
            return self._first_time

    def increment(self, logger=None):
        delta_time = self._time_func() - self._start
        if self._first_time is None:
            if not delta_time:
                logger.warning("Timing returning zero, may be inaccurate")
            # first may differ due to caching, discard
            self._first_time = delta_time
            self.n = 1
            if logger:
                logger.debug("First evaluation time: %g s", delta_time)

        else:
            self.n += 1
            self.time_sum += delta_time
        if logger:
            logger.debug("Average evaluation time: %g s", self.get_time_avg())


class CobayaComponent(HasLogger, HasDefaults):
    """
    Base class for a theory, likelihood or sampler with associated .yaml parameter file
    that can set attributes.
    """
    # The next lists of options apply when comparing existing versus new info at resuming.
    # When defining it for subclasses, redefine append adding this list to new entries.
    _at_resume_prefer_new: List[str] = ["version"]
    _at_resume_prefer_old: List[str] = []

    def __init__(self, info: InfoDictIn = empty_dict,
                 name: Optional[str] = None,
                 timing: Optional[bool] = None,
                 packages_path: Optional[str] = None,
                 initialize=True, standalone=True):
        if standalone:
            # TODO: would probably be more natural if defaults were always read here
            default_info = self.get_defaults(input_options=info)
            default_info.update(info)
            info = default_info

        self.set_instance_defaults()
        self._name = name or self.get_qualified_class_name()
        self.packages_path = packages_path or resolve_packages_path()
        # set attributes from the info (from yaml file or directly input dictionary)
        for k, value in info.items():
            try:
                # MARKED FOR DEPRECATION IN v3.0
                # NB: cannot ever raise an error, since users may use "path_install" for
                #     their own purposes. When considered *fully* deprecated, simply
                #     remove this whole block.
                if k == "path_install":
                    self.log.warning(
                        "*DEPRECATION*: `path_install` will be deprecated "
                        "in the next version. Please use `packages_path` instead.")
                    setattr(self, packages_path_input, value)
                # END OF DEPRECATION BLOCK
                setattr(self, k, value)
            except AttributeError:
                raise AttributeError("Cannot set {} attribute for {}!".format(k, self))
        self.set_logger(name=self._name)
        self.set_timing_on(timing)
        try:
            if initialize:
                self.initialize()
        except AttributeError as e:
            if '_params' in str(e):
                raise LoggedError(self.log, "use 'initialize_with_params' if you need to "
                                            "initialize after input and output parameters"
                                            " are set (%s, %s)", self, e)
            raise

    def set_timing_on(self, on):
        self.timer = Timer() if on else None

    def get_name(self) -> str:
        """
        Get the name. This is usually set by the name used in the input .yaml, but
        otherwise defaults to the fully-qualified class name.

        :return: name string
        """
        return self._name

    def __repr__(self):
        return self.get_name()

    def close(self, *args):
        """Finalizes the class, if something needs to be cleaned up."""
        pass

    def set_instance_defaults(self):
        """
        Can use this to define any default instance attributes before setting to the
        input dictionary (from inputs or defaults)
        """
        pass

    def initialize(self):
        """
        Initializes the class (called from __init__, before other initializations).
        """
        pass

    def get_version(self) -> Union[None, str, InfoDict]:
        """
        Get version information for this component.

        :return: string or dict of values or None
        """
        return None

    def has_version(self):
        """
        Whether to track version information for this component
        """
        return True

    @classmethod
    def compare_versions(cls, version_a, version_b, equal=True):
        """
        Checks whether ``version_a`` is equal or higher than ``version_b``.

        For strictly higher, pass ``equal=False`` (default: ``True``).

        :return: bool
        """
        va, vb = version.parse(version_a), version.parse(version_b)
        if va >= vb if equal else va > vb:
            return True
        return False

    def __exit__(self, exception_type, exception_value, traceback):
        if self.timer and self.timer.n:
            self.log.info("Average evaluation time for %s: %g s  (%d evaluations)" % (
                self.get_name(), self.timer.get_time_avg(), self.timer.n_avg()))
        self.close()


class ComponentCollection(dict, HasLogger):
    """
    Base class for a dictionary of components (e.g. likelihoods or theories)
    """

    def get_helper_theory_collection(self):
        return self

    def add_instance(self, name, component):
        helpers = component.get_helper_theories()
        component.update_for_helper_theories(helpers)
        self.get_helper_theory_collection().update(helpers)
        self[name] = component

    def dump_timing(self):
        timers = [component for component in self.values() if component.timer]
        if timers:
            sep = "\n   "
            self.log.info(
                "Average computation time:" + sep + sep.join(
                    ["%s : %g s (%d evaluations, %g s total)" % (
                        component.get_name(), component.timer.get_time_avg(),
                        component.timer.n_avg(), component.timer.time_sum)
                     for component in timers]))

    def get_versions(self, add_version_field=False) -> InfoDict:
        """
        Get version dictionary
        :return: dictionary of versions for all components
        """

        def format_version(x):
            return {"version": x} if add_version_field else x

        return {component.get_name(): format_version(component.get_version())
                for component in self.values() if component.has_version()}

    def get_speeds(self, ignore_sub=False) -> InfoDict:
        """
        Get speeds dictionary
        :return: dictionary of versions for all components
        """
        from cobaya.theory import HelperTheory
        return {component.get_name(): {"speed": component.speed}
                for component in self.values() if
                not (ignore_sub and isinstance(component, HelperTheory))}

    # Python magic for the "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for component in self.values():
            component.__exit__(exception_type, exception_value, traceback)
