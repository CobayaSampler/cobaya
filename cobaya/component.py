import inspect
import os
import sys
import time
from collections.abc import Mapping
from importlib import import_module, resources
from inspect import cleandoc
from typing import get_type_hints

from packaging import version

import cobaya
from cobaya.conventions import cobaya_package, kinds, reserved_attributes
from cobaya.log import HasLogger, LoggedError, get_logger
from cobaya.mpi import is_main_process
from cobaya.tools import (
    VersionCheckError,
    deepcopy_where_possible,
    get_base_classes,
    get_internal_class_component_name,
    load_module,
    resolve_packages_path,
)
from cobaya.typing import Any, InfoDict, InfoDictIn, empty_dict, validate_type
from cobaya.yaml import yaml_dump, yaml_load, yaml_load_file


class Timer:
    def __init__(self):
        self.n = 0
        self.time_sum = 0.0
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


class Description:
    """Allows for calling get_desc as both class and instance method."""

    def __get__(self, instance, cls):
        if instance is None:

            def return_func(info=None):
                return cls._get_desc(info)

        else:

            def return_func(info=None):
                return cls._get_desc(info=instance.__dict__)

        return_func.__doc__ = cleandoc(
            """
            Returns a short description of the class. By default, returns the class'
            docstring.

            You can redefine this method to dynamically generate the description based
            on the class initialisation ``info`` (see e.g. the source code of MCMC's
            *class method* :meth:`~.mcmc._get_desc`)."""
        )
        return return_func


class HasDefaults:
    """
    Base class for components that can read settings from a .yaml file.
    Class methods provide the methods needed to get the defaults information
    and associated data.

    """

    @classmethod
    def get_qualified_names(cls) -> list[str]:
        if cls.__module__ == "__main__":
            return [cls.__name__]
        parts = cls.__module__.split(".")
        if len(parts) > 1:
            # get shortest reference
            try:
                imported = import_module(".".join(parts[:-1]))
            except ImportError:
                pass
            else:
                if getattr(imported, cls.__name__, None) is cls:
                    parts = parts[:-1]
        # allow removing class name that is CamelCase equivalent of module name
        if parts[-1] == cls.__name__ or (
            cls.__name__.lower() == parts[-1][:1] + parts[-1][1:].replace("_", "")
        ):
            return [".".join(parts[i:]) for i in range(len(parts))]
        else:
            return [
                ".".join(parts[i:]) + "." + cls.__name__ for i in range(len(parts) + 1)
            ]

    @classmethod
    def get_qualified_class_name(cls) -> str:
        """
        Get the distinct shortest reference name for the class of the form
        module.ClassName or module.submodule.ClassName etc.
        For Cobaya components the name is relative to subpackage for the relevant kind of
        class (e.g. Likelihood names are relative to cobaya.likelihoods).

        For external classes it loads the shortest fully qualified name of the form
        package.ClassName or package.module.ClassName or
        package.subpackage.module.ClassName, etc.
        """
        qualified_names = cls.get_qualified_names()
        if qualified_names[0].startswith("cobaya."):
            return qualified_names[2]
        else:
            # external
            return qualified_names[0]

    @classmethod
    def get_class_path(cls) -> str:
        """
        Get the file path for the class.
        """
        return os.path.abspath(os.path.dirname(inspect.getfile(cls)))

    @classmethod
    def get_file_base_name(cls) -> str:
        """
        Gets the string used as the name for .yaml, .bib files, typically the
        class name or an un-CamelCased class name
        """
        return cls.__dict__.get("file_base_name") or cls.__name__

    @classmethod
    def get_root_file_name(cls) -> str:
        return os.path.join(cls.get_class_path(), cls.get_file_base_name())

    @classmethod
    def get_yaml_file(cls) -> str | None:
        """
        Gets the file name of the .yaml file for this component if it exists on file
        (otherwise None).
        """
        filename = cls.get_root_file_name() + ".yaml"
        if os.path.exists(filename):
            return filename
        return None

    get_desc = Description()

    @classmethod
    def _get_desc(cls, info=None):
        return cleandoc(cls.__doc__) if cls.__doc__ else ""

    @classmethod
    def get_bibtex(cls) -> str | None:
        """
        Get the content of .bibtex file for this component. If no specific bibtex
        from this class, it will return the result from an inherited class if that
        provides bibtex.
        """
        if filename := cls.__dict__.get("bibtex_file"):
            bib = cls.get_text_file_content(filename)
        else:
            bib = cls.get_associated_file_content(".bibtex")
        if bib:
            return bib
        for base in cls.__bases__:
            if issubclass(base, HasDefaults) and base is not HasDefaults:
                return base.get_bibtex()
        return None

    @classmethod
    def get_associated_file_content(
        cls, ext: str, file_root: str | None = None
    ) -> str | None:
        """
        Return the content of the associated file, if it exists.

        This function handles extracting package files when they may be
        inside a zipped package and thus not directly accessible.

        Returns:
            The content of the file as a string, if it exists and can be read. None otherwise.
        """

        return cls.get_text_file_content((file_root or cls.get_file_base_name()) + ext)

    @classmethod
    def get_text_file_content(cls, file_name: str) -> str | None:
        """
        Return the content of a file in the directory of the module, if it exists.
        """
        package = inspect.getmodule(cls).__package__
        try:
            if os.path.split(str(file_name))[0]:
                raise ValueError(f"{file_name} must be a bare file name, without path.")
            # NB: resources.read_text is considered deprecated from 3.9, and will fail
            if sys.version_info < (3, 9):
                return resources.read_text(package, file_name)
            with (resources.files(package) / file_name).open(
                "r", encoding="utf-8-sig", errors="strict"
            ) as fp:
                text_content = fp.read()
            return text_content
        except Exception:
            return None

    @classmethod
    def get_class_options(cls, input_options=empty_dict) -> InfoDict:
        """
        Returns dictionary of names and values for class variables that can also be
        input and output in yaml files, by default it takes all the
        (non-inherited and non-private) attributes of the class excluding known
        specials.

        Could be overridden using input_options to dynamically generate defaults,
        e.g. a set of input parameters generated depending on the input_options.

        :param input_options: optional dictionary of input parameters
        :return:  dict of names and values
        """
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("_")
            and k not in reserved_attributes
            and not inspect.isroutine(v)
            and not isinstance(v, property)
        }

    @classmethod
    def get_defaults(
        cls, return_yaml=False, yaml_expand_defaults=True, input_options=empty_dict
    ) -> dict | str:
        """
        Return defaults for this component_or_class, with syntax:

        .. code::

           option: value
           [...]

           params:
             [...]  # if required

           prior:
             [...]  # if required

        If keyword `return_yaml` is set to True, it returns literally that,
        whereas if False (default), it returns the corresponding Python dict.

        Note that in external components installed as zip_safe=True packages files cannot
        be accessed directly.
        In this case using !default .yaml includes currently does not work.

        Also note that if you return a dictionary it may be modified (return a deep copy
        if you want to keep it).

        if yaml_expand_defaults then !default: file includes will be expanded

        input_options may be a dictionary of input options, e.g. in case default params
        are dynamically dependent on an input variable
        """
        if "class_options" in cls.__dict__:
            log = get_logger(cls.get_qualified_class_name())
            raise LoggedError(
                log,
                "class_options (in %s) should now be replaced by "
                "public attributes defined directly in the class"
                % cls.get_qualified_class_name(),
            )
        yaml_text = cls.get_associated_file_content(".yaml")
        options = cls.get_class_options(input_options=input_options)
        if options and yaml_text:
            yaml_options = yaml_load(yaml_text)
            if both := set(yaml_options).intersection(options):
                raise LoggedError(
                    get_logger(cls.get_qualified_class_name()),
                    "%s: class has .yaml and class variables/options "
                    "that define the same keys: %s \n"
                    "(type declarations without values are fine "
                    "with yaml file as well).",
                    cls.get_qualified_class_name(),
                    list(both),
                )
            options.update(yaml_options)
            yaml_text = None
        if return_yaml and not yaml_expand_defaults:
            return yaml_text or ""
        this_defaults = (
            yaml_load_file(cls.get_yaml_file(), yaml_text)
            if yaml_text
            else deepcopy_where_possible(options)
        )
        # start with this one to keep the order such that most recent class options
        # near the top. Update below to actually override parameters with these.
        defaults = this_defaults.copy()
        if not return_yaml:
            for base in cls.__bases__:
                if issubclass(base, HasDefaults) and base is not HasDefaults:
                    defaults.update(base.get_defaults(input_options=input_options))
        defaults.update(this_defaults)
        if return_yaml:
            return yaml_dump(defaults)
        else:
            return defaults

    @classmethod
    def get_modified_defaults(cls, defaults, input_options=empty_dict):
        """
        After defaults dictionary is loaded, you can dynamically modify them here
        as needed,e.g. to add or remove defaults['params']. Use this when you don't
        want the inheritance-recursive nature of get_defaults() or don't only
        want to affect class attributes (like get_class_options() does0.
        """
        return defaults

    @classmethod
    def get_annotations(cls) -> InfoDict:
        d = {}
        for base in cls.__bases__:
            if issubclass(base, HasDefaults) and base is not HasDefaults:
                d.update(base.get_annotations())
        d.update(cls.__annotations__)
        return d


class CobayaComponent(HasLogger, HasDefaults):
    """
    Base class for a theory, likelihood or sampler with associated .yaml parameter file
    that can set attributes.
    """

    # The next lists of options apply when comparing existing versus new info at resuming.
    # When defining it for subclasses, redefine append adding this list to new entries.
    _at_resume_prefer_new: list[str] = ["version"]
    _at_resume_prefer_old: list[str] = []

    _enforce_types: bool = False

    def __init__(
        self,
        info: InfoDictIn = empty_dict,
        name: str | None = None,
        timing: bool | None = None,
        packages_path: str | None = None,
        initialize=True,
        standalone=True,
    ):
        if standalone:
            # TODO: would probably be more natural if defaults were always read here
            default_info = self.get_defaults(input_options=info)
            default_info = self.get_modified_defaults(default_info, input_options=info)
            default_info.update(info)
            info = default_info

        self.set_instance_defaults()
        self._name = name or self.get_qualified_class_name()
        self.packages_path = packages_path or resolve_packages_path()
        # set attributes from the info (from yaml file or directly input dictionary)
        annotations = self.get_annotations()
        for k, value in info.items():
            # Preserve dict type for empty dicts (turned into None by recursive_update)
            if value is None and isinstance(getattr(self, k, None), Mapping):
                value = {}
            self.validate_info(k, value, annotations)
            try:
                setattr(self, k, value)
            except AttributeError:
                raise AttributeError(f"Cannot set {k} attribute for {self}!")
        self.set_logger(name=self._name)
        self.validate_attributes(annotations)

        self.set_timing_on(timing)
        try:
            if initialize:
                self.initialize()
        except AttributeError as e:
            if "_params" in str(e):
                raise LoggedError(
                    self.log,
                    "use 'initialize_with_params' if you need to "
                    "initialize after input and output parameters"
                    " are set (%s, %s)",
                    self,
                    e,
                )
            raise

    def set_timing_on(self, on):
        self.timer = Timer() if on else None

    def get_name(self) -> str:
        """
        Get the name. This is usually set by the name used in the input .yaml, but
        otherwise defaults to the fully-qualified class name.

        :return: name string
        """
        return getattr(self, "_name", self.__class__.__name__)

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

    def get_version(self) -> None | str | InfoDict:
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

    def validate_info(self, name: str, value: Any, annotations: dict):
        """
        Does any validation on parameter k read from an input dictionary or yaml file,
        before setting the corresponding class attribute.
        This check is always done, even if _enforce_types is not set.

        :param name: name of parameter
        :param value: value
        :param annotations: resolved inherited dictionary of attributes for this class
        """

        if annotations.get(name) is bool and value and isinstance(value, str):
            raise AttributeError(
                "Class '%s' parameter '%s' should be True "
                "or False, got '%s'" % (self, name, value)
            )

    def validate_attributes(self, annotations: dict):
        """
        If _enforce_types or cobaya.typing.enforce_type_checking is set, this
        checks all class attributes against the annotation types

        :param annotations: resolved inherited dictionary of attributes for this class
        :raises: TypeError if any attribute does not match the annotation type
        """
        check = cobaya.typing.enforce_type_checking
        if check or self._enforce_types and check is not False:
            hints = get_type_hints(self.__class__)  # resolve any deferred attributes
            for name in annotations:
                validate_type(
                    hints[name], getattr(self, name, None), self.get_name() + ":" + name
                )

    @classmethod
    def get_kind(cls):
        """Return, as a string, the kind of this component."""
        return next(
            k for k, k_cls in get_base_classes().items() if issubclass(cls, k_cls)
        )

    @classmethod
    def compare_versions(cls, version_a, version_b, equal=True):
        """
        Checks whether ``version_a`` is equal or higher than ``version_b``.

        For strictly higher, pass ``equal=False`` (default: ``True``).

        :return: bool
        """
        va, vb = version.parse(version_a), version.parse(version_b)
        return va >= vb if equal else va > vb

    def __exit__(self, exception_type, exception_value, traceback):
        if self.timer and self.timer.n:
            self.log.info(
                "Average evaluation time for %s: %g s  (%d evaluations)"
                % (self.get_name(), self.timer.get_time_avg(), self.timer.n_avg())
            )
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
        if timers := [component for component in self.values() if component.timer]:
            sep = "\n   "
            self.log.info(
                "Average computation time:"
                + sep
                + sep.join(
                    [
                        "%s : %g s (%d evaluations, %g s total)"
                        % (
                            component.get_name(),
                            component.timer.get_time_avg(),
                            component.timer.n_avg(),
                            component.timer.time_sum,
                        )
                        for component in timers
                    ]
                )
            )

    def get_versions(self, add_version_field=False) -> InfoDict:
        """
        Get version dictionary

        :return: dictionary of versions for all components
        """

        def format_version(x):
            return {"version": x} if add_version_field else x

        return {
            component.get_name(): format_version(component.get_version())
            for component in self.values()
            if component.has_version()
        }

    def get_speeds(self, ignore_sub=False) -> InfoDict:
        """
        Get speeds dictionary

        :return: dictionary of versions for all components
        """
        from cobaya.theory import HelperTheory

        return {
            component.get_name(): {"speed": component.speed}
            for component in self.values()
            if not (ignore_sub and isinstance(component, HelperTheory))
        }

    # Python magic for the "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for component in self.values():
            component.__exit__(exception_type, exception_value, traceback)


class ComponentNotFoundError(LoggedError):
    """
    Exception to be raised when a component class name could not be identified
    (in order to distinguish that case from any other error occurring at import time).
    """

    pass  # necessary or it won't print the given error message!


def get_component_class(
    name,
    kind=None,
    component_path=None,
    class_name=None,
    allow_external=True,
    allow_internal=True,
    logger=None,
    not_found_level=None,
    min_package_version=None,
):
    """
    Retrieves the requested component class from its reference name. The name can be a
    fully-qualified package.module.classname string, or an internal name of the particular
    kind. If the last element of name is not a class, assume class has the same name and
    is in that module.

    The name can be a fully-qualified package.module.classname string, or an internal name
    of the particular kind. If the last element of name is not a class, assume class has
    the same name and is in that module.

    If ``class_name`` is passed, it will be looked up instead of the first argument.

    If the first argument is a class, it is simply returned.

    By default tries to load internal components first, then if that fails external ones.
    ``component_path`` can be used to specify a specific external location, and in that
    case the class is only looked-for there.

    If the class is not found, it raises :class:`component.ComponentNotFoundError`. If
    this exception will be handled at a higher level, you may pass
    `not_found_level='debug'` to prevent printing non-important messages at error-level
    logging.

    Any other exception means that the component was found but could not be imported.

    If ``allow_external=True`` (default), allows loading explicit name from anywhere on
    path.

    If ``allow_internal=True`` (default), will first try to load internal components. In
    this case, if ``kind=None`` (default), instead of ``theory|likelihood|sampler``, it
    tries to guess it if the name is unique (slow!).

    If allow_external=True, min_package_version can specify a minimum version of the
    external package.
    """
    if not isinstance(name, str):
        return name
    if class_name:
        return get_component_class(
            class_name,
            kind=kind,
            component_path=component_path,
            allow_external=allow_external,
            allow_internal=allow_internal,
            logger=logger,
        )
    if "." in name:
        module_name, class_name = name.rsplit(".", 1)
    else:
        module_name = name
        class_name = None
    assert allow_internal or allow_external
    _not_found_msg = f"'{name}' could not be found."
    if not logger:
        logger = get_logger(__name__)

    def get_matching_class_name(_module: Any, _class_name, none=False):
        cls = getattr(_module, _class_name, None)
        if cls is None and _class_name == _class_name.lower():
            # where the _class_name may be a module name, find CamelCased class
            cls = module_class_for_name(_module, _class_name)
        if cls or none:
            return cls
        else:
            return getattr(_module, _class_name)

    def return_class(_module_name, **kwargs):
        _module: Any = load_module(_module_name, path=component_path, **kwargs)
        if not class_name and hasattr(_module, "get_cobaya_class"):
            return _module.get_cobaya_class()
        _class_name = class_name or module_name
        if not (cls := get_matching_class_name(_module, _class_name, none=True)):
            _module = load_module(
                _module_name + "." + _class_name, path=component_path, **kwargs
            )
            cls = get_matching_class_name(_module, _class_name)
        if not isinstance(cls, type):
            return get_matching_class_name(cls, _class_name)
        else:
            return cls

    def check_kind_and_return(cls):
        """
        If a component kind is specified, checks that the class ``cls`` has the correct
        inheritance, and raises ``TypeError`` if it doesn't.

        Returns the original class if no error occurred.
        """
        if kind is not None:
            if not issubclass(cls, get_base_classes()[kind]):
                raise TypeError(f"Class '{name}' is not a standard class of type {kind}.")
        return cls

    def check_if_ComponentNotFoundError_and_raise(
        _excpt, not_found_msg=_not_found_msg, logger=logger, level="debug"
    ):
        """
        If the exception looks like the target class not being found, turns it into a
        `ComponentNotFoundError`, so that it can be caught appropriately.
        """
        # Could not find this module in particular (ensuring that it does not mean one
        # imported within it).
        is_module_not_found = isinstance(_excpt, ModuleNotFoundError)
        # the module to be imported may not be the last field in the name
        did_not_find_this_module_in_particular = any(
            str(_excpt).rstrip("'").endswith(module) for module in name.split(".")
        )
        if is_module_not_found and did_not_find_this_module_in_particular:
            raise ComponentNotFoundError(logger, not_found_msg, level=level)
        logger.error(f"There was a problem when importing '{name}':")
        raise _excpt

    # Lookup logic:
    # 1. If `component_path` is specified, load the class from there or fail.
    # 2. Otherwise (if allow_internal), look for an internal class with the given name,
    #    looping over possible kinds (theory, likelihood, sampler) if kind not given
    # 3. If internal class-lookup failed (and no `component_path` was provided),
    #    look for an external class in a module *in the current folder* (otherwise handled
    #    already in step 1).
    if component_path:
        try:
            return check_kind_and_return(
                return_class(module_name, min_version=min_package_version)
            )
        except VersionCheckError:
            raise
        except Exception as excpt:
            check_if_ComponentNotFoundError_and_raise(
                excpt,
                not_found_msg=(_not_found_msg + f" Tried loading from {component_path}"),
            )
    if allow_internal:
        # If unknown type, loop recursive call with all possible types
        if kind is None:
            for this_kind in kinds:
                try:
                    return get_component_class(
                        name,
                        this_kind,
                        allow_external=False,
                        allow_internal=True,
                        logger=logger,
                        not_found_level="debug",
                    )
                except ComponentNotFoundError:
                    pass  # do not raise it yet. check all kinds.
            # If we get here, the class was not found for any kind
        else:
            internal_module_name = get_internal_class_component_name(module_name, kind)
            try:
                # No need to check kind
                return return_class(internal_module_name, package=cobaya_package)
            except Exception as excpt:
                try:
                    check_if_ComponentNotFoundError_and_raise(
                        excpt,
                        not_found_msg=_not_found_msg[:-1]
                        + (
                            " as internal, trying external."
                            if allow_external
                            else " as internal component."
                        ),
                    )
                except ComponentNotFoundError:
                    pass  # do not raise it yet. try external (if allowed)
    if allow_external:
        try:
            # Now looking in the current folder only (component_path case handled above)
            return check_kind_and_return(
                return_class(module_name, min_version=min_package_version)
            )
        except VersionCheckError:
            raise
        except Exception as excpt:
            try:
                check_if_ComponentNotFoundError_and_raise(
                    excpt,
                    not_found_msg=(
                        _not_found_msg[:-1]
                        + " as external component in the current path."
                    ),
                )
            except ComponentNotFoundError:
                pass  # do not raise it yet. Give a report below.
    # If we got here, didn't work. Give a report of what has been tried.
    tried = " and ".join(allow_internal * ["internal"] + allow_external * ["external"])
    add_msg = f" Tried loading {tried} classes. No component path was given."
    raise ComponentNotFoundError(logger, _not_found_msg + add_msg, level=not_found_level)


def module_class_for_name(m, name):
    """Get Camel- or uppercase class name matching name in module m."""
    result = None
    valid_names = {name, name[:1] + name[1:].replace("_", "")}
    for cls in classes_in_module(m, subclass_of=CobayaComponent):
        if cls.__name__.lower() in valid_names:
            if result is not None:
                raise ValueError(f"More than one class with same lowercase name {name}")
            result = cls
    return result


def classes_in_module(m, subclass_of=None, allow_imported=False) -> set[type]:
    """
    Returns all classes in a module, optionally imposing that they are a subclass of
    ``subclass_of``, and optionally including imported ones with ``allow_imported=True``
    (default False).
    """
    return {
        cls
        for _, cls in inspect.getmembers(m, inspect.isclass)
        if (not subclass_of or issubclass(cls, subclass_of))
        and (allow_imported or cls.__module__ == m.__name__)
    }


class ComponentNotInstalledError(LoggedError):
    """
    Exception to be raised manually at component initialization or install check if some
    external dependency of the component is missing.
    """

    pass  # necessary or it won't print the given error message!


def _bare_load_external_module(
    name,
    path=None,
    min_version=None,
    reload=False,
    get_import_path=None,
    logger=None,
    not_installed_level=None,
):
    """
    Loads an external module ``name``.

    If a ``path`` is given, it looks for an installation there and fails if it does
    not find one. If ``path`` is not given, tries a global ``import``.

    Raises :class:`component.ComponetNotInstalledError` if the module could not be
    imported.

    If ``min_version`` given, may raise :class:`~tools.VersionCheckError`.

    If ``get_import_path`` (callable, takes ``path``) and ``path`` are given, the function
    is called before attempting to load the module, and is expected to return the
    directory from which the module should be imported (useful e.g. if different from the
    root source directory). It can check e.g. for compilation of non-Python source. It
    should raise ``FileNotFoundError`` with a meaningful error message if the expected
    import directory does not exist.

    If ``reload=True`` (default: ``False``), deletes the module from memory previous
    to loading it.
    """
    if not logger:
        logger = get_logger(__name__)
    import_path = None
    if path:
        try:
            if get_import_path:
                import_path = get_import_path(path)
                if is_main_process():
                    logger.debug(
                        f"'{name}' to be imported from (sub)directory {import_path}"
                    )
            else:
                import_path = path
                if not os.path.exists(import_path):
                    raise FileNotFoundError
        except FileNotFoundError as excpt:
            raise ComponentNotInstalledError(
                logger,
                f"No (compiled) installation of '{name}' at {path}: {excpt}",
                level=not_installed_level,
            )
    try:
        # check_path=True may be redundant with check_external_module above
        return load_module(
            name,
            path=import_path,
            min_version=min_version,
            check_path=bool(path),
            reload=reload,
        )
    except ModuleNotFoundError as excpt:
        path_msg = f"from {path}" if path else "(tried global import)"
        raise ComponentNotInstalledError(
            logger,
            f"Could not import '{name}' {path_msg}: {excpt}",
            level=not_installed_level,
        )


def load_external_module(
    module_name=None,
    path=None,
    install_path=None,
    min_version=None,
    get_import_path=None,
    reload=False,
    logger=None,
    not_installed_level=None,
    default_global=False,
):
    """
    Tries to load an external module at initialisation, dealing with explicit paths
    and Cobaya's installation path.

    If a ``path`` was given, it is enforced (may use ``path="global"`` to force a global
    import).

    If no explicit ``path`` was given, try first from Cobaya's ``install_path``,
    and if it fails try a global import.

    ``install_path`` is the path where Cobaya installed this requisite, up to and
    including the source for the requisite, e.g. ``[...]/cobaya_packages/code/[requisite]`

    If ``get_import_path`` (callable, takes ``path``) and ``path`` are given, the function
    is called before attempting to load the module, and is expected to return the
    directory from which the module should be imported (useful e.g. if different from the
    root source directory). It can check e.g. for compilation of non-Python source. It
    should raise ``FileNotFoundError`` with a meaningful error message if the expected
    import directory does not exist.

    If ``reload=True`` (default: ``False``), deletes the module from memory previous
    to loading it.

    If ``min_version`` given, may raise :class:`~tools.VersionCheckError`.

    May raise :class:`component.ComponentNotInstalledError` if the module was not
    found. If this exception will be handled at a higher level, you may pass
    `not_installed_level='debug'` to prevent printing non-important messages at
    error-level logging.

    If default_global=True, always attempts to load from the global path if not
    installed at path (e.g. pip install).
    """
    if not logger:
        logger = get_logger(__name__)
    load_kwargs = {
        "name": module_name,
        "path": path,
        "get_import_path": get_import_path,
        "min_version": min_version,
        "reload": reload,
        "logger": logger,
    }
    if isinstance(path, str):
        if path.lower() == "global":
            msg_tried = "global import (`path='global'` given)"
            load_kwargs["path"] = None
        else:
            msg_tried = f"import from {path}"
    elif install_path:
        load_kwargs["path"] = install_path
        default_global = True
        msg_tried = (
            "import of Cobaya-installed version, but "
            "defaulting to global import if not found"
        )
    else:
        msg_tried = "global import (no `path` or Cobaya installation path given)"
    try:
        if is_main_process():
            logger.debug(f"Attempting {msg_tried}.")
        module = _bare_load_external_module(not_installed_level="debug", **load_kwargs)
    except ComponentNotInstalledError:
        if default_global:
            if is_main_process():
                logger.debug("Defaulting to global import.")
            load_kwargs["path"] = None
            module = _bare_load_external_module(
                not_installed_level=not_installed_level, **load_kwargs
            )
        else:
            raise
    # Check from where was the module actually loaded
    if is_main_process():
        logger.info(
            f"`{module_name}` module loaded successfully from "
            f"{os.path.dirname(os.path.realpath(os.path.abspath(module.__file__)))}"
        )
    return module
