"""
.. module:: tools

:Synopsis: General tools
:Author: Jesus Torrado

"""

import inspect
import numbers
import os
import platform
import re
import sys
import warnings
from abc import ABC, abstractmethod
from ast import parse
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
from inspect import cleandoc, getfullargspec
from itertools import chain, permutations
from types import ModuleType
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import scipy.stats as stats
from packaging import version

from cobaya.conventions import (
    dump_sort_cosmetic,
    kinds,
    packages_path_arg,
    packages_path_config_file,
    packages_path_env,
    packages_path_input,
    subfolders,
)
from cobaya.log import HasLogger, LoggedError, get_logger
from cobaya.typing import Kind

# Set up logger
log = get_logger(__name__)


def str_to_list(x) -> list:
    """
    Makes sure that the input is a list of strings (could be string).
    """
    return [x] if isinstance(x, str) else x


def ensure_dict(iterable_or_dict):
    """
    For iterables, returns dict with elements as keys and null values.
    """
    if not isinstance(iterable_or_dict, Mapping):
        return dict.fromkeys(iterable_or_dict)
    return iterable_or_dict


def change_key(info, old, new, value):
    """
    Change dictionary key without making new dict or changing order
    :param info: dictionary
    :param old: old key name
    :param new: new key name
    :param value: value for key
    :return: info (same instance)
    """
    k = list(info)
    v = list(info.values())
    info.clear()
    for key, oldv in zip(k, v):
        if key == old:
            info[new] = value
        else:
            info[key] = oldv
    return info


def get_internal_class_component_name(name, kind) -> str:
    """
    Gets qualified name of internal component, relative to the package source,
    of a likelihood, theory or sampler.
    """
    return "." + subfolders[kind] + "." + name


def get_base_classes() -> dict[Kind, Any]:
    """
    Return the base classes for the different kinds.
    """
    from cobaya.likelihood import Likelihood
    from cobaya.sampler import Sampler
    from cobaya.theory import Theory

    return {
        "sampler": Sampler,
        "likelihood": Likelihood,  # type: ignore
        "theory": Theory,
    }


class PythonPath:
    """
    A context that keeps sys.path unchanged, optionally adding path during the context
    at the beginning of the directory search list.
    """

    def __init__(self, path=None, when=True):
        self.path = path if when else None

    def __enter__(self):
        self.old_path = sys.path[:]
        if self.path:
            sys.path.insert(0, os.path.abspath(self.path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.path[:] = self.old_path


@contextmanager
def working_directory(path):
    if path:
        original_cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(original_cwd)
    else:
        yield


def check_module_path(module, path):
    """
    Raises ``ModuleNotFoundError`` is ``module`` was not loaded from the given ``path``.
    """
    module_path = os.path.dirname(os.path.realpath(os.path.abspath(module.__file__)))
    if not module_path.startswith(os.path.realpath(os.path.abspath(path))):
        raise ModuleNotFoundError(
            f"Module {module.__name__} successfully loaded, but not from requested path:"
            f" {path}, but instead from {module_path}"
        )


class VersionCheckError(ValueError):
    """
    Exception to be raised when the installed version of a component (or its requisites)
    is older than a reference one.
    """

    pass  # necessary or it won't print the given error message!


def check_module_version(module: Any, min_version):
    """
    Tries to get the module version and raises :class:`tools.VersionCheckError` if not
    found or older than the specified ``min_version``.
    """
    if not hasattr(module, "__version__") or version.parse(
        module.__version__
    ) < version.parse(min_version):
        raise VersionCheckError(
            "Module %s at %s is version %s but the minimum required version is %s."
            % (
                module.__name__,
                os.path.dirname(module.__file__),
                getattr(module, "__version__", "(non-given)"),
                min_version,
            )
        )


def load_module(
    name, package=None, path=None, min_version=None, check_path=False, reload=False
) -> ModuleType:
    """
    Loads and returns the Python module ``name`` from ``path`` (default: ``None``, meaning
    current working directory) and as part of ``package`` (default: ``None``).

    Because of the way Python looks for modules to import, it is not guaranteed by default
    that the module will be loaded from the given ``path``. This can be enforced with
    ``check_path=True`` (default: ``False``), which will raise ``ModuleNotFoundError`` if
    a module was loaded but not from the given ``path``.

    If some version tag is passed as ``min_version``, it will try to get the module
    version, and may raise :class:`tools.VersionCheckError` if no version tag is found
    or if the found one is older than the specified ``min_version``.

    If ``reload=True`` (default: ``False``), deletes the module from memory previous to
    loading it.

    This is a low-level function. You may want to use instead
    :func:`component.load_external_module`, which interacts with Cobaya's installation and
    logging systems.
    """
    with PythonPath(path):
        # Force reload if requested.
        # Use with care and only in install checks (e.g. for version upgrade checks):
        # will delete all references from previous imports!
        if reload and name in sys.modules:
            del sys.modules[name]
        module = import_module(name, package=package)
    if path and check_path:
        check_module_path(module, path)
    if min_version:
        check_module_version(module, str(min_version))
    return module


def get_compiled_import_path(source_path):
    """
    Returns the folder containing the compiled ``.so`` Python wrapper of a low-level
    language (C, Fortran) code package within the given ``source_path``, e.g.
    ``[source_path]/build/lib.linux-x86_64-3.8``.

    Raises ``FileNotFoundError`` if either the ``build`` or ``lib.[...]`` subfolder does
    not exist, which may indicate a failed compilation of the source package.
    """
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Source path {source_path} not found.")
    build_path = os.path.join(source_path, "build")
    if not os.path.isdir(build_path):
        raise FileNotFoundError(
            f"`build` folder not found for source path {source_path}."
            f" Maybe compilation failed?"
        )
    # Folder starts with `lib.` and ends with either MAJOR.MINOR (standard) or
    # MAJORMINOR (some anaconda versions)
    re_lib = re.compile(f"^lib\\..*{sys.version_info.major}\\.*{sys.version_info.minor}$")
    try:
        post = next(d for d in os.listdir(build_path) if re.fullmatch(re_lib, d))
    except StopIteration:
        raise FileNotFoundError(
            f"No `lib.[...]` folder found containing compiled products at {source_path}. "
            "This may mean that the compilation process failed, of that it was assuming "
            "the wrong python version (current version: "
            f"{sys.version_info.major}.{sys.version_info.minor})"
        )
    return os.path.join(build_path, post)


def import_all_classes(
    path, pkg: str, subclass_of: type, hidden=False, helpers=False, stem: str = None
):
    import pkgutil

    from cobaya.theory import HelperTheory

    result = set()
    if stem:
        stem_root, stem_rest = stem.split(".", 1) if "." in stem else (stem, "")
        stem_module = pkg + "." + stem_root
    else:
        stem_module = ""
        stem_rest = None
    for module_loader, name, is_pkg in pkgutil.iter_modules([path]):
        if hidden or not name.startswith("_"):
            module_name = pkg + "." + name
            if stem and module_name != stem_module:
                continue
            m = load_module(module_name)
            if hidden or not getattr(m, "_is_abstract", False):
                for class_name, cls in inspect.getmembers(m, inspect.isclass):
                    if (
                        issubclass(cls, subclass_of)
                        and (helpers or not issubclass(cls, HelperTheory))
                        and cls.__module__ == module_name
                        and (hidden or not cls.__dict__.get("_is_abstract"))
                    ):
                        result.add(cls)
                if is_pkg:
                    result.update(
                        import_all_classes(
                            os.path.dirname(m.__file__),
                            m.__name__,
                            subclass_of,
                            hidden,
                            helpers,
                            stem_rest,
                        )
                    )
    return result


def get_available_internal_classes(kind, hidden=False, stem=None):
    """
    Gets all class names of a given kind.
    """

    from cobaya.component import CobayaComponent

    path = os.path.join(os.path.dirname(__file__), subfolders[kind])
    return import_all_classes(
        path, "cobaya.%s" % subfolders[kind], CobayaComponent, hidden, stem=stem
    )


def get_all_available_internal_classes(hidden=False, stem=None):
    return set(chain(*(get_available_internal_classes(k, hidden, stem) for k in kinds)))


def get_available_internal_class_names(
    kind=None, hidden=False, stem=None
) -> Iterable[str]:
    return sorted(
        {
            cls.get_qualified_class_name()
            for cls in (
                get_available_internal_classes(kind, hidden, stem)
                if kind
                else get_all_available_internal_classes(hidden, stem)
            )
        }
    )


def replace_optimizations(function_string: str) -> str:
    # make fast version of stats.norm.logpdf for fixed scale and loc
    # can save quite a lot of time evaluating Gaussian priors
    if "stats.norm.logpdf" not in function_string:
        return function_string
    number = r"[+-]?(\d+([.]\d*)?(e[+-]?\d+)?|[.]\d+(e[+-]?\d+)?)"
    regex = (
        r"stats\.norm\.logpdf\((?P<arg>[^,\)]+),"
        r"\s*loc\s*=\s*(?P<loc>%s)\s*,"
        r"\s*scale\s*=\s*(?P<scale>%s)\s*\)" % (number, number)
    )
    p = re.compile(regex)
    match = p.search(function_string)
    if not match:
        return function_string
    span = match.span()
    loc, scale = float(match.group("loc")), float(match.group("scale"))
    replacement = "(-({} {:+.16g})**2/{:.16g} {:+.16g})".format(
        match.group("arg"), -loc, 2 * scale**2, -np.log(2 * np.pi * scale**2) / 2
    )
    return function_string[0 : span[0]] + replacement + function_string[span[1] :]


def get_external_function(string_or_function, name=None):
    """
    Processes an external prior or likelihood, given as a string or a function.

    If the input is a string, it must be evaluable to a function. It can contain import
    statements using :module:`importlib`'s ``import_module``, e.g.
    ``import_module("my_file").my_function``. You can access :module:`scipy.stats` and
    :module:`numpy` members under the handles ``stats`` and ``np`` respectively.

    It is aware of the "value" field for parameters.

    Returns the function.
    """
    if isinstance(string_or_function, Mapping):
        string_or_function = string_or_function.get("value")
    if isinstance(string_or_function, str):
        try:
            scope = globals()
            scope["stats"] = stats
            scope["np"] = np
            string_or_function = replace_optimizations(string_or_function)
            with PythonPath(os.curdir, when="import_module" in string_or_function):
                function = eval(string_or_function, scope)
        except Exception as e:
            raise LoggedError(
                log,
                "Failed to load external function%s: '%r'",
                " '%s'" % name if name else "",
                e,
            )
    else:
        function = string_or_function
    if not callable(function):
        raise LoggedError(
            log,
            "The external function provided "
            + ("for '%s' " % name if name else "")
            + "is not an actual function. Got: '%r'",
            function,
        )
    return function


def recursive_mappings_to_dict(mapping):
    """
    Recursively converts every ``OrderedDict``, `MappingProxyType`` etc. inside the
    argument into ``dict``.
    """
    if isinstance(mapping, Mapping):
        return {k: recursive_mappings_to_dict(v) for k, v in mapping.items()}
    else:
        return mapping


_Dict = TypeVar("_Dict", bound=Mapping)


def recursive_update(base: _Dict | None, update: _Dict, copied=True) -> _Dict:
    """
    Recursive dictionary update, from `this stackoverflow question
    <https://stackoverflow.com/questions/3232943>`_.
    Modified for yaml input, where None and {} are almost equivalent
    """
    updated: dict = (
        deepcopy_where_possible(base)
        if copied and base  # type: ignore
        else base or {}
    )
    for update_key, update_value in (update or {}).items():
        if isinstance(update_value, Mapping):
            updated[update_key] = recursive_update(
                updated.get(update_key, {}), update_value, copied=False
            )
        elif update_value is None:
            if update_key not in updated:
                updated[update_key] = {}
        else:
            updated[update_key] = update_value
    # Trim terminal dicts
    for k, v in (updated or {}).items():
        if isinstance(v, Mapping) and len(v) == 0:
            updated[k] = None
    return updated  # type: ignore


def invert_dict(dict_in: Mapping) -> dict:
    """
    Inverts a dictionary, where values in the returned ones are always lists of the
    original keys. Order is not preserved.
    """
    dict_out: dict = {v: [] for v in dict_in.values()}
    for k, v in dict_in.items():
        dict_out[v].append(k)
    return dict_out


def ensure_latex(string):
    """Inserts $'s at the beginning and end of the string, if necessary."""
    if string.strip()[0] != r"$":
        string = r"$" + string
    if string.strip()[-1] != r"$":
        string += r"$"
    return string


def ensure_nolatex(string):
    """Removes $'s at the beginning and end of the string, if necessary."""
    return string.strip().lstrip("$").rstrip("$")


class NumberWithUnits:
    unit: str | None

    def __init__(self, n_with_unit: Any, unit: str, dtype=float, scale=None):
        """
        Reads number possibly with some `unit`, e.g. 10s, 4d.
        Loaded from a case-insensitive string of a number followed by a unit,
        or just a number in which case the unit is set to None.

        :param n_with_unit: number string or number
        :param unit: unit string
        :param dtype: type for number
        :param scale: multiple to apply for the unit
        """
        self.value: int | float = np.nan

        def cast(x):
            try:
                val = float(x)
                if dtype is int and np.isfinite(val):
                    # in case ints are given in exponential notation, make int(float())
                    if val == 0:
                        return val
                    sign = 1 if val > 0 else -1
                    return sign * int(max(abs(val), 1))
                return val
            except ValueError as excpt:
                raise LoggedError(
                    log, "Could not convert '%r' to a number.", x
                ) from excpt

        if isinstance(n_with_unit, str):
            n_with_unit = n_with_unit.lower()
            unit = unit.lower()
            if n_with_unit.endswith(unit):
                self.unit = unit
                if n_with_unit == unit:
                    self.unit_value = dtype(1)
                else:
                    self.unit_value = cast(n_with_unit[: -len(unit)])
            else:
                raise LoggedError(
                    log, "string '%r' does not have expected unit %s.", n_with_unit, unit
                )
        else:
            self.unit = None
            self.unit_value = cast(n_with_unit)
            self.value = self.unit_value
        self.set_scale(scale if scale is not None else 1)

    def set_scale(self, scale):
        """Applies a numerical value for the scale, updating the attr. `value`."""
        if self.unit:
            self.scale = scale
            self.value = self.unit_value * scale

    def __bool__(self):
        return bool(self.unit_value)


def read_dnumber(n: Any, dim: int):
    """
    Reads number possibly as a multiple of dimension `dim`.
    """
    return NumberWithUnits(n, "d", dtype=int, scale=dim).value


def truncate_to_end_line(file_name):
    with open(file_name, "r+b") as inp:
        # Find the last complete line
        inp.seek(0, 2)  # Go to the end of the file
        pos = inp.tell() - 1
        while pos > 0 and inp.read(1) != b"\n":
            pos -= 1
            inp.seek(pos, 0)
        if pos > 0:
            inp.seek(pos + 1, 0)
            inp.truncate()


def load_DataFrame(file_name, skip=0, root_file_name=None):
    """
    Loads a `pandas.DataFrame` from a text file
    with column names in the first line, preceded by ``#``.

    Can skip any number of first lines, and thin with some factor.
    """
    with open(file_name, encoding="utf-8-sig") as inp:
        top_line = inp.readline().strip()
        if not top_line.startswith("#"):
            # try getdist format chains with .paramnames file
            if root_file_name and os.path.exists(root_file_name + ".paramnames"):
                from getdist import ParamNames

                from cobaya.conventions import OutPar, derived_par_name_separator

                names = ParamNames(root_file_name + ".paramnames").list()
                for i, name in enumerate(names):
                    if name.startswith(OutPar.chi2 + "_") and not name.startswith(
                        OutPar.chi2 + derived_par_name_separator
                    ):
                        names[i] = name.replace(
                            OutPar.chi2 + "_", OutPar.chi2 + derived_par_name_separator
                        )
                cols = ["weight", "minuslogpost"] + names
                inp.seek(0)
            else:
                raise LoggedError(
                    log, "Input sample file does not have header: %s", file_name
                )
        else:
            cols = [a.strip() for a in top_line.lstrip("#").split()]
        if 0 < skip < 1:
            # turn into #lines (need to know total line number)
            n = sum(1 for _ in inp)
            skip = int(round(skip * n)) + 1  # match getdist
            inp.seek(0)
        data = pd.read_csv(
            inp,
            sep=" ",
            header=None,
            names=cols,
            comment="#",
            skipinitialspace=True,
            skiprows=skip,
            index_col=False,
        )

    if not data.empty:
        # Check if the last row contains any NaNs
        if data.iloc[-1].isna().any():
            log.warning("Last row of %s is incomplete or contains NaNs", file_name)
            # If the second-to-last row exists and doesn't contain NaNs,
            # delete the last row assuming this was due to crash on write
            if len(data) > 1 and not data.iloc[-2].isna().any():
                data = data.iloc[:-1]
                log.info(f"Saving {file_name} deleting last (in)complete line")
                truncate_to_end_line(file_name)
    return data


def prepare_comment(comment):
    """Prepares a string (maybe containing multiple lines) to be written as a comment."""
    return (
        "\n".join(["# " + line.lstrip("#") for line in comment.split("\n") if line])
        + "\n"
    )


def is_valid_variable_name(name):
    try:
        parse("%s=None" % name)
        return True
    except SyntaxError:
        return False


def get_scipy_1d_pdf(
    definition: float | Sequence | dict,
) -> stats.distributions.rv_frozen:
    """
    Generates a 1d prior from scipy's pdf's using the given arguments.

    Parameters
    ----------
    definition : float or tuple or dict
        A prior specification, that is, a length-2 tuple specifying a range for a uniform
        prior, or a dictionary that may specify the scipy distribution as ``dist``(default
        if not present: ``uniform``) and the arguments to be passed to that scipy
        distribution. ``loc`` and ``scale`` can alternatively be passed as a ``min`` and
        ``max`` range. A single number for a delta-like prior is also possible.

    Returns
    -------
    stats.rv_frozen
        An initialized scipy.stats distribution instance with the given parameters.

    Raises
    ------
    ValueError
        If the given arguments cannot produce a scipy dist.
    """
    if not definition:
        raise ValueError(
            "Please pass *either* a range [min, max] as arguments, or a dictionary."
        )
    # If list of 2 numbers, it's a uniform prior; if a single number, a delta prior
    if isinstance(definition, numbers.Real):
        kwargs = {"dist": "uniform", "loc": definition, "scale": 0}
    elif (
        isinstance(definition, Sequence)
        and len(definition) == 2
        and all(isinstance(n, numbers.Real) for n in definition)
    ):
        kwargs = {"dist": "uniform", "min": definition[0], "max": definition[1]}
    elif isinstance(definition, dict):
        kwargs = deepcopy(definition)
    else:
        raise ValueError(
            f"Invalid type {type(definition)} for prior definition: {definition}"
        )
    # Get distribution from scipy
    dist = kwargs.pop("dist", "uniform")
    if not isinstance(dist, str):
        raise ValueError(f"If present 'dist' must be a string. Got {type(dist)}.")
    if "min" in kwargs or "max" in kwargs:
        if dist == "truncnorm":
            if "a" in kwargs or "b" in kwargs:
                raise ValueError(
                    "You cannot use the 'a/b' convention and the 'min/max' "
                    "convention at the same time. Either use one or the other."
                )
            loc, scale = kwargs.get("loc", 0), kwargs.get("scale", 1)
            kwargs["a"] = (kwargs.pop("min") - loc) / scale
            kwargs["b"] = (kwargs.pop("max") - loc) / scale
        else:
            # Recover (loc, scale) from (min, max)
            # For coherence with scipy.stats, defaults are (min, max) = (0, 1)
            if "loc" in kwargs or "scale" in kwargs:
                raise ValueError(
                    "You cannot use the 'loc/scale' convention and the 'min/max' "
                    "convention at the same time. Either use one or the other."
                )
            minmaxvalues = {"min": 0.0, "max": 1.0}
            for bound, default in minmaxvalues.items():
                value = kwargs.pop(bound, default)
                try:
                    minmaxvalues[bound] = float(value)
                except (TypeError, ValueError) as excpt:
                    raise ValueError(
                        f"Invalid value {bound}: {value} (must be a number)."
                    ) from excpt
            kwargs["loc"] = minmaxvalues["min"]
            kwargs["scale"] = minmaxvalues["max"] - minmaxvalues["min"]
    if kwargs.get("scale", 1) < 0:
        raise ValueError(
            f"Invalid negative range or scale. Prior definition was {definition}."
        )
    # Check for improper priors
    if not np.all(np.isfinite([kwargs.get("loc", 0), kwargs.get("scale", 1)])):
        raise ValueError("Improper prior: infinite/undefined range or scale.")
    try:
        pdf_dist = getattr(import_module("scipy.stats", dist), dist)
    except AttributeError as attr_excpt:
        raise ValueError(
            f"'{dist}' is not a valid scipy.stats distribution."
        ) from attr_excpt
    # Generate and return the frozen distribution
    try:
        return pdf_dist(**kwargs)
    except TypeError as tp_excpt:
        raise ValueError(
            f"Error when initializing scipy.stats.{dist}: <<{tp_excpt}>>. "
            "This probably means that the distribution {dist} "
            "does not recognize the parameter mentioned in the 'scipy' error above."
        ) from tp_excpt


def _fast_norm_logpdf(norm_dist):
    """WARNING: logpdf(nan) = -inf"""
    scale = norm_dist.kwds["scale"]
    m_log_scale = -np.log(scale) - np.log(2 * np.pi) / 2
    loc = norm_dist.kwds["loc"]

    def fast_logpdf(x):
        return m_log_scale - ((x - loc) / scale) ** 2 / 2

    return fast_logpdf


def _KL_norm(m1, S1, m2, S2):
    """Performs the Gaussian KL computation, without input testing."""
    dim = S1.shape[0]
    S2inv = np.linalg.inv(S2)
    return 0.5 * (
        np.trace(S2inv.dot(S1))
        + (m1 - m2).dot(S2inv).dot(m1 - m2)
        - dim
        + np.linalg.slogdet(S2)[1]
        - np.linalg.slogdet(S1)[1]
    )


def KL_norm(m1=None, S1=(), m2=None, S2=(), symmetric=False):
    """Kullback-Leibler divergence between 2 gaussians."""
    S1, S2 = (np.atleast_2d(S) for S in [S1, S2])
    assert S1.shape[0], "Must give at least S1"
    dim = S1.shape[0]
    if m1 is None:
        m1 = np.zeros(dim)
    if not S2.shape[0]:
        S2 = np.identity(dim)
    if m2 is None:
        m2 = np.zeros(dim)
    if symmetric:
        return _KL_norm(m1, S1, m2, S2) + _KL_norm(m2, S2, m1, S1)
    return _KL_norm(m1, S1, m2, S2)


def choleskyL_corr(M):
    r"""
    Gets the Cholesky lower triangular matrix :math:`L` (defined as :math:`M=LL^T`)
    for the matrix ``M``, in the form :math:`L = S L^\prime` where S is diagonal.

    Can be used to create an affine transformation that decorrelates a sample
    :math:`x=\{x_i\}` with covariance M, as :math:`x=Ly`,
    where :math:`L` is extracted from M and y has identity covariance.

    Returns a tuple of a matrix :math:`S` containing the square roots of the diagonal
    of the input matrix (the standard deviations, if a covariance is given),
    and the scale-free :math:`L^\prime=S^{-1}L`.
    (could just use Cholesky directly for proposal)
    """
    std_diag, corr = cov_to_std_and_corr(M)
    return np.diag(std_diag), np.linalg.cholesky(corr)


def cov_to_std_and_corr(cov):
    """
    Gets the standard deviations (as a 1D array
    and the correlation matrix of a covariance matrix.
    """
    std = np.sqrt(np.diag(cov))
    inv_std = 1 / std
    corr = inv_std[:, np.newaxis] * cov * inv_std[np.newaxis, :]
    np.fill_diagonal(corr, 1.0)
    return std, corr


def are_different_params_lists(list_A, list_B, name_A="A", name_B="B"):
    """
    Compares two parameter lists, and returns a dict with the following keys
    (only present if applicable, and where [A] and [B] are substituted with
    `name_A` and `name_B`):

      `duplicates_[A|B]`: duplicate elements in list A|B
      `[A]_but_not_[B]`: elements from A missing in B
      `[B]_but_not_[A]`: vice versa
    """
    result = {}
    names = {"A": name_A, "B": name_B}
    # Duplicates
    list_A_copy, list_B_copy = list_A[:], list_B[:]
    for n, lst in zip(names, [list_A_copy, list_B_copy]):
        [lst.pop(i) for i in sorted([lst.index(x) for x in set(lst)])[::-1]]
        if lst:
            result["duplicate_%s" % n] = list(set(lst))
    sets = {"A": set(list_A), "B": set(list_B)}
    for n1, n2 in [["A", "B"], ["B", "A"]]:
        missing = sets[n1].difference(sets[n2])
        if missing:
            result[f"{names[n1]}_but_not_{names[n2]}"] = list(missing)
    return result


def create_banner(msg, symbol="*", length=None):
    """
    Puts message into an attention-grabbing banner.

    The banner is delimited by two lines of ``symbol`` (default ``*``)
    of length ``length`` (default: length of message).
    """
    msg_clean = cleandoc(msg)
    if not length:
        length = max(len(line) for line in msg_clean.split("\n"))
    return symbol * length + "\n" + msg_clean + "\n" + symbol * length + "\n"


def warn_deprecation_version(logger=None):
    msg = """
    You are using an archived version of Cobaya, which is no longer maintained.
    Unless intentionally doing so, please, update asap to the latest version
    (e.g. with ``python -m pip install cobaya --upgrade``).
    """
    from cobaya import __obsolete__

    if __obsolete__:
        for line in create_banner(msg).split("\n"):
            getattr(logger, "warning", (lambda x: print("*WARNING*", x)))(line)


def warn_deprecation(logger=None):
    warn_deprecation_version(logger=logger)


def progress_bar(logger, percentage, final_text=""):
    """Very simple, multiline, logger-compatible progress bar, with increments of 5%."""
    progress = int(percentage / 5)
    logger.info(
        " |%s| %3d%% %s", "@" * progress + "-" * (20 - progress), percentage, final_text
    )


def fuzzy_match(input_string, choices, n=3, score_cutoff=50):
    """
    Simple wrapper for fuzzy search of strings within a list.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Suppress message about optional dependency
        from fuzzywuzzy import process as fuzzy_process
    try:
        return list(
            zip(
                *(
                    fuzzy_process.extractBests(
                        input_string, choices, score_cutoff=score_cutoff
                    )
                )
            )
        )[0][:n]
    except IndexError:
        return []


def similar_internal_class_names(name, kind=None):
    """
    Returns a list of suggestions for class names similar to the given one.

    To be used e.g. when no class was found with the given name.

    If a ``kind`` is not given, a dictionary of ``{kind: [list of suggestions]}`` is
    returned instead.
    """
    if kind is None:
        suggestions = {
            kind: fuzzy_match(name, get_available_internal_class_names(kind), n=3)
            for kind in kinds
        }
        # Further trim the set by pooling them all and selecting again.
        all_names = list(chain(*suggestions.values()))
        best_names = fuzzy_match(name, all_names, n=3)
        suggestions = {
            kind: [n for n in names if n in best_names]
            for kind, names in suggestions.items()
        }
        return {kind: sugg for kind, sugg in suggestions.items() if sugg}
    else:
        return fuzzy_match(name, get_available_internal_class_names(kind), n=3)


def has_non_yaml_reproducible(info):
    for value in info.values():
        if (
            callable(value)
            or isinstance(value, Mapping)
            and has_non_yaml_reproducible(value)
        ):
            return True
    return False


_R = TypeVar("_R")


def deepcopy_where_possible(base: _R) -> _R:
    """
    Deepcopies an object whenever possible. If the object cannot be copied, returns a
    reference to the original object (this applies recursively to values of
    a dictionary, and converts all Mapping objects into dict).

    Rationale: cobaya tries to manipulate its input as non-destructively as possible,
    and to do that it works on a copy of it; but some of the values passed to cobaya
    may not be copyable (if they are not pickleable). This function provides a
    compromise solution. To allow dict comparisons and make the copy mutable it converts
    MappingProxyType and other Mapping types into plain dict.
    """
    if isinstance(base, Mapping):
        _copy = {}
        for key, value in base.items():
            _copy[key] = deepcopy_where_possible(value)
        return _copy  # type: ignore
    if isinstance(base, (HasLogger, type)):
        return base  # type: ignore
    else:
        # Special case: instance methods can be copied, but should not be.
        if isinstance(base, Callable) and hasattr(base, "__self__"):
            return base
        try:
            return deepcopy(base)
        except Exception:
            return base


def get_class_methods(cls, not_base=None, start="get_", excludes=(), first="self"):
    methods = {}
    for k, v in inspect.getmembers(cls):
        if (
            k.startswith(start)
            and k not in excludes
            and (not_base is None or not hasattr(not_base, k))
            and getfullargspec(v).args[:1] == [first]
            and not getattr(v, "_is_abstract", None)
        ):
            methods[k[len(start) :]] = v
    return methods


def get_properties(cls):
    return [k for k, v in inspect.getmembers(cls) if isinstance(v, property)]


def sort_parameter_blocks(blocks, speeds, footprints, oversample_power=0.0):
    """
    Find optimal ordering, such that one minimises the time it takes to vary every
    parameter, one by one, in a basis in which they are mixed-down (i.e after a
    Cholesky transformation). To do that, compute that "total cost" for every permutation
    of the blocks order, and find the minimum.

    This algorithm is described in the appendix of the Cobaya paper (TODO: add reference!)

    NB: Rows in ``footprints`` must be in the same order as ``blocks`` and columns in the
    same order as ``speeds``.

    Returns: ``(i_optimal_ordering, cumulative_per_param_costs, oversample_factors)``,
               with costs and oversampling factors following optimal ordering.
    """
    n_params_per_block = np.array([len(b) for b in blocks])
    all_costs = 1 / np.array(speeds)
    all_footprints = np.array(footprints)
    tri_lower = np.tri(len(n_params_per_block))

    def get_cost_per_param_per_block(ordering):
        return np.minimum(1, tri_lower.T.dot(all_footprints[ordering])).dot(all_costs)

    if oversample_power >= 1:
        optimal_ordering, _, _ = sort_parameter_blocks(
            blocks, speeds, footprints, oversample_power=1 - 1e-3
        )
        orderings = [optimal_ordering]
    else:
        orderings = list(permutations(np.arange(len(n_params_per_block))))
    permuted_costs_per_param_per_block = np.array(
        [get_cost_per_param_per_block(list(o)) for o in orderings]
    )
    permuted_oversample_factors = np.array(
        [
            ((this_cost[0] / this_cost) ** oversample_power)
            for this_cost in permuted_costs_per_param_per_block
        ]
    )
    total_costs = np.array(
        [
            (n_params_per_block[list(o)] * permuted_oversample_factors[i]).dot(
                permuted_costs_per_param_per_block[i]
            )
            for i, o in enumerate(orderings)
        ]
    )
    i_optimal: int = np.argmin(total_costs)  # type: ignore
    optimal_ordering = orderings[i_optimal]
    costs = permuted_costs_per_param_per_block[i_optimal]
    oversample_factors = np.floor(permuted_oversample_factors[i_optimal]).astype(int)
    return optimal_ordering, costs, oversample_factors


def find_with_regexp(regexp, root, walk_tree=False):
    """
    Returns all files found which are compatible with the given regexp in directory root,
    including their path in their name.

    If regexp is defined as ``None``, it matches all files inside ``root``.

    Set walk_tree=True if there is more than one directory level (default: `False`).
    """
    if regexp is None:
        regexp = ".*"
    if isinstance(regexp, str):
        regexp = re.compile(regexp)
    try:
        if walk_tree:
            files = []
            for root_folder, sub_folders, file_names in os.walk(root, topdown=True):
                files += [(root_folder, fname) for fname in file_names]
                files += [(root_folder, dname) for dname in sub_folders]
        else:
            files = [(root, f) for f in os.listdir(root)]
    except FileNotFoundError:
        files = []
    return [
        os.path.join(path, f2)
        for path, f2 in files
        if f2 == getattr(regexp.match(f2), "group", lambda: None)()
    ]


def get_translated_params(params_info, params_list):
    """
    Return a dict `{p: p_prime}`, being `p` parameters from `params_info` and `p_prime`
    their equivalent one in `params_list`, taking into account possible renames found
    inside `params_info`.

    The returned dict keeps the order of `params_info`.
    """
    translations = {}
    for p, pinfo in params_info.items():
        renames = {p}.union(str_to_list(pinfo.get("renames", [])))
        try:
            trans = next(r for r in renames if r in params_list)
            translations[p] = trans
        except StopIteration:
            continue
    return translations


def get_cache_path():
    """
    Gets path for cached data, and creates it if it does not exist.

    Defaults to the system's temp folder.
    """
    if platform.system() == "Windows":
        base = os.environ.get("CSIDL_LOCAL_APPDATA", os.environ.get("TMP")) or ""
        cache_path = os.path.join(base, "cobaya/Cache")
    elif platform.system() == "Linux":
        base = os.environ.get(
            "XDG_CACHE_HOME", os.path.join(os.environ["HOME"], ".cache")
        )
        cache_path = os.path.join(base, "cobaya")
    elif platform.system() == "Darwin":
        base = os.path.join(os.environ["HOME"], "Library/Caches")
        cache_path = os.path.join(base, "cobaya")
    else:
        base = os.environ.get("TMP", "")
        cache_path = os.path.join(base, "cobaya")
    try:
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
    except Exception as e:
        raise LoggedError(
            log, "Could not create cache folder %r. Reason: %r", cache_path, str(e)
        )
    return cache_path


def get_config_path():
    """
    Gets path for config files, and creates it if it does not exist.
    """
    config_path = None
    try:
        if platform.system() == "Windows":
            base = os.environ.get("LOCALAPPDATA")
            if not base:
                raise ValueError("Application folder not defined.")
            config_path = os.path.join(base, "cobaya")
        elif platform.system() == "Linux":
            base = os.environ.get(
                "XDG_CONFIG_HOME", os.path.join(os.environ["HOME"], ".config")
            )
            config_path = os.path.join(base, "cobaya")
        elif platform.system() == "Darwin":
            base = os.path.join(os.environ["HOME"], "Library/Application Support")
            config_path = os.path.join(base, "cobaya")
        else:
            raise ValueError("Could not find system type.")
        if not os.path.exists(config_path):
            os.makedirs(config_path)
    except Exception as e:
        raise LoggedError(
            log, "Could not get config folder %r. Reason: %r", config_path, str(e)
        )
    return config_path


def load_config_file():
    """
    Returns the config info, stored in the config file, or an empty dict if not present.
    """
    # Just-in-time import to avoid recursion
    from cobaya.yaml import yaml_load_file

    try:
        return (
            yaml_load_file(os.path.join(get_config_path(), packages_path_config_file))
            or {}
        )
    except Exception:
        return {}


def write_config_file(config_info, append=True):
    """
    Writes the given info into the config file.
    """
    # Just-in-time import to avoid recursion
    from cobaya.yaml import yaml_dump_file

    try:
        info = {}
        if append:
            info.update(load_config_file())
        info.update(config_info)
        yaml_dump_file(
            os.path.join(get_config_path(), packages_path_config_file),
            info,
            error_if_exists=False,
        )
    except Exception as e:
        log.error(
            "Could not write the external packages' installation path into the "
            "config file. Reason: %r",
            str(e),
        )


def load_packages_path_from_config_file():
    """
    Returns the external packages' path stored in the config file,
    or `None` if it can't be found.
    """
    return load_config_file().get(packages_path_input)


def write_packages_path_in_config_file(packages_path):
    """
    Writes the external packages' installation path into the config file.

    Relative paths are converted into absolute ones.
    """
    write_config_file({packages_path_input: os.path.abspath(packages_path)})


def resolve_packages_path(infos=None):
    f"""
    Gets the external packages' installation path given some infos.
    If more than one occurrence of the external packages path in the infos,
    raises an error.

    If there is no external packages' path defined in the given infos,
    defaults to the env variable `{packages_path_env}`, and in its absence to that stored
    in the config file.

    If no path at all could be found, returns `None`.
    """
    if not infos:
        infos = []
    elif isinstance(infos, Mapping):
        infos = [infos]
    paths = {
        os.path.realpath(p)
        for p in [info.get(packages_path_input) for info in infos]
        if p
    }
    if len(paths) == 1:
        return list(paths)[0]
    elif len(paths) > 1:
        raise LoggedError(
            log,
            "More than one packages installation path defined in the given infos. "
            "Cannot resolve a unique one to use. "
            "Maybe specify one via a command line argument '-%s [...]'?",
            packages_path_arg[0],
        )
    return os.environ.get(packages_path_env) or load_packages_path_from_config_file()


def sort_cosmetic(info):
    f"""
    Returns a sorted version of the given info dict, re-ordered as {dump_sort_cosmetic!r},
    and finally the rest of the blocks/options.
    """
    sorted_info = dict()
    for k in dump_sort_cosmetic:
        if k in info:
            sorted_info[k] = info[k]
    sorted_info.update({k: v for k, v in info.items() if k not in sorted_info})
    return sorted_info


def combine_1d(new_list, old_list=None):
    """
    Combines+sorts+uniquifies two lists of values. Sorting is in ascending order.

    If `old_list` given, it is assumed to be a sorted and uniquified array (e.g. the
    output of this function when passed as first argument).

    Uses `np.unique`, which distinguishes numbers up to machine precision.
    """
    new_list = np.atleast_1d(new_list)
    if old_list is not None:
        new_list = np.concatenate((old_list, new_list))
    return np.unique(new_list)


class PoolND(ABC):
    r"""
    Stores a list of ``N``-tuples ``[x_1, x_2...]`` for later retrieval given some
    ``N``-tuple ``x``.

    Tuples are sorted internally, and then by ascending order of their values.

    Tuples are uniquified internally up to machine precision, and an adaptive
    tolerance (relative to min absolute and relative differences in the list) is
    applied at retrieving.

    Adaptive tolerance is defined between limits ``[atol|rtol]_[min|max]``.
    """

    values: np.ndarray

    def __init__(
        self,
        values=(),
        rtol_min=1e-5,
        rtol_max=1e-3,
        atol_min=1e-8,
        atol_max=1e-6,
        logger=None,
    ):
        assert values is not None and len(values) != 0, (
            "Pool needs to be initialised with at least one value."
        )
        assert rtol_min <= rtol_max, (
            f"rtol_min={rtol_min} must be smaller or equal to rtol_max={rtol_max}"
        )
        assert atol_min <= atol_max, (
            f"atol_min={atol_min} must be smaller or equal to ato_max={atol_max}"
        )
        self.atol_min, self.atol_max = atol_min, atol_max
        self.rtol_min, self.rtol_max = rtol_min, rtol_max
        if logger is None:
            self.log = get_logger(self.__class__.__name__)
        else:
            self.log = logger
        self.update(values)

    @property
    def d(self):
        return len(self.values.shape)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, *args, **kwargs):
        return self.values.__getitem__(*args, **kwargs)

    def _update_tolerances(self):
        """Adapts tolerance to the differences in the list."""
        if self.d == 1:
            # Assumes that the pool is sorted!
            values = self.values
        else:
            values = np.copy(self.values)
            values.flatten()
            values = combine_1d(values)
        if len(values) > 1:
            differences = values[1:] - values[:-1]
            min_difference = np.min(differences)
            self._adapt_atol_min = self.atol_min * min_difference
            self._adapt_atol_max = self.atol_max * min_difference
            min_rel_difference = np.min(differences / values[1:])
            self._adapt_rtol_min = self.rtol_min * min_rel_difference
            self._adapt_rtol_max = self.rtol_max * min_rel_difference
        else:  # single-element list
            self._adapt_atol_min = self.atol_min * values[0]
            self._adapt_atol_max = self.atol_max * values[0]
            self._adapt_rtol_min = self.rtol_min
            self._adapt_rtol_max = self.rtol_max

    def update(self, values):
        """Adds a set of values, uniquifies and sorts."""
        values = self._check_values(values)
        self._update_values(values)
        self._update_tolerances()

    @abstractmethod
    def _check_values(self, values):
        """
        Checks that the input values are correctly formatted and re-formats them if
        necessary.

        Returns a correctly formatted array.

        Internal sorting is enforced, but external is ignored.
        """

    @abstractmethod
    def _update_values(self, values):
        """Combines given and existing pool. Should assign ``self.values``."""

    def find_indices(self, values):
        """
        Finds the indices of elements in array ``values`` in the pool.

        For ``dim > 1`` it expects internally ascending-sorted pairs.

        Calls ``numpy.isclose`` for robust comparison, using adaptive ``rtol``, ``atol``
        limits.

        Raises ValueError if not all elements were found, each only once.
        """
        values = self._check_values(values)
        # Fast search first if possible
        indices = self._fast_find_indices(values)
        i_not_found = np.where(indices == -1)[0]
        if len(i_not_found):
            # TODO: since the pool is sorted already, we could take advantage of that,
            #       sort the target values (and save indices to recover original order),
            #       and iterate only over the part of the pool starting where the last
            #       value was found. But since running rapid test first, prob not needed.
            indices_i_prev_not_found = np.concatenate(
                [self._pick_at_most_one(x, None, None) for x in values[i_not_found]]
            )
            if len(indices_i_prev_not_found) < len(i_not_found):
                raise ValueError(
                    f"Could not find some of {list(values)} in pool {list(self.values)}. "
                    "If there appear to be values in the pool close to the requested "
                    "ones, increase max tolerances."
                )
            indices[i_not_found] = indices_i_prev_not_found
        return indices

    def _fast_find_indices(self, values):
        """
        Fast way to find indices, possibly ignoring tolerance, e.g. using np.where(a==b).

        It should check that the right elements have been found, and return an array
        of length ``values.shape[0]`` with ``-1`` for elements that where not found.
        """
        # if no dimensionality-specific implementation: none found
        return np.full(shape=len(values), fill_value=-1)

    def _pick_at_most_one(self, x, pool=None, rtol=None, atol=None):
        """
        Iterates over the pool (full pool if ``pool`` is ``None``) to find the index of a
        single element ``x``, using the provided tolerances.

        It uses the test function ``self._where_isclose(pool, x, rtol, atol)``, returning
        an array of indices of matches.

        Tolerances start at the minimum one, and, until an element is found, are
        progressively increased until the maximum tolerance is reached.
        """
        if pool is None:
            pool = self.values
        # Start with min tolerance for safety
        if rtol is None:
            rtol = self._adapt_rtol_min
        if atol is None:
            atol = self._adapt_atol_min
        i = self._where_isclose(pool, x, rtol=rtol, atol=atol)
        if not len(i):  # none found
            # Increase tolerance (if allowed) until one found
            if rtol > self._adapt_rtol_max and atol > self._adapt_atol_max:
                # Nothing was found despite high tolerance
                return np.empty(shape=0, dtype=int)
            if rtol <= self._adapt_rtol_max:
                rtol *= 10
            if atol <= self._adapt_atol_max:
                atol *= 10
            return self._pick_at_most_one(x, pool, rtol, atol)
        elif len(i) > 1:  # more than one found
            # Decrease tolerance (if allowed!) until only one found
            if rtol < self._adapt_rtol_min and atol < self._adapt_atol_min:
                # No way to find only one element despite low tolerance
                return np.empty(shape=0, dtype=int)
            # Factor not a divisor of the one above, to avoid infinite loops
            if rtol >= self._adapt_rtol_min:
                rtol /= 3
            if atol >= self._adapt_atol_min:
                atol /= 3
            return self._pick_at_most_one(x, pool, rtol, atol)
        else:
            return i

    @abstractmethod
    def _where_isclose(self, pool, x, rtol, atol):
        """
        Returns an array of indices of matches.

        E.g. in 1D it works as ``np.where(np.isclose(pool, x, rtol=rtol, atol=atol))[0]``.
        """


class Pool1D(PoolND):
    r"""
    Stores a list of values ``[x_1, x_2...]`` for later retrieval given some ``x``.

    ``x`` values are uniquified internally up to machine precision, and an adaptive
    tolerance (relative to min absolute and relative differences in the list) is
    applied at retrieving.

    Adaptive tolerance is defined between limits ``[atol|rtol]_[min|max]``.
    """

    def _check_values(self, values):
        return np.atleast_1d(values)

    def _update_values(self, values):
        self.values = combine_1d(values, getattr(self, "values", None))

    def _fast_find_indices(self, values):
        i_insert_left = np.clip(
            np.searchsorted(self.values, values), a_min=None, a_max=len(self) - 1
        )
        return np.where(
            self._cond_isclose(
                self.values[i_insert_left],
                values,
                rtol=self._adapt_rtol_min,
                atol=self._adapt_atol_min,
            ),
            i_insert_left,
            -1,
        )

    def _cond_isclose(self, pool, x, rtol, atol):
        return np.isclose(pool, x, rtol=rtol, atol=atol)

    def _where_isclose(self, pool, x, rtol, atol):
        return np.where(self._cond_isclose(pool, x, rtol=rtol, atol=atol))[0]


def check_2d(pairs, allow_1d=True):
    """
    Checks that the input is a pair (x1, x2) or a list of them.

    Returns a list of pairs as a 2d array with tuples sorted internally.

    Does not sort the pairs with respect to each other or checks for duplicates.

    If `allow_1d=True` (default) a list of more than 2 single values can be passed,
    and will be converted into an internally-sorted list of all possible pairs,
    as a 2d array.

    Raises ``ValueError`` if the argument is badly formatted.
    """
    pairs = np.array(pairs)
    if len(pairs.shape) == 1:
        if len(pairs) < 2:  # Single element or just a number
            raise ValueError(f"Needs at least a pair of values. Got {list(pairs)}.")
        elif len(pairs) == 2:  # Single pair
            pairs = np.atleast_2d(pairs)
        elif len(pairs) > 2:  # list -> generate combinations
            if allow_1d:
                pairs = np.array(
                    list(
                        chain(
                            *[
                                [[x_i, x_j] for x_j in pairs[i + 1 :]]
                                for i, x_i in enumerate(pairs)
                            ]
                        )
                    )
                )
            else:
                raise ValueError(f"Not a (list of) pair(s) of values: {list(pairs)}.")
    elif (len(pairs.shape) == 2 and pairs.shape[1] != 2) or len(pairs.shape) != 2:
        raise ValueError(f"Not a (list of) pair(s) of values: {pairs}.")
    return np.sort(pairs, axis=-1)  # internal sorting


def combine_2d(new_pairs, old_pairs=None):
    """
    Combines+sorts+uniquifies two lists of pairs of values.

    Pairs will be internally sorted in ascending order, and with respect to each other in
    ascending order of the first value.

    `new_pairs` can be a list of more than 2 elements, from which all possible
    internally-sorted combinations will be generated.

    If `old_pairs` given, it is assumed to be a sorted and uniquified array (e.g. the
    output of this function when passed as first argument).

    Raises ``ValueError`` if the first argument is badly formatted (e.g. not a list
    of pairs of values).
    """
    new_pairs = check_2d(new_pairs)
    if old_pairs is not None:
        new_pairs = np.concatenate((old_pairs, new_pairs))
    return np.unique(new_pairs, axis=0)


class Pool2D(PoolND):
    r"""
    Stores a list of pairs ``[(x_1, y_1), (x_2, y_2)...]`` for later retrieval given some
    ``(x, y)``.

    Pairs are uniquified internally up to machine precision, and an adaptive
    tolerance (relative to min absolute and relative differences in the list) is
    applied at retrieving.

    Adaptive tolerance is defined between limits ``[atol|rtol]_[min|max]``.
    """

    def _update_values(self, values):
        self.values = combine_2d(values, getattr(self, "values", None))

    def _check_values(self, values):
        return check_2d(values)

    def _fast_find_indices(self, values):
        # first, locate 1st component
        i_insert_left = np.clip(
            np.searchsorted(self.values[:, 0], values[:, 0]),
            a_min=None,
            a_max=len(self) - 1,
        )
        # we do not need to clip the "right" index, because we will use it as an endpoint
        # for a slice, which is safe
        i_insert_right = np.searchsorted(self.values[:, 0], values[:, 0], side="right")
        slices = np.array([i_insert_left, i_insert_right]).T
        # Now test the resulting slice(s) for a match in the 2nd component
        i_maybe_found = np.clip(
            [
                slices[i][0]
                + np.searchsorted(
                    self.values[slices[i][0] : slices[i][1], 1], values[i][1]
                )
                for i in range(len(values))
            ],
            a_min=None,
            a_max=len(self) - 1,
        )
        return np.where(
            self._cond_isclose(
                self.values[i_maybe_found],
                values,
                rtol=self._adapt_rtol_min,
                atol=self._adapt_atol_min,
            ),
            i_maybe_found,
            -1,
        )

    def _cond_isclose(self, pool, x, rtol, atol):
        return np.all(np.isclose(pool, x, rtol=rtol, atol=atol), axis=-1)

    def _where_isclose(self, pool, x, rtol, atol):
        return np.where(self._cond_isclose(pool, x, rtol=rtol, atol=atol))[0]
