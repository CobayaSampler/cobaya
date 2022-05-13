"""
.. module:: tools

:Synopsis: General tools
:Author: Jesus Torrado

"""

# Global
import os
import sys
import platform
import warnings
import inspect
import re
import numbers
import pandas as pd
import numpy as np
from itertools import chain
from importlib import import_module
from copy import deepcopy
from packaging import version
from itertools import permutations
from typing import Mapping, Sequence, Any, List, TypeVar, Optional, Union, Iterable, Dict
from types import ModuleType
from inspect import cleandoc, getfullargspec
from ast import parse
from abc import ABC, abstractmethod

# Local
from cobaya.conventions import subfolders, kinds, packages_path_config_file, \
    packages_path_env, packages_path_arg, dump_sort_cosmetic, packages_path_input
from cobaya.log import LoggedError, HasLogger, get_logger
from cobaya.typing import Kind

# Set up logger
log = get_logger(__name__)


def str_to_list(x) -> List:
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
    return '.' + subfolders[kind] + '.' + name


def get_base_classes() -> Dict[Kind, Any]:
    """
    Return the base classes for the different kinds.
    """
    from cobaya.likelihood import Likelihood
    from cobaya.theory import Theory
    from cobaya.sampler import Sampler
    return {"sampler": Sampler, "likelihood": Likelihood,  # type: ignore
            "theory": Theory}


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


def check_module_path(module, path):
    """
    Raises ``ModuleNotFoundError`` is ``module`` was not loaded from the given ``path``.
    """
    module_path = os.path.dirname(os.path.realpath(os.path.abspath(module.__file__)))
    if not module_path.startswith(os.path.realpath(os.path.abspath(path))):
        raise ModuleNotFoundError(
            f"Module {module.__name__} successfully loaded, but not from requested path:"
            f" {path}, but instead from {module_path}")


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
    if not hasattr(module, "__version__") or \
            version.parse(module.__version__) < version.parse(min_version):
        raise VersionCheckError(
            "Module %s at %s is version %s but the minimum required version is %s." %
            (module.__name__, os.path.dirname(module.__file__),
             getattr(module, "__version__", "(non-given)"), min_version))


def load_module(name, package=None, path=None, min_version=None,
                check_path=False, reload=False) -> ModuleType:
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
        if name in sys.modules and reload:
            del sys.modules[name]
        module = import_module(name, package=package)
    if path and check_path:
        check_module_path(module, path)
    if min_version:
        check_module_version(module, min_version)
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
        raise FileNotFoundError(f"`build` folder not found for source path {source_path}."
                                f" Maybe compilation failed?")
    # Folder starts with `lib.` and ends with either MAJOR.MINOR (standard) or
    # MAJORMINOR (some anaconda versions)
    re_lib = re.compile(
        f"^lib\\..*{sys.version_info.major}\\.*{sys.version_info.minor}$")
    try:
        post = next(d for d in os.listdir(build_path) if re.fullmatch(re_lib, d))
    except StopIteration:
        raise FileNotFoundError(
            f"No `lib.[...]` folder found containing compiled products at {source_path}. "
            "This may mean that the compilation process failed, of that it was assuming "
            "the wrong python version (current version: "
            f"{sys.version_info.major}.{sys.version_info.minor})")
    return os.path.join(build_path, post)


def import_all_classes(path, pkg, subclass_of, hidden=False, helpers=False):
    import pkgutil
    from cobaya.theory import HelperTheory
    result = set()
    for (module_loader, name, ispkg) in pkgutil.iter_modules([path]):
        if hidden or not name.startswith('_'):
            module_name = pkg + '.' + name
            m = load_module(module_name)
            if hidden or not getattr(m, '_is_abstract', False):
                for class_name, cls in inspect.getmembers(m, inspect.isclass):
                    if issubclass(cls, subclass_of) and \
                            (helpers or not issubclass(cls, HelperTheory)) and \
                            cls.__module__ == module_name and \
                            (hidden or not cls.__dict__.get('_is_abstract')):
                        result.add(cls)
                if ispkg:
                    result.update(import_all_classes(os.path.dirname(m.__file__),
                                                     m.__name__, subclass_of, hidden))
    return result


def get_available_internal_classes(kind, hidden=False):
    """
    Gets all class names of a given kind.
    """

    from cobaya.component import CobayaComponent
    path = os.path.join(os.path.dirname(__file__), subfolders[kind])
    return import_all_classes(path, 'cobaya.%s' % subfolders[kind], CobayaComponent,
                              hidden)


def get_all_available_internal_classes(hidden=False):
    return set(chain(*(get_available_internal_classes(k, hidden) for k in kinds)))


def get_available_internal_class_names(kind=None, hidden=False) -> Iterable[str]:
    return sorted(set(cls.get_qualified_class_name() for cls in
                      (get_available_internal_classes(kind, hidden) if kind
                       else get_all_available_internal_classes(hidden))))


def replace_optimizations(function_string: str) -> str:
    # make fast version of stats.norm.logpdf for fixed scale and loc
    # can save quite a lot of time evaluating Gaussian priors
    if 'stats.norm.logpdf' not in function_string:
        return function_string
    number = r"[+-]?(\d+([.]\d*)?(e[+-]?\d+)?|[.]\d+(e[+-]?\d+)?)"
    regex = r"stats\.norm\.logpdf\((?P<arg>[^,\)]+)," \
            r"\s*loc\s*=\s*(?P<loc>%s)\s*," \
            r"\s*scale\s*=\s*(?P<scale>%s)\s*\)" % (number, number)
    p = re.compile(regex)
    match = p.search(function_string)
    if not match:
        return function_string
    span = match.span()
    loc, scale = float(match.group("loc")), float(match.group("scale"))
    replacement = "(-(%s %+.16g)**2/%.16g %+.16g)" % (
        match.group("arg"), -loc, 2 * scale ** 2, -np.log(2 * np.pi * scale ** 2) / 2)
    return function_string[0:span[0]] + replacement + function_string[span[1]:]


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
            import scipy.stats as stats  # provide default scope for eval
            scope['stats'] = stats
            scope['np'] = np
            string_or_function = replace_optimizations(string_or_function)
            with PythonPath(os.curdir, when="import_module" in string_or_function):
                function = eval(string_or_function, scope)
        except Exception as e:
            raise LoggedError(
                log, "Failed to load external function%s: '%r'",
                " '%s'" % name if name else "", e)
    else:
        function = string_or_function
    if not callable(function):
        raise LoggedError(
            log, "The external function provided " +
                 ("for '%s' " % name if name else "") +
                 "is not an actual function. Got: '%r'", function)
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


_Dict = TypeVar('_Dict', bound=Mapping)


def recursive_update(base: Optional[_Dict], update: _Dict, copied=True) -> _Dict:
    """
    Recursive dictionary update, from `this stackoverflow question
    <https://stackoverflow.com/questions/3232943>`_.
    Modified for yaml input, where None and {} are almost equivalent
    """
    updated: dict = (deepcopy_where_possible(base) if copied and base  # type: ignore
                     else base or {})
    for update_key, update_value in (update or {}).items():
        if isinstance(update_value, Mapping):
            updated[update_key] = recursive_update(updated.get(update_key, {}),
                                                   update_value, copied=False)
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
    unit: Optional[str]

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
        self.value: Union[int, float] = np.nan

        def cast(x):
            try:
                if dtype == int:
                    # in case ints are given in exponential notation, make int(float())
                    return int(float(x))
                else:
                    return float(x)
            except ValueError:
                raise LoggedError(log, "Could not convert '%r' to a number.", x)

        if isinstance(n_with_unit, str):
            n_with_unit = n_with_unit.lower()
            unit = unit.lower()
            if n_with_unit.endswith(unit):
                self.unit = unit
                if n_with_unit == unit:
                    self.unit_value = dtype(1)
                else:
                    self.unit_value = cast(n_with_unit[:-len(unit)])
            else:
                raise LoggedError(log, "string '%r' does not have expected unit %s.",
                                  n_with_unit, unit)
        else:
            self.unit = None
            self.unit_value = cast(n_with_unit)
            self.value = self.unit_value
        self.set_scale(scale if scale is not None else 1)

    def set_scale(self, scale):
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


def load_DataFrame(file_name, skip=0, root_file_name=None):
    """
    Loads a `pandas.DataFrame` from a text file
    with column names in the first line, preceded by ``#``.

    Can skip any number of first lines, and thin with some factor.
    """
    with open(file_name, "r", encoding="utf-8-sig") as inp:
        top_line = inp.readline().strip()
        if not top_line.startswith('#'):
            # try getdist format chains with .paramnames file
            if root_file_name and os.path.exists(root_file_name + '.paramnames'):
                from getdist import ParamNames
                from cobaya.conventions import OutPar, derived_par_name_separator
                names = ParamNames(root_file_name + '.paramnames').list()
                for i, name in enumerate(names):
                    if name.startswith(OutPar.chi2 + '_') and not name.startswith(
                            OutPar.chi2 + derived_par_name_separator):
                        names[i] = name.replace(OutPar.chi2 + '_',
                                                OutPar.chi2 + derived_par_name_separator)
                cols = ['weight', 'minuslogpost'] + names
                inp.seek(0)
            else:
                raise LoggedError(log, "Input sample file does not have header: %s",
                                  file_name)
        else:
            cols = [a.strip() for a in top_line.lstrip("#").split()]
        if 0 < skip < 1:
            # turn into #lines (need to know total line number)
            n = sum(1 for _ in inp)
            skip = int(round(skip * n)) + 1  # match getdist
            inp.seek(0)
        data = pd.read_csv(
            inp, sep=" ", header=None, names=cols, comment="#", skipinitialspace=True,
            skiprows=skip, index_col=False)

        return data


def prepare_comment(comment):
    """Prepares a string (maybe containing multiple lines) to be written as a comment."""
    return "\n".join(
        ["# " + line.lstrip("#") for line in comment.split("\n") if line]) + "\n"


def is_valid_variable_name(name):
    try:
        parse("%s=None" % name)
        return True
    except SyntaxError:
        return False


def get_scipy_1d_pdf(info):
    """Generates 1d priors from scipy's pdf's from input info."""
    param = list(info)[0]
    info2 = deepcopy(info[param])
    if not info2:
        raise LoggedError(log, "No specific prior info given for "
                               "sampled parameter '%s'." % param)
    # If list of 2 numbers, it's a uniform prior
    elif isinstance(info2, Sequence) and len(info2) == 2 and all(
            isinstance(n, numbers.Real) for n in info2):
        info2 = {"min": info2[0], "max": info2[1]}
    elif not isinstance(info2, Mapping):
        raise LoggedError(log, "Prior format not recognized for %s: %r "
                               "Check documentation for prior specification.",
                          param, info2)
    # What distribution?
    try:
        dist = info2.pop("dist").lower()
    # Not specified: uniform by default
    except KeyError:
        dist = "uniform"
    # Number: uniform with 0 width
    except AttributeError:
        dist = "uniform"
        info2 = {"loc": info2, "scale": 0}
    try:
        pdf_dist = getattr(import_module("scipy.stats", dist), dist)
    except AttributeError:
        raise LoggedError(
            log, "Error creating the prior for parameter '%s': "
                 "The distribution '%s' is unknown to 'scipy.stats'. "
                 "Check the list of allowed possibilities in the docs.", param, dist)
    # Recover loc,scale from min,max
    # For coherence with scipy.stats, defaults are min,max=0,1
    if "min" in info2 or "max" in info2:
        if "loc" in info2 or "scale" in info2:
            raise LoggedError(
                log, "You cannot use the 'loc/scale' convention and the 'min/max' "
                     "convention at the same time. Either use one or the other.")
        minmaxvalues = {"min": 0., "max": 1.}
        for limit in minmaxvalues:
            value = info2.pop(limit, minmaxvalues[limit])
            try:
                minmaxvalues[limit] = float(value)
            except (TypeError, ValueError):
                raise LoggedError(
                    log, "Invalid value '%s: %r' in param '%s' (it must be a number)",
                    limit, value, param)
        if minmaxvalues["max"] < minmaxvalues["min"]:
            raise LoggedError(
                log, "Minimum larger than maximum: '%s, %s' for param '%s'",
                minmaxvalues["min"], minmaxvalues["max"], param)
        info2["loc"] = minmaxvalues["min"]
        info2["scale"] = minmaxvalues["max"] - minmaxvalues["min"]

    for x in ["loc", "scale", "min", "max"]:
        if isinstance(info2.get(x), str):
            raise LoggedError(log, "%s should be a number (got '%s')", x, info2.get(x))
    # Check for improper priors
    if not np.all(np.isfinite([info2.get(x, 0) for x in ["loc", "scale", "min", "max"]])):
        raise LoggedError(log, "Improper prior for parameter '%s'.", param)
    # Generate and return the frozen distribution
    try:
        return pdf_dist(**info2)
    except TypeError as tp:
        raise LoggedError(
            log,
            "'scipy.stats' produced an error: <<%r>>. "
            "This probably means that the distribution '%s' "
            "does not recognize the parameter mentioned in the 'scipy' error above.",
            str(tp), dist)


def _fast_norm_logpdf(self, x):
    """WARNING: logpdf(nan) = -inf"""
    if not hasattr(self, "_cobaya_mlogscale"):
        self._cobaya_mlogscale = -np.log(self.kwds["scale"])
    x_ = (np.array(x) - self.kwds["loc"]) / self.kwds["scale"]
    # noinspection PyProtectedMember
    return self.dist._logpdf(x_) + self._cobaya_mlogscale


def KL_norm(m1=None, S1=np.array([]), m2=None, S2=np.array([])):
    """Kullback-Leibler divergence between 2 gaussians."""
    S1, S2 = [np.atleast_2d(S) for S in [S1, S2]]
    assert S1.shape[0], "Must give at least S1"
    dim = S1.shape[0]
    if m1 is None:
        m1 = np.zeros(dim)
    if not S2.shape[0]:
        S2 = np.identity(dim)
    if m2 is None:
        m2 = np.zeros(dim)
    S2inv = np.linalg.inv(S2)
    KL = 0.5 * (np.trace(S2inv.dot(S1)) + (m1 - m2).dot(S2inv).dot(m1 - m2) -
                dim + np.log(np.linalg.det(S2) / np.linalg.det(S1)))
    return KL


def choleskyL(M, return_scale_free=False):
    r"""
    Gets the Cholesky lower triangular matrix :math:`L` (defined as :math:`M=LL^T`)
    of a given matrix ``M``.

    Can be used to create an affine transformation that decorrelates a sample
    :math:`x=\{x_i\}` as :math:`y=Lx`, where :math:`L` is extracted from the
    covariance of the sample.

    If ``return_scale_free=True`` (default: ``False``), returns a tuple of
    a matrix :math:`S` containing the square roots of the diagonal of the input matrix
    (the standard deviations, if a covariance is given), and the scale-free
    :math:`L^\prime=S^{-1}L`.
    """
    std_diag, corr = cov_to_std_and_corr(M)
    Lprime = np.linalg.cholesky(corr)
    if return_scale_free:
        return std_diag, Lprime
    else:
        return np.linalg.inv(std_diag).dot(Lprime)


def cov_to_std_and_corr(cov):
    """
    Gets the standard deviations (as a diagonal matrix)
    and the correlation matrix of a covariance matrix.
    """
    std_diag = np.diag(np.sqrt(np.diag(cov)))
    invstd_diag = np.linalg.inv(std_diag)
    corr = invstd_diag.dot(cov).dot(invstd_diag)
    return std_diag, corr


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
    for n, l in zip(names, [list_A_copy, list_B_copy]):
        [l.pop(i) for i in sorted([l.index(x) for x in set(l)])[::-1]]
        if l:
            result["duplicate_%s" % n] = list(set(l))
    sets = {"A": set(list_A), "B": set(list_B)}
    for n1, n2 in [["A", "B"], ["B", "A"]]:
        missing = sets[n1].difference(sets[n2])
        if missing:
            result["%s_but_not_%s" % (names[n1], names[n2])] = list(missing)
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
    logger.info(" |%s| %3d%% %s",
                "@" * progress + "-" * (20 - progress), percentage, final_text)


def fuzzy_match(input_string, choices, n=3, score_cutoff=50):
    """
    Simple wrapper for fuzzy search of strings within a list.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Suppress message about optional dependency
        from fuzzywuzzy import process as fuzzy_process
    try:
        return list(zip(*(fuzzy_process.extractBests(
            input_string, choices, score_cutoff=score_cutoff))))[0][:n]
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
            for kind in kinds}
        # Further trim the set by pooling them all and selecting again.
        all_names = list(chain(*suggestions.values()))
        best_names = fuzzy_match(name, all_names, n=3)
        suggestions = {kind: [n for n in names if n in best_names]
                       for kind, names in suggestions.items()}
        return {kind: sugg for kind, sugg in suggestions.items() if sugg}
    else:
        return fuzzy_match(name, get_available_internal_class_names(kind), n=3)


def has_non_yaml_reproducible(info):
    for value in info.values():
        if callable(value) or \
                isinstance(value, Mapping) and has_non_yaml_reproducible(value):
            return True
    return False


_R = TypeVar('_R')


def deepcopy_where_possible(base: _R) -> _R:
    """
    Deepcopies an object whenever possible. If the object cannot be copied, returns a
    reference to the original object (this applies recursively to values of
    a dictionary, and converts all Mapping objects into dict).

    Rationale: cobaya tries to manipulate its input as non-destructively as possible,
    and to do that it works on a copy of it; but some of the values passed to cobaya
    may not be copyable (if they are not pickleable). This function provides a
    compromise solution. To allow dict comparisons and make the copy mutable it converts
    MappingProxyType, OrderedDict and other Mapping types into plain dict.
    """
    if isinstance(base, Mapping):
        _copy = {}
        for key, value in base.items():
            _copy[key] = deepcopy_where_possible(value)
        return _copy  # type: ignore
    if isinstance(base, (HasLogger, type)):
        return base  # type: ignore
    else:
        try:
            return deepcopy(base)
        except:
            return base


def get_class_methods(cls, not_base=None, start='get_', excludes=(), first='self'):
    methods = {}
    for k, v in inspect.getmembers(cls):
        if k.startswith(start) and k not in excludes and \
                (not_base is None or not hasattr(not_base, k)) and \
                getfullargspec(v).args[:1] == [first] and \
                not getattr(v, '_is_abstract', None):
            methods[k[len(start):]] = v
    return methods


def get_properties(cls):
    return [k for k, v in inspect.getmembers(cls) if isinstance(v, property)]


def sort_parameter_blocks(blocks, speeds, footprints, oversample_power=0.):
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
            blocks, speeds, footprints, oversample_power=1 - 1e-3)
        orderings = [optimal_ordering]
    else:
        orderings = list(permutations(np.arange(len(n_params_per_block))))
    permuted_costs_per_param_per_block = np.array(
        [get_cost_per_param_per_block(list(o)) for o in orderings])
    permuted_oversample_factors = np.array(
        [((this_cost[0] / this_cost) ** oversample_power)
         for this_cost in permuted_costs_per_param_per_block])
    total_costs = np.array(
        [(n_params_per_block[list(o)] * permuted_oversample_factors[i])
             .dot(permuted_costs_per_param_per_block[i])
         for i, o in enumerate(orderings)])
    i_optimal: int = np.argmin(total_costs)  # type: ignore
    optimal_ordering = orderings[i_optimal]
    costs = permuted_costs_per_param_per_block[i_optimal]
    oversample_factors = np.floor(permuted_oversample_factors[i_optimal]).astype(int)
    return optimal_ordering, costs, oversample_factors


def find_with_regexp(regexp, root, walk_tree=False):
    """
    Returns all files found which are compatible with the given regexp in directory root,
    including their path in their name.

    Set walk_tree=True if there is more than one directory level (default: `False`).
    """
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
    return [os.path.join(path, f2) for path, f2 in files
            if f2 == getattr(regexp.match(f2), "group", lambda: None)()]


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
        base = os.environ.get("XDG_CACHE_HOME",
                              os.path.join(os.environ["HOME"], ".cache"))
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
            log, "Could not create cache folder %r. Reason: %r", cache_path, str(e))
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
            base = os.environ.get("XDG_CONFIG_HOME",
                                  os.path.join(os.environ["HOME"], ".config"))
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
            log, "Could not get config folder %r. Reason: %r", config_path, str(e))
    return config_path


def load_config_file():
    """
    Returns the config info, stored in the config file, or an empty dict if not present.
    """
    # Just-in-time import to avoid recursion
    from cobaya.yaml import yaml_load_file
    try:
        return yaml_load_file(
            os.path.join(get_config_path(), packages_path_config_file))
    except:
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
        yaml_dump_file(os.path.join(get_config_path(), packages_path_config_file),
                       info, error_if_exists=False)
    except Exception as e:
        log.error("Could not write the external packages' installation path into the "
                  "config file. Reason: %r", str(e))


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
    # noinspection PyStatementEffect
    """
    Gets the external packages' installation path given some infos.
    If more than one occurrence of the external packages path in the infos,
    raises an error.

    If there is no external packages' path defined in the given infos,
    defaults to the env variable `%s`, and in its absence to that stored
    in the config file.

    If no path at all could be found, returns `None`.
    """ % packages_path_env
    if not infos:
        infos = []
    elif isinstance(infos, Mapping):
        infos = [infos]
    # MARKED FOR DEPRECATION IN v3.0
    for info in infos:
        if info.get("modules"):
            raise LoggedError(log, "The input field 'modules' has been deprecated."
                                   "Please use instead %r", packages_path_input)
    # END OF DEPRECATION BLOCK
    paths = set(os.path.realpath(p) for p in
                [info.get(packages_path_input) for info in infos] if p)
    if len(paths) == 1:
        return list(paths)[0]
    elif len(paths) > 1:
        raise LoggedError(
            log, "More than one packages installation path defined in the given infos. "
                 "Cannot resolve a unique one to use. "
                 "Maybe specify one via a command line argument '-%s [...]'?",
            packages_path_arg[0])
    path_env = os.environ.get(packages_path_env)
    # MARKED FOR DEPRECATION IN v3.0
    old_env = "COBAYA_MODULES"
    path_old_env = os.environ.get(old_env)
    if path_old_env and not path_env:
        raise LoggedError(log, "The env var %r has been deprecated in favor of %r",
                          old_env, packages_path_env)
    # END OF DEPRECATION BLOCK
    if path_env:
        return path_env
    return load_packages_path_from_config_file()


def sort_cosmetic(info):
    # noinspection PyStatementEffect
    """
    Returns a sorted version of the given info dict, re-ordered as %r, and finally the
    rest of the blocks/options.
    """ % dump_sort_cosmetic
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

    def __init__(self, values=(),
                 rtol_min=1e-5, rtol_max=1e-3, atol_min=1e-8, atol_max=1e-6, logger=None):
        assert values is not None and len(values) != 0, \
            "Pool needs to be initialised with at least one value."
        assert rtol_min <= rtol_max, \
            f"rtol_min={rtol_min} must be smaller or equal to rtol_max={rtol_max}"
        assert atol_min <= atol_max, \
            f"atol_min={atol_min} must be smaller or equal to ato_max={atol_max}"
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
                [self._pick_at_most_one(x, None, None) for x in values[i_not_found]])
            if len(indices_i_prev_not_found) < len(i_not_found):
                raise ValueError(
                    f"Could not find some of {list(values)} in pool {list(self.values)}. "
                    "If there appear to be a values close to the values in the pool,"
                    " increase max tolerances.")
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
            np.searchsorted(self.values, values), a_min=None, a_max=len(self) - 1)
        return np.where(
            self._cond_isclose(
                self.values[i_insert_left], values, rtol=self._adapt_rtol_min,
                atol=self._adapt_atol_min),
            i_insert_left, -1)

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
                pairs = np.array(list(chain(*[[[x_i, x_j] for x_j in pairs[i + 1:]]
                                              for i, x_i in enumerate(pairs)])))
            else:
                raise ValueError(f"Not a (list of) pair(s) of values: {list(pairs)}.")
    elif (len(pairs.shape) == 2 and pairs.shape[1] != 2) or len(pairs.shape) != 2:
        raise ValueError(f"Not a (list of) pair(s) of values: {list(pairs)}.")
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
        i_insert_left = np.clip(np.searchsorted(self.values[:, 0], values[:, 0]),
                                a_min=None, a_max=len(self) - 1)
        # we do not need to clip the "right" index, because we will use it as an endpoint
        # for a slice, which is safe
        i_insert_right = np.searchsorted(self.values[:, 0], values[:, 0], side="right")
        slices = np.array([i_insert_left, i_insert_right]).T
        i_maybe_found = [
            slices[i][0] + np.searchsorted(
                self.values[slices[i][0]:slices[i][1], 1], values[i][1])
            for i in range(len(values))]
        return np.where(
            self._cond_isclose(
                self.values[i_maybe_found], values, rtol=self._adapt_rtol_min,
                atol=self._adapt_atol_min),
            i_maybe_found, -1)

    def _cond_isclose(self, pool, x, rtol, atol):
        return np.all(np.isclose(pool, x, rtol=rtol, atol=atol), axis=-1)

    def _where_isclose(self, pool, x, rtol, atol):
        return np.where(self._cond_isclose(pool, x, rtol=rtol, atol=atol))[0]
