"""
.. module:: tools

:Synopsis: General tools
:Author: Jesus Torrado

"""

# Global
import os
import sys
import logging
import platform
import warnings
import inspect
import pandas as pd
import numpy as np  # don't delete: necessary for get_external_function
from importlib import import_module
from copy import deepcopy
from packaging import version
from itertools import permutations
from typing import Mapping
from types import ModuleType
from inspect import cleandoc, getfullargspec
from math import gcd
from ast import parse

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    # Suppress message about optional dependency
    from fuzzywuzzy import process as fuzzy_process

# Local
from cobaya import __obsolete__
from cobaya.conventions import _cobaya_package, subfolders, partag, kinds, _packages_path, \
    _packages_path_config_file, _packages_path_env, _packages_path_arg, \
    _dump_sort_cosmetic
from cobaya.log import LoggedError

# Set up logger
log = logging.getLogger(__name__.split(".")[-1])


def str_to_list(x):
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


def get_internal_class_component_name(name, kind):
    """
    Gets qualified name of internal component, relative to the package source,
    of a likelihood, theory or sampler.
    """
    return '.' + subfolders[kind] + '.' + name


def get_base_classes():
    from cobaya.likelihood import Likelihood
    from cobaya.theory import Theory
    from cobaya.sampler import Sampler
    return {kinds.sampler: Sampler, kinds.likelihood: Likelihood, kinds.theory: Theory}


def get_kind(name, allow_external=True):
    """
    Given a helpfully unique component name, tries to determine it's kind:
    ``sampler``, ``theory`` or ``likelihood``.
    """
    try:
        return next(
            k for k in kinds
            if name in get_available_internal_class_names(k))
    except StopIteration:
        if allow_external:
            cls = get_class(name, None_if_not_found=True, allow_internal=False)
            if cls:
                for kind, tp in get_base_classes().items():
                    if issubclass(cls, tp):
                        return kind

        raise LoggedError(log, "Could not find component with name %r", name)


class PythonPath:
    """
    A context that keeps sys.path unchanged, optionally adding path during the context.
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


class VersionCheckError(ValueError):
    pass


def check_component_path(component, path):
    if not os.path.realpath(os.path.abspath(component.__file__)).startswith(
            os.path.realpath(os.path.abspath(path))):
        raise LoggedError(
            log, "Component %s successfully loaded, but not from requested path: %s.",
            component.__name__, path)


def check_component_version(component, min_version):
    if not hasattr(component, "__version__") or \
            version.parse(component.__version__) < version.parse(min_version):
        raise VersionCheckError(
            "component %s at %s is version %s but required %s or higher" %
            (component.__name__, os.path.dirname(component.__file__),
             getattr(component, "__version__", "(non-given)"), min_version))


def load_module(name, package=None, path=None, min_version=None,
                check_path=False) -> ModuleType:
    with PythonPath(path):
        component = import_module(name, package=package)
    if path and check_path:
        check_component_path(component, path)
    if min_version:
        check_component_version(component, min_version)
    return component


def get_class(name, kind=None, None_if_not_found=False, allow_external=True,
              allow_internal=True, component_path=None):
    """
    Retrieves the requested class from its reference name. The name can be a
    fully-qualified package.module.classname string, or an internal name of the particular
    kind. If the last element of name is not a class, assume class has the same name and
    is in that module.

    By default tries to load internal components first, then if that fails external ones.
    component_path can be used to specify a specific external location.

    Raises ``ImportError`` if class not found in the appropriate place in the source tree
    and is not a fully qualified external name.

    If 'kind=None' is not given, tries to guess it if the name is unique (slow!).

    If allow_external=True, allows loading explicit name from anywhere on path.
    If allow_internal=True, will first try to load internal components
    """
    if allow_internal and kind is None:
        kind = get_kind(name)
    if '.' in name:
        module_name, class_name = name.rsplit('.', 1)
    else:
        module_name = name
        class_name = None
    assert allow_internal or allow_external

    def return_class(_module_name, package=None):
        _module = load_module(_module_name, package=package, path=component_path)
        if not class_name and hasattr(_module, "get_cobaya_class"):
            return _module.get_cobaya_class()
        _class_name = class_name or module_name
        if hasattr(_module, _class_name):
            cls = getattr(_module, _class_name)
        else:
            _module = load_module(_module_name + '.' + _class_name,
                                  package=package, path=component_path)
            cls = getattr(_module, _class_name)
        if not inspect.isclass(cls):
            return getattr(cls, _class_name)
        else:
            return cls

    try:
        if component_path:
            return return_class(module_name)
        elif allow_internal:
            internal_module_name = get_internal_class_component_name(module_name, kind)
            return return_class(internal_module_name, package=_cobaya_package)
        else:
            raise Exception()
    except:
        exc_info = sys.exc_info()
        if allow_external and not component_path:
            try:
                import_module(module_name)
            except Exception:
                pass
            else:
                try:
                    return return_class(module_name)
                except:
                    exc_info = sys.exc_info()
        if ((exc_info[0] is ModuleNotFoundError and
             str(exc_info[1]).rstrip("'").endswith(name))):
            if None_if_not_found:
                return None
            if allow_internal:
                raise LoggedError(
                    log, "%s '%s' not found. Maybe you meant one of the following "
                         "(capitalization is important!): %s",
                    kind.capitalize(), name,
                    fuzzy_match(name, get_available_internal_class_names(kind), n=3))
            else:
                raise LoggedError(log, "'%s' not found", name)
        else:
            log.error("There was a problem when importing %s '%s':", kind or "external",
                      name)
            raise exc_info[1]


def import_all_classes(path, pkg, subclass_of, hidden=False, helpers=False):
    import pkgutil
    result = set()
    from cobaya.theory import HelperTheory
    for (module_loader, name, ispkg) in pkgutil.iter_modules([path]):
        if hidden or not name.startswith('_'):
            module_name = pkg + '.' + name
            m = load_module(module_name)
            for class_name, cls in inspect.getmembers(m, inspect.isclass):
                if issubclass(cls, subclass_of) and \
                        (helpers or not issubclass(cls, HelperTheory)) and \
                        cls.__module__ == module_name:
                    result.add(cls)
            if ispkg:
                result.update(import_all_classes(os.path.dirname(m.__file__), m.__name__,
                                                 subclass_of, hidden))
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
    result = set()
    for classes in [get_available_internal_classes(k, hidden) for k in kinds]:
        result.update(classes)
    return result


def get_available_internal_class_names(kind, hidden=False):
    return sorted(set(
        cls.get_qualified_class_name() for cls in
        get_available_internal_classes(kind, hidden)))


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
        string_or_function = string_or_function.get(partag.value, None)
    if isinstance(string_or_function, str):
        try:
            scope = globals()
            import scipy.stats as stats  # provide default scope for eval
            scope['stats'] = stats
            scope['np'] = np
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


def recursive_update(base, update):
    """
    Recursive dictionary update, from `this stackoverflow question
    <https://stackoverflow.com/questions/3232943>`_.
    Modified for yaml input, where None and {} are almost equivalent
    """
    base = base or {}
    for update_key, update_value in (update or {}).items():
        update_value = update_value if update_value is not None else {}
        if isinstance(update_value, Mapping):
            base[update_key] = recursive_update(
                base.get(update_key, {}), update_value)
        else:
            base[update_key] = update_value
    # Trim terminal (o)dicts
    for k, v in (base or {}).items():
        if isinstance(v, Mapping) and len(v) == 0:
            base[k] = None
    return base


def invert_dict(dict_in):
    """
    Inverts a dictionary, where values in the returned ones are always lists of the
    original keys. Order is not preserved.
    """
    dict_out = {v: [] for v in dict_in.values()}
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

    def __init__(self, n_with_unit, unit: str, dtype=float, scale=None):
        """
        Reads number possibly with some `unit`, e.g. 10s, 4d.
        Loaded from a a case-insensitive string of a number followed by a unit,
        or just a number in which case the unit is set to None.

        :param n_with_unit: number string or number
        :param unit: unit string
        :param dtype: type for number
        :param scale: multiple to apply for the unit
        """
        self.value = None

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


def read_dnumber(n, dim):
    """
    Reads number possibly as a multiple of dimension `dim`.
    """
    return NumberWithUnits(n, "d", dtype=int, scale=dim).value


def load_DataFrame(file_name, skip=0, thin=1):
    """
    Loads a `pandas.DataFrame` from a text file
    with column names in the first line, preceded by ``#``.

    Can skip any number of first lines, and thin with some factor.
    """
    with open(file_name, "r") as inp:
        cols = [a.strip() for a in inp.readline().lstrip("#").split()]
        if 0 < skip < 1:
            # turn into #lines (need to know total line number)
            for n, line in enumerate(inp):
                pass
            skip = int(skip * (n + 1))
            inp.seek(0)
        thin = int(thin)
        skiprows = lambda i: i < skip or i % thin
        if thin != 1:
            raise LoggedError(log, "thin is not supported yet")
        # TODO: looks like this thinning is not correctly account for weights???
        return pd.read_csv(
            inp, sep=" ", header=None, names=cols, comment="#", skipinitialspace=True,
            skiprows=skiprows, index_col=False)


def prepare_comment(comment):
    """Prepares a string (maybe containing multiple lines) to be written as a comment."""
    return "\n".join(
        ["# " + line.lstrip("#") for line in comment.split("\n") if line]) + "\n"


# Self describing
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
    # What distribution?
    try:
        dist = info2.pop(partag.dist).lower()
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
        minmaxvalues = {"min": 0, "max": 1}
        for limit in minmaxvalues:
            try:
                value = info2.pop(limit, minmaxvalues[limit])
                minmaxvalues[limit] = np.float(value)
            except (TypeError, ValueError):
                raise LoggedError(
                    log, "Invalid value '%s: %r' in param '%s' (it must be a number)",
                    limit, value, param)
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


def _fast_uniform_logpdf(self, x):
    # not normally used since uniform handled as special case
    """WARNING: logpdf(nan) = -inf"""
    if not hasattr(self, "_cobaya_mlogscale"):
        self._cobaya_mlogscale = -np.log(self.kwds["scale"])
        self._cobaya_max = self.kwds["loc"] + self.kwds["scale"]
        self._cobaya_loc = self.kwds['loc']
    if self._cobaya_loc <= x <= self._cobaya_max:
        return self._cobaya_mlogscale
    else:
        return -np.inf


def _fast_norm_logpdf(self, x):
    """WARNING: logpdf(nan) = -inf"""
    if not hasattr(self, "_cobaya_mlogscale"):
        self._cobaya_mlogscale = -np.log(self.kwds["scale"])
    x_ = (np.array(x) - self.kwds["loc"]) / self.kwds["scale"]
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


def relative_to_int(numbers, precision=1 / 10):
    """
    Turns relative numbers (e.g. relative speeds) into integer,
    up to some given `precision` on differences.
    """
    numbers = np.array(np.round(np.array(numbers) / min(numbers) / precision), dtype=int)
    return np.array(
        numbers / np.ufunc.reduce(np.frompyfunc(gcd, 2, 1), numbers), dtype=int)


def create_banner(msg, symbol="*", length=None):
    """
    Puts message into an attention-grabbing banner.

    The banner is delimited by two lines of ``symbol`` (default ``*``)
    of length ``length`` (default: length of message).
    """
    msg_clean = cleandoc(msg)
    if not length:
        length = max([len(line) for line in msg_clean.split("\n")])
    return symbol * length + "\n" + msg_clean + "\n" + symbol * length + "\n"


def warn_deprecation_version(logger=None):
    msg = """
    You are using an archived version of Cobaya, which is no longer maintained.
    Unless intentionally doing so, please, update asap to the latest version
    (e.g. with ``python -m pip install cobaya --upgrade``).
    """
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
    try:
        return list(zip(*(fuzzy_process.extractBests(
            input_string, choices, score_cutoff=score_cutoff))))[0][:n]
    except IndexError:
        return []


def deepcopy_where_possible(base):
    """
    Deepcopies an object whenever possible. If the object cannot be copied, returns a
    reference to the original object (this applies recursively to keys and values of
    a dictionary, and converts all Mapping objects into dict).

    Rationale: cobaya tries to manipulate its input as non-destructively as possible,
    and to do that it works on a copy of it; but some of the values passed to cobaya
    may not be copyable (if they are not pickleable). This function provides a
    compromise solution. To allow dict comparisons and make the copy mutable it converts
    MappingProxyType, OrderedDict and other Mapping types into plain dict.
    """
    if isinstance(base, Mapping):
        _copy = {}
        for key, value in (base or {}).items():
            key_copy = deepcopy(key)
            _copy[key_copy] = deepcopy_where_possible(value)
        return _copy
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
                getfullargspec(v).args[:1] == [first]:
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
    i_optimal = np.argmin(total_costs)
    optimal_ordering = orderings[i_optimal]
    costs = permuted_costs_per_param_per_block[i_optimal]
    oversample_factors = np.floor(permuted_oversample_factors[i_optimal]).astype(int)
    return optimal_ordering, costs, oversample_factors


def find_with_regexp(regexp, root, walk_tree=False):
    """
    Returns all files found which are compatible with the given regexp in directory root,
    including their path in their name.

    Set `walk_tree=True` if there is more that one directory level (default: `False`).
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
        renames = {p}.union(set(str_to_list(pinfo.get(partag.renames, []))))
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
        base = os.environ.get("CSIDL_LOCAL_APPDATA", os.environ.get("TMP"))
        cache_path = os.path.join(base, "cobaya/Cache")
    elif platform.system() == "Linux":
        base = os.environ.get("XDG_CACHE_HOME",
                              os.path.join(os.environ["HOME"], ".cache"))
        cache_path = os.path.join(base, "cobaya")
    elif platform.system() == "Darwin":
        base = os.path.join(os.environ["HOME"], "Library/Caches")
        cache_path = os.path.join(base, "cobaya")
    else:
        base = os.environ.get("TMP")
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
            os.path.join(get_config_path(), _packages_path_config_file))
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
        yaml_dump_file(os.path.join(get_config_path(), _packages_path_config_file),
                       info, error_if_exists=False)
    except Exception as e:
        log.error("Could not write the external packages installation path into the "
                  "config file. Reason: %r", str(e))


def load_packages_path_from_config_file():
    """
    Returns the external packages path stored in the config file,
    or `None` if it can't be found.
    """
    return load_config_file().get(_packages_path)


def write_packages_path_in_config_file(packages_path):
    """
    Writes the external packages installation path into the config file.

    Relative paths are converted into absolute ones.
    """
    write_config_file({_packages_path: os.path.abspath(packages_path)})


def resolve_packages_path(infos=None):
    """
    Gets the external packages installation path given some infos.
    If more than one occurrence of the external packages path in the infos,
    raises an error.

    If there is no external packages path defined in the given infos,
    defaults to the env variable `%s`, and in its absence to that stored
    in the config file.

    If no path at all could be found, returns `None`.
    """ % _packages_path_env
    if not infos:
        infos = []
    elif isinstance(infos, Mapping):
        infos = [infos]
    # MARKED FOR DEPRECATION IN v3.0
    # BEHAVIOUR TO BE REPLACED BY ERROR:
    [check_deprecated_modules_path(info) for info in infos]
    # END OF DEPRECATION BLOCK
    paths = set(p for p in [info.get(_packages_path) for info in infos] if p)
    if len(paths) == 1:
        return list(paths)[0]
    elif len(paths) > 1:
        raise LoggedError(
            log, "More than one packages installation path defined in the given infos. "
                 "Cannot resolve a unique one to use. "
                 "Maybe specify one via a command line argument '-%s [...]'?",
            _packages_path_arg[0])
    path_env = os.environ.get(_packages_path_env)
    # MARKED FOR DEPRECATION IN v3.0
    old_env = "COBAYA_MODULES"
    path_old_env = os.environ.get(old_env)
    if path_old_env and not path_env:
        log.warning("*DEPRECATION*: The env var %r will be deprecated in favor of %r in "
                    "the next version. Please, use that one instead.",
                    old_env, _packages_path_env)
        # BEHAVIOUR TO BE REPLACED BY ERROR:
        path_env = path_old_env
    # END OF DEPRECATION BLOCK -- CONTINUES BELOW!
    if path_env:
        return path_env
    return load_packages_path_from_config_file()


def sort_cosmetic(info):
    """
    Returns a sorted version of the given info dict, re-ordered as %r, and finally the
    rest of the blocks/options.
    """ % _dump_sort_cosmetic
    sorted_info = dict()
    for k in _dump_sort_cosmetic:
        if k in info:
            sorted_info[k] = info[k]
    sorted_info.update({k: v for k, v in info.items() if k not in sorted_info})
    return sorted_info


# MARKED FOR DEPRECATION IN v3.0
def check_deprecated_modules_path(info):
    if info.get("modules"):
        log.warning("*DEPRECATION*: The input field 'modules' will be deprecated in "
                    "favor of %r in the next version. Please, use that one instead.",
                    _packages_path)
        # BEHAVIOUR TO BE REPLACED BY ERROR:
        if not info.get(_packages_path):
            info[_packages_path] = info["modules"]
# END OF DEPRECATION BLOCK
