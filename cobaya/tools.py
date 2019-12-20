"""
.. module:: tools

:Synopsis: General tools
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

# Global
import os
import sys
from copy import deepcopy
from importlib import import_module
import six
import numpy as np  # don't delete: necessary for get_external_function
import pandas as pd
from collections import OrderedDict as odict
from ast import parse
import warnings
import inspect
from packaging import version

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    # Suppress message about optional dependency
    from fuzzywuzzy import process as fuzzy_process
if not six.PY3:
    ModuleNotFoundError = ImportError
    # noinspection PyUnresolvedReferences,PyDeprecation
    from inspect import cleandoc, getargspec as getfullargspec
    # noinspection PyDeprecation
    from fractions import gcd
    from collections import Mapping
else:
    from inspect import cleandoc, getfullargspec
    from math import gcd
    # noinspection PyCompatibility
    from collections.abc import Mapping

# Local
from cobaya import __obsolete__
from cobaya.conventions import _package, subfolders, partag, kinds
from cobaya.log import LoggedError

# Logger
import logging

log = logging.getLogger(__name__.split(".")[-1])


def str_to_list(x):
    return [x] if isinstance(x, six.string_types) else x


def change_key(info, old, new, value):
    """
    Change dictionary key without making new dict or changing order
    :param info: dictionary
    :param old: old key name
    :param new: new key name
    :param value: value for key
    :return: info (same instance)
    """
    k = list(info.keys())
    v = list(info.values())
    info.clear()
    for key, oldv in zip(k, v):
        if key == old:
            info[new] = value
        else:
            info[key] = oldv
    return info


def get_internal_class_module(name, kind):
    """
    Gets qualified name of internal module, relative to the package source,
    of a likelihood, theory or sampler.
    """
    return '.' + subfolders[kind] + '.' + name


def get_kind(name, fail_if_not_found=True):
    """
    Given a helpfully unique module name, tries to determine it's kind:
    ``sampler``, ``theory`` or ``likelihood``.
    """
    try:
        return next(
            k for k in kinds
            if name in get_available_internal_class_names(k))
    except StopIteration:
        if fail_if_not_found:
            raise LoggedError(log, "Could not determine kind of module %s", name)
        else:
            return None


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


def check_module_path(module, path):
    if not os.path.realpath(os.path.abspath(module.__file__)).startswith(
            os.path.realpath(os.path.abspath(path))):
        raise LoggedError(
            log, "Module %s successfully loaded, but not from requested path: %s.",
            module.__name__, path)


def check_module_version(module, min_version):
    if not hasattr(module, "__version__") or \
            version.parse(module.__version__) < version.parse(min_version):
        raise VersionCheckError(
            "module %s at %s is version %s but required %s or higher" %
            (module.__name__, os.path.dirname(module.__file__),
             getattr(module, "__version__", "(non-given)"), min_version))


def load_module(name, package=None, path=None, min_version=None, check_path=False):
    with PythonPath(path):
        module = import_module(name, package=package)
    if path and check_path:
        check_module_path(module, path)
    if min_version:
        check_module_version(module, min_version)
    return module


def get_class(name, kind=None, None_if_not_found=False, allow_external=True,
              module_path=None, return_module=False):
    """
    Retrieves the requested class from its reference name. The name can be a
    fully-qualified package.module.classname string, or an internal name of the particular
    kind. If the last element of name is not a class, assume class has the same name and
    is in that module.

    By default tries to load internal modules first, then if that fails internal ones.
    module_path can be used to specify a specific external location.

    Raises ``ImportError`` if class not found in the appropriate place in the source tree
    and is not a fully qualified external name.

    If 'kind=None' is not given, tries to guess it if the name is unique (slow!).

    If allow_external=True, allows loading explicit name from anywhere on path.
    """
    if kind is None:
        kind = get_kind(name)
    if '.' in name:
        module_name, class_name = name.rsplit('.', 1)
    else:
        allow_external = False
        module_name = name
        class_name = name

    def get_return(_cls, _mod):
        if return_module:
            return _cls, _mod
        else:
            return _cls

    def return_class(_module_name, package=None):
        _module = load_module(_module_name, package=package, path=module_path)
        if hasattr(_module, class_name):
            cls = getattr(_module, class_name)
        else:
            _module = load_module(_module_name + '.' + class_name,
                                  package=package, path=module_path)
            cls = getattr(_module, class_name)
        if not inspect.isclass(cls):
            return get_return(getattr(cls, class_name), cls)
        else:
            return get_return(cls, _module)

    try:
        if module_path:
            return return_class(module_name)
        else:
            internal_module = get_internal_class_module(module_name, kind)
            return return_class(internal_module, package=_package)
    except:
        exc_info = sys.exc_info()
        if allow_external and not module_path:
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
                return get_return(None, None)
            raise LoggedError(
                log, "%s '%s' not found. Maybe you meant one of the following "
                     "(capitalization is important!): %s",
                kind.capitalize(), name,
                fuzzy_match(name, get_available_internal_class_names(kind), n=3))
        else:
            log.error("There was a problem when importing %s '%s':", kind, name)
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
    if hasattr(string_or_function, "keys"):
        string_or_function = string_or_function.get(partag.value, None)
    if isinstance(string_or_function, six.string_types):
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


def recursive_odict_to_dict(dictio):
    """
    Recursively converts every ``OrderedDict`` inside the argument into ``dict``.
    """
    if hasattr(dictio, "keys"):
        return {k: recursive_odict_to_dict(v) for k, v in dictio.items()}
    else:
        return dictio


def recursive_update(base, update):
    """
    Recursive dictionary update, from `this stackoverflow question
    <https://stackoverflow.com/questions/3232943>`_.
    Modified for yaml input, where None and {} are almost equivalent
    """
    base = base or odict()
    for update_key, update_value in (update or {}).items():
        update_value = update_value if update_value is not None else odict()
        if isinstance(update_value, Mapping):
            base[update_key] = recursive_update(
                base.get(update_key, odict()), update_value)
        else:
            base[update_key] = update_value
    # Trim terminal (o)dicts
    for k, v in (base or {}).items():
        if isinstance(v, Mapping) and len(v) == 0:
            base[k] = None
    return base


def ensure_latex(string):
    """Inserts $'s at the beginning and end of the string, if necessary."""
    if string.strip()[0] != r"$":
        string = r"$" + string
    if string.strip()[-1] != r"$":
        string = string + r"$"
    return string


def ensure_nolatex(string):
    """Removes $'s at the beginning and end of the string, if necessary."""
    return string.strip().lstrip("$").rstrip("$")


def read_dnumber(n, d, dtype=float):
    """Reads number as multiples of a given dimension."""
    try:
        if isinstance(n, six.string_types):
            if n[-1].lower() == "d":
                if n.lower() == "d":
                    return d
                return dtype(n[:-1]) * d
            raise ValueError
    except ValueError:
        raise LoggedError(log, "Could not convert '%r' to a number.", n)
    return n


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
    param = list(info.keys())[0]
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
        if isinstance(info2.get(x), six.string_types):
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


def warn_deprecation_python2(logger=None):
    msg = """
    Python 2 support will eventually be dropped
    (it is already unsupported by many scientific Python modules).

    Please use Python 3!

    In some systems, the Python 3 command may be `python3` instead of `python`.
    If that is the case, use `pip3` instead of `pip` to install Cobaya.
    """
    if not six.PY3:
        for line in create_banner(msg).split("\n"):
            getattr(logger, "warning", (lambda x: print("*WARNING*", x)))(line)


def warn_deprecation_version(logger=None):
    msg = """
    You are using an archived version of Cobaya, which is no longer maintained.
    Unless intentionally doing so, please, update asap to the latest version
    (e.g. with ``pip install cobaya --upgrade``).
    """
    if __obsolete__:
        for line in create_banner(msg).split("\n"):
            getattr(logger, "warning", (lambda x: print("*WARNING*", x)))(line)


def warn_deprecation(logger=None):
    warn_deprecation_python2(logger=logger)
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
    a dictionary).

    Rationale: cobaya tries to manipulate its input as non-destructively as possible,
    and to do that it works on a copy of it; but some of the values passed to cobaya
    may not be copyable (if they are not pickleable). This function provides a
    compromise solution.
    """
    if isinstance(base, Mapping):
        _copy = base.__class__()
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
