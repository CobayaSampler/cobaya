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
import scipy.stats as stats  # don't delete: necessary for get_external_function
from collections import OrderedDict as odict
from ast import parse
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    # Suppress message about optional dependency
    from fuzzywuzzy import process as fuzzy_process
if six.PY3:
    from inspect import cleandoc, getfullargspec as getargspec
    from math import gcd
    from collections.abc import Mapping
else:
    ModuleNotFoundError = ImportError
    from inspect import cleandoc, getargspec
    from fractions import gcd
    from collections import Mapping

# Local
from cobaya import __obsolete__
from cobaya.conventions import _package, subfolders, _p_dist, _likelihood, _p_value
from cobaya.conventions import _sampler, _theory
from cobaya.log import LoggedError

# Logger
import logging

log = logging.getLogger(__name__.split(".")[-1])


# Deepcopy workaround:
def deepcopyfix(olddict):
    if not hasattr(olddict, "keys"):
        return deepcopy(olddict)
    newdict = {}
    for key in olddict:
        if (key == 'theory' or key == 'instance' or key == 'external'):
            newdict[key] = olddict[key]
        else:
            # print(key)
            newdict[key] = deepcopy(olddict[key])
    return newdict


def get_class_base_filename(name, kind):
    """
    Gets absoluate file base name for a class module, relative to the package source, of a likelihood, theory or sampler.
    """
    if '.' not in name: name += '.' + name
    return os.path.join(os.path.dirname(__file__), subfolders[kind], name.replace('.', os.sep))


def get_class_module(name, kind):
    """
    Gets qualified module name, relative to the package source, of a likelihood, theory or sampler.
    """
    return '.' + subfolders[kind] + '.' + name


def get_kind(name, fail_if_not_found=True):
    """
    Given a hefully unique module name, tries to determine it's kind:
    ``sampler``, ``theory`` or ``likelihood``.
    """
    try:
        return next(
            k for k in [_sampler, _theory, _likelihood]
            if name in get_available_modules(k))
    except StopIteration:
        if fail_if_not_found:
            raise LoggedError(log, "Could not determine kind of module %s", name)
        else:
            return None


def get_class(name, kind=None, None_if_not_found=False):
    """
    Retrieves the requested likelihood (default) or theory class.

    ``info`` must be a dictionary of the kind ``{[class_name]: [options]}``.

    Raises ``ImportError`` if class not found in the appropriate place in the source tree.

    If 'kind=None' is not given, tries to guess it if the module name is unique (slow!).
    """
    if kind is None:
        kind = get_kind(name)
    class_name = name.split('.')[-1]
    class_folder = get_class_module(name, kind)
    try:
        return getattr(import_module(class_folder, package=_package), class_name)
    except:
        if ((sys.exc_info()[0] is ModuleNotFoundError and
             str(sys.exc_info()[1]).rstrip("'").endswith(name))):
            if None_if_not_found:
                return None
            raise LoggedError(
                log, "%s '%s' not found. Maybe you meant one of the following "
                "(capitalization is important!): %s",
                kind.capitalize(), name,
                fuzzy_match(name, get_available_modules(kind), n=3))
        else:
            log.error("There was a problem when importing %s '%s':", kind, name)
            raise sys.exc_info()[1]


def get_available_modules(kind):
    """
    Gets all modules' names of a given kind.
    """
    folders = sorted([
        f for f in os.listdir(os.path.join(os.path.dirname(__file__), subfolders[kind]))
        if (not f.startswith("_") and not f.startswith(".") and
        os.path.isdir(os.path.join(os.path.dirname(__file__), subfolders[kind], f)))])
    with_nested = []
    for f in folders:
        dotpy_files = sorted([
            f for f in os.listdir(
                os.path.join(os.path.dirname(__file__), subfolders[kind], f))
            if f.lower().endswith(".py")])
        # if *non-empty* __init__, assume it containts a sigle module named as the folder
        try:
            __init__filename = next(
                p for p in dotpy_files if os.path.splitext(p)[0] == "__init__")
            __init__with_path = os.path.join(
                os.path.dirname(__file__), subfolders[kind], f, __init__filename)
        except:
            __init__filename = None
        if __init__filename and os.path.getsize(__init__with_path):
            with_nested += [f]
        else:
            dotpy_files = [f for f in dotpy_files if not f.startswith("_")]
            with_nested += [f + "." +  os.path.splitext(dpyf)[0]for dpyf in dotpy_files
                            if os.path.splitext(dpyf)[0] != f]
    return with_nested


def get_external_function(string_or_function, name=None, or_class=False):
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
        string_or_function = string_or_function.get(_p_value, None)
    if isinstance(string_or_function, six.string_types):
        try:
            if "import_module" in string_or_function:
                sys.path.append(os.path.realpath(os.curdir))
            function = eval(string_or_function)
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
        raise LoggedError(log, "No specific prior info given for sampled parameter '%s'." % param)
    # What distribution?
    try:
        dist = info2.pop(_p_dist).lower()
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
    """WARNING: logpdf(nan) = -inf"""
    if not hasattr(self, "_cobaya_mlogscale"):
        self._cobaya_mlogscale = -np.log(self.kwds["scale"])
        self._cobaya_max = self.kwds["loc"] + self.kwds["scale"]
    x_ = np.array(x)
    return np.where(np.logical_and(x_ >= self.kwds["loc"], x_ <= self._cobaya_max),
                    self._cobaya_mlogscale, -np.inf)


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
    (only present if applicable):

      `duplicates_[A|B]`: duplicate elemnts in list 1|2
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
    return (symbol * length + "\n" + msg_clean + "\n" + symbol * length + "\n")


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
    You are using an archived version of Cobaya, which is no loger maintained.
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
        _copy = (base.__class__)()
        for key, value in (base or {}).items():
            key_copy = deepcopy(key)
            _copy[key_copy] = deepcopy_where_possible(value)
        return _copy
    else:
        try:
            return deepcopy(base)
        except:
            return base
