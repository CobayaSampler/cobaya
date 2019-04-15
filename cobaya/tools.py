"""
.. module:: tools

:Synopsis: General tools
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import absolute_import, division

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

if six.PY3:
    from inspect import cleandoc, getfullargspec as getargspec
    from math import gcd
    from collections.abc import Mapping
else:
    from inspect import cleandoc, getargspec
    from fractions import gcd
    from collections import Mapping

# Local
from cobaya import __obsolete__
from cobaya.conventions import _package, subfolders, _p_dist, _likelihood, _p_value
from cobaya.log import HandledException

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


def get_folder(name, kind, sep=os.sep, absolute="True"):
    """
    Gets folder, relative to the package source, of a likelihood, theory or sampler.
    """
    pre = (os.path.dirname(__file__) + sep if absolute
           else "" + (sep if sep == "." else ""))
    return pre + subfolders[kind] + sep + name


def get_class(name, kind=_likelihood):
    """
    Retrieves the requested likelihood (default) or theory class.

    ``info`` must be a dictionary of the kind ``{[class_name]: [options]}``.

    Raises ``ImportError`` if class not found in the appropriate place in the source tree.
    """
    class_folder = get_folder(name, kind, sep=".", absolute=False)
    try:
        return getattr(import_module(class_folder, package=_package), name)
    except:
        if sys.exc_info()[0] == ImportError and str(sys.exc_info()[1]).endswith(name):
            log.error("%s '%s' not found (wrong capitalization?)",
                      kind.capitalize(), name)
            raise HandledException
        else:
            log.error("There was a problem when importing %s '%s':", kind, name)
            raise sys.exc_info()[1]


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
        string_or_function = string_or_function.get(_p_value, None)
    if isinstance(string_or_function, six.string_types):
        try:
            if "import_module" in string_or_function:
                sys.path.append(os.path.realpath(os.curdir))
            function = eval(string_or_function)
        except Exception as e:
            log.error("Failed to load external function%s: '%r'",
                      " '%s'" % name if name else "", e)
            raise HandledException
    else:
        function = string_or_function
    if not callable(function):
        log.error("The external function provided " +
                  ("for '%s'x " % name if name else "") +
                  "is not an actual function. Got: '%r'", function)
        raise HandledException
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
    for update_key, update_value in (update or {}).items():
        update_value = update_value or odict()
        base = base or odict()
        if isinstance(update_value, Mapping):
            base[update_key] = recursive_update(
                base.get(update_key, odict()), update_value)
        else:
            base[update_key] = update_value
    # Trim terminal (o)dicts
    for k, v in (base or {}).items():
        if v in [odict(), {}]:
            base[k] = None
    return base


def make_header(kind, module=None):
    """Creates a header for a particular module of a particular kind."""
    return ("=" * 80).join(["", "\n %s" % kind.title() +
                            (": %s" % module if module else "") + "\n", "\n"])


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
        log.error("Could not convert '%r' to a number.", n)
        raise HandledException
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
        log.error("No specific prior info given for sampler parameter '%s'." % param)
        raise HandledException
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
        log.error("Error creating the prior for parameter '%s': "
                  "The distribution '%s' is unknown to 'scipy.stats'. "
                  "Check the list of allowed possibilities in the docs.", param, dist)
        raise HandledException
    # Recover loc,scale from min,max
    # For coherence with scipy.stats, defaults are min,max=0,1
    if "min" in info2 or "max" in info2:
        if "loc" in info2 or "scale" in info2:
            log.error("You cannot use the 'loc/scale' convention and the 'min/max' "
                      "convention at the same time. Either use one or the other.")
            raise HandledException
        try:
            mini = np.float(info2.pop("min"))
        except KeyError:
            mini = 0
        try:
            maxi = np.float(info2.pop("max"))
        except KeyError:
            maxi = 1
        info2["loc"] = mini
        info2["scale"] = maxi - mini
    # Check for improper priors
    if not np.all(np.isfinite([info2.get(x, 0) for x in ["loc", "scale", "min", "max"]])):
        log.error("Improper prior for parameter '%s'.", param)
        raise HandledException
    # Generate and return the frozen distribution
    try:
        return pdf_dist(**info2)
    except TypeError as tp:
        log.error(
            "'scipy.stats' produced an error: <<%r>>. "
            "This probably means that the distribution '%s' "
            "does not recognize the parameter mentioned in the 'scipy' error above.",
            tp.message, dist)
        raise HandledException


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


def compare_params_lists(list_A, list_B):
    """
    Compares two parameter lists, and returns a dict with the following keys
    (only present if applicable):

      `duplicates_[A|B]`: duplicate elemnts in list 1|2
    """
    result = {}
    names = ["A", "B"]
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
            result["%s_but_not_%s" % (n1, n2)] = list(missing)
    return result


def relative_to_int(numbers, precision=1 / 10):
    """
    Turns relative numbers (e.g. relative speeds) into integer,
    up to some given `precision` on differences.
    """
    numbers = np.array(np.round(np.array(numbers) / min(numbers) / precision), dtype=int)
    return np.array(
        numbers / np.ufunc.reduce(np.frompyfunc(gcd, 2, 1), numbers), dtype=int)


def create_banner(msg):
    """
    Puts message into an attention-grabbing banner.
    """
    msg_clean = cleandoc(msg)
    longest_line_len = max([len(line) for line in msg_clean.split("\n")])
    return ("*" * longest_line_len + "\n" + msg_clean + "\n" + "*" * longest_line_len)


def warn_deprecation_python2():
    msg = """
    *WARNING*: Python 2 support will eventually be dropped
    (it is already unsupported by many scientific Python modules).

    Please use Python 3!

    In some systems, the Python 3 command may be python3 instead of python.
    If that is the case, use pip3 instead of pip to install Cobaya.
    """
    if not six.PY3:
        print(create_banner(msg))


def warn_deprecation_version():
    msg = """
    *WARNING*: You are using an obsolete version of Cobaya, which is no loger maintained.

    Please, update asap to the last version (e.g. with ``pip install cobaya --upgrade``).
    """
    if __obsolete__:
        print(create_banner(msg))


def warn_deprecation():
    warn_deprecation_python2()
    warn_deprecation_version()
