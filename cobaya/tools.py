"""
.. module:: tools

:Synopsis: General tools
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# Global
import os
from collections import OrderedDict as odict
from copy import deepcopy
from importlib import import_module
import numpy as np
import inspect
import scipy.stats as stats
from collections import namedtuple

# Local
from cobaya.conventions import package, subfolders, input_likelihood
from cobaya.conventions import input_p_label, input_p_dist, input_prior, input_debug
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)


def get_folder(name, kind, sep="/", absolute="True"):
    """
    Gets folder, relative to the package source, of a likelihood, theory or sampler.
    """
    pre = os.path.dirname(__file__)+sep if absolute else ""+(sep if sep=="." else "")
    return pre + subfolders[kind] + sep + name

def get_class(name, kind=input_likelihood):
    """
    Retrieves the requested likelihood (default) or theory class.

    ``info`` must be a dictionary of the kind ``{[class_name]: [options]}``.

    Raises ``ImportError`` if class not found in the appropriate place in the source tree.
    """
    class_folder = get_folder(name, kind, sep=".", absolute=False)
    try:
        return getattr(import_module(class_folder, package=package), name)
    except ImportError:
        log.error("%s '%s' not found. Maybe check capitalisation.", kind.capitalize(), name)
        raise HandledException


# Tuple that contains and describes an external function
external_tuple = namedtuple("external_tuple", ("logp", "args"))

def get_external_function(string_or_function):
    """
    Processes an external prior or likelihood, given as a string or a function.

    If the input is a string, it must be evaluable to a function. It can contain import
    statements using :module:`importlib`'s ``import_module``, e.g.
    ``import_module("my_file").my_function``. You can access :module:`scipy.stats` and
    :module:`numpy` members under the handles ``stats`` and ``np`` respectively.

    Returns a named tuple ``("logp": [function], "args": [list of arguments])``.
    """
    if isinstance(string_or_function, basestring):
        try:
            function = eval(string_or_function)
        except Exception, e:
            log.error("Failed to load external function: '%r'", e)
            raise HandledException
    else:
        function = string_or_function
    if not callable(function):
        log.error("The external function provided is not an actual function.")
        raise HandledException
    return external_tuple(function, inspect.getargspec(function)[0])

def make_header(kind, module):
    """Created a header for a particular module of a particular kind."""
    return ("="*80).join(["", "\n %s : %s \n"%(kind.title(), module), "\n"])

def get_labels(params_info):
    """
    Extract labels from input info (sampled and derived).
    Uses parameter name if label not defined.
    Ensures LaTeX notation.
    """
    labels = odict()
    for p in params_info:
        label = params_info[p].get(input_p_label)
        if not label:
            label = p
        labels[p] = ensure_latex(label)
    return labels

def ensure_latex(string):
    """Inserts $'s at the beginning and end of the string, if necessary."""
    if string.strip()[0] != r"$":
        string = r"$"+string
    if string.strip()[-1] != r"$":
        string = string+r"$"
    return string

def ensure_nolatex(string):
    """Removes $'s at the beginning and end of the string, if necessary."""
    return string.strip().lstrip("$").rstrip("$")

def get_scipy_1d_pdf(info):
    """Generates 1d priors from scipy's pdf's from input info."""
    param = info.keys()[0]
    info2 = deepcopy(info[param])
    # What distribution?
    try:
        dist = info2.pop(input_p_dist).lower()
    # Not specified: uniform by default
    except KeyError:
        dist = "uniform"
    # Number: uniform with 0 width
    except AttributeError:
        dist = "uniform"
        info2 = {"loc":info2, "scale":0}
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
        info2["scale"] = maxi-mini
    # Generate and return the frozen distribution
    try:
        return pdf_dist(**info2)
    except TypeError as tp:
        log.error(
            "'scipy.stats' produced an error: <<%r>>. "
            "This probably means that the distribution '%s' "
            "does not recognise the parameter mentioned in the 'scipy' error above.",
            tp.message, dist)
        raise HandledException

# log-uniform distribution (missing in scipy.stats), with some methods implemented
# -- for now, not inheriting from scipy.stats.rv_continuous, since the output of
#    most of the methods would be different.
#from scipy.stats import uniform
#def loguniform():
#    def __init__(loc=1, scale=0):
#        self.uniform =
