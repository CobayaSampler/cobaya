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

# Local
from cobaya.conventions import subfolders
from cobaya.conventions import input_p_label, input_p_dist, input_prior, input_debug
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)


# Other I/O-related tools #################################################################

def get_folder(name, kind, sep="/", absolute="True"):
    pre = os.path.dirname(__file__) if absolute else ""
    return pre + sep + subfolders[kind] + sep + name

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

def make_header(kind, module):
    """Created a header for a particular module of a particular kind."""
    return ("="*80).join(["", "\n %s : %s \n"%(kind.title(), module), "\n"])

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
