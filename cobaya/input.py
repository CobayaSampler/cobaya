"""
.. module:: input

:Synopsis: Input-related functions
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import os
from collections import OrderedDict as odict
import numpy as np
import datetime
from numbers import Number
from getdist import MCSamples

# Local
from cobaya.conventions import subfolders, defaults_file, input_params, input_p_label
from cobaya.conventions import input_prior, input_theory, input_likelihood, input_sampler
from cobaya.tools import get_labels
from cobaya.yaml_custom import yaml_custom_load
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)

def load_input(input_file):
    """
    Loads general info, and splits it into the right parts.
    """
    file_name, extension = os.path.splitext(input_file)
    file_name = os.path.basename(file_name)
    if extension in (".yaml",".yml"):
        info = load_input_yaml(input_file)
        # if output_prefix not defined, default to input_file name (sans ext.) as prefix;
        if "output_prefix" not in info:
            info["output_prefix"] = file_name
        # warn if no output, since we are in shell-invocation mode.
        elif info["output_prefix"] == None:
            log.warning("WARNING: Output explicitly supressed with 'ouput_prefix: null'")
    else:
        log.error("Extension '%s' of input file '%s' not recognised.", extension, input_file)
        raise HandledException
    return info

def load_input_yaml(input_file, defaults_file=True):
    """Wrapper to load a yaml file."""
    with open(input_file,"r") as stream:
        lines = "".join(stream.readlines())
    file_name = (input_file if not defaults_file
                 else "/".join(input_file.split(os.sep)[-2:]))
    return yaml_custom_load(lines, file_name=defaults_file)

# MPI wrapper for loading the input info
def load_input_MPI(input_file):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Load input (only one process does read the input file)
    if rank == 0: 
        info = load_input(input_file)
    else:
        info = None
    info = comm.bcast(info, root=0)
    return info

def get_modules(*infos):
    """Returns modules all requested as an odict ``{kind: set([modules])}``."""
    modules = odict()
    for info in infos:
        for field in [input_theory, input_likelihood, input_sampler]:
            if not field in modules:
                modules[field] = set()
            modules[field] |= (lambda v: set(v) if v else set())(info.get(field))
            modules[field] = modules[field].difference(set([None]))
    # pop empty fields
    for k,v in modules.iteritems():
        if not v:
            modules.pop(k)
    return modules

# Class initialisation using default and input info
def load_input_and_defaults(instance, input_info, kind=None, load_defaults_file=True):
    """
    Loads the default info for a class instance, updates with the input info,
    and returns the updated default+input info.
    """
    # 1. Loading defaults
    class_name = instance.__class__.__name__
    # start with class-level-defaults, if they exist
    defaults = odict([[class_name,getattr(instance, "_parent_defaults", odict())]])
    # then load the class-specific defaults from its defaults.yaml file
    if load_defaults_file:
        class_defaults_path = os.path.join(
            os.path.dirname(__file__), subfolders[kind], class_name, defaults_file)
        class_defaults = load_input_yaml(class_defaults_path, defaults_file=True)[kind]
        if not class_name in class_defaults:
            log.error(
                "The defaults file should contain the field '%s:%s'.", kind, class_name)
            raise HandledException
        defaults[class_name].update(class_defaults[class_name])
    # 2. Update it with the input_info -- parameters dealt with later (see "load_params")
    if kind != "sampler":
        setattr(instance, "_params_defaults", defaults[class_name].pop(input_params, odict()))
    # consistency is checked only up to first level! (i.e. subkeys may not match)
    if input_info == None:
        input_info = {}
    for k,v in input_info.iteritems():
        if k in defaults[class_name]:
            defaults[class_name][k] = v
        else:
            log.error("'%s' does not recognise the input option '%s'. "
                      "To see the allowed options, check out the file '%s'",
                      class_name, k, class_defaults_path)
            raise HandledException
    # 3. Set the options as attibutes with the updated info
    for k,v in defaults[class_name].iteritems():
        setattr(instance, k, v)
    return defaults


def load_params(instance, params_info, allow_unknown_prefixes=[]):
    """
    Merges the default and input parameters, separating them into:
    - An ordered dictionary of *fixed* parameters
    - An ordered dictionary of *sampled* parameters
    - An ordered dictionary of *derived* parameters
    Use the keyword `allow_unknown_prefixes` to specify a list of prefixes for
    parameters names such that parameters starting by these are kept even if
    not previously defined -- use `allow_unknown_prefixes=[""]` for accepting
    any parameter name.
    """
    class_name = instance.__class__.__name__
    # Create placeholders if they are not there yet
    params_kinds = ("fixed", "sampled", "derived")
    for pt in params_kinds:
        if not hasattr(instance, pt):
            setattr(instance, pt, odict())
    if not params_info:
        return
    # Safeguard for prefixes
    if isinstance(allow_unknown_prefixes, basestring):
        allow_unknown_prefixes = [allow_unknown_prefixes]
    # Store params
    for p, v in params_info.iteritems():
# This was for parameters sharing a name and distinguished with the lik or theory class
# as a prefix -- deprecated, waiting for an updated implementation
#        # If it does not belong to this class, ignore
#        if separator in param:
#            pclass, param = param.split(separator)
#            if pclass != class_name:
#                log.info("Class '%s': Ignoring parameter '%s', meant for '%s' instead."%(
#                    class_name, param, pclass))
#                print("TODO: really do something here!")
        # Fixed parameter: number; Derived parameter: no `prior` key; Sampled: `prior` key
        if isinstance(v, Number):
            kind = "fixed"
        else:
            if v == None:
                v = {}
            if v.get("prior"):
                kind = "sampled"
            else:
                kind = "derived"
        # Already there?
        old_kind = None
        for kind2 in params_kinds:
            if p in getattr(instance, kind2):
                old_kind = kind2
        # If not there but unknown parameters matching prefix allowed, pretend it's there
        if old_kind == None:
            match_prefix = [p.startswith(prefix) for prefix in allow_unknown_prefixes]
            if not any(match_prefix):
                log.error(
                    "The parameter '%s' defined in the input file "
                    "is not known by '%s'. "
                    "The only parameter names allowed are those defined "
                    "in the 'defaults' file"+
                    (" or those starting with "+str(allow_unknown_prefixes)+"."
                     if allow_unknown_prefixes != None else "."), p, class_name)
                raise HandledException
            old_kind = kind
        # Inherit now-undefined latex label and derived parameter limits
        if old_kind in ["sampled", "derived"] and kind in ["sampled", "derived"]:
            if not v.get(input_p_label):
                old_label = getattr(instance, kind).get(p, {}).get(input_p_label, None)
                v[input_p_label] = old_label
        if old_kind == "derived" and kind in "derived":
            for lim in ["min", "max"]:
                if not v.get(lim):
                    old_lim = getattr(instance, kind).get(p, {}).get(lim, None)
                    v[lim] = old_lim
        # If same kind, change value (don't delete, to keep ordering)
        if old_kind == kind:
            getattr(instance, kind)[p] = v
        else:
            getattr(instance, old_kind).pop(p)
            getattr(instance, kind)[p] = v
    # Check for duplicates!
    all_names = []
    for kind in params_kinds:
        all_names += list(getattr(instance, kind).keys())
    if len(all_names) != len(set(all_names)):
        log.error("Some parameter of '%s' has been defined twice!", class_name)
        raise HandledException

def get_updated_params_info(instance):
    """Re-creates the params info from the loaded, updated parameters."""
    info = odict()
    for kind in ("fixed", "sampled", "derived"):
        for p, v in getattr(instance, kind).iteritems():
            info[p] = v
    return info
