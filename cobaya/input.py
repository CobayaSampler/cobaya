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
from copy import deepcopy
from importlib import import_module

# Local
from cobaya.conventions import package, _defaults_file, _params, _p_label
from cobaya.conventions import _prior, _theory, _likelihood, _sampler, _external
from cobaya.tools import get_folder
from cobaya.yaml_custom import yaml_load_file
from cobaya.log import HandledException
from cobaya.parametrisation import is_sampled_param, is_derived_param

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
        info = yaml_load_file(input_file)
        # if output_prefix not defined, default to input_file name (sans ext.) as prefix;
        if "output_prefix" not in info:
            info["output_prefix"] = file_name
        # warn if no output, since we are in shell-invocation mode.
        elif info["output_prefix"] is None:
            log.warning("WARNING: Output explicitly supressed with 'ouput_prefix: null'")
    else:
        log.error("Extension of input file '%s' not recognised.", input_file)
        raise HandledException
    return info


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
    """Returns modules all requested as an odict ``{kind: set([modules])}``.
    Priors are not included."""
    modules = odict()
    for info in infos:
        for field in [_theory, _likelihood, _sampler]:
            if field not in modules:
                modules[field] = set()
            modules[field] |= (lambda v: set(v) if v else set())(info.get(field))
            modules[field] = modules[field].difference(set([None]))
    # pop empty fields
    for k,v in modules.iteritems():
        if not v:
            modules.pop(k)
    return modules


def get_full_info(info):
    """
    Creates an updated info starting from the defaults for each module and updating it
    with the input info.
    """
    # Don't modify the original input!
    input_info = deepcopy(info)
    # Creates an equivalent info using only the defaults
    full_info = odict()
    default_params_info = odict()
    default_prior_info = odict()
    modules = get_modules(input_info)
    for block in modules:
        full_info[block] = odict()
        for module in modules[block]:
            # Start with the default class options
            full_info[block][module] = deepcopy(getattr(
                import_module(package+"."+block, package=package), "class_options", {}))
            # Go on with defaults
            path_to_defaults = os.path.join(get_folder(module, block), _defaults_file)
            try:
                default_module_info = yaml_load_file(path_to_defaults)
            except IOError:
                # probably an external module
                default_module_info = {block: {module: {}}}
                log.debug("Module %s:%s does not have a defaults file. "%(block, module) +
                          "Maybe it is an external module.")
            try:
                full_info[block][module].update(default_module_info[block][module] or {})
            except KeyError:
                log.error("The defaults file for '%s' should be structured "
                          "as %s:%s:{[options]}.", module, block, module)
                raise HandledException
            # Update the options with the input file
            # Consistency is checked only up to first level! (i.e. subkeys may not match)
            # First deal with cases "no options" and "external function"
            try:
                input_info[block][module] = input_info[block][module] or {}
            except TypeError:
                log.error("Your input info is not well formatted at the '%s' block. "
                          "It must be a dictionary {'%s':{options}, ...}. ", block, block)
                raise HandledException
            if not hasattr(input_info[block][module], "get"):
                input_info[block][module] = {_external: input_info[block][module]}
            options_not_recognised = (set(input_info[block][module])
                                      .difference(set([_external]))
                                      .difference(set(full_info[block][module])))
            if options_not_recognised:
                log.error("'%s' does not recognise some options: '%r'. "
                          "To see the allowed options, check out the file '%s'",
                          module, tuple(options_not_recognised), path_to_defaults)
                raise HandledException
            full_info[block][module].update(input_info[block][module])
            # Store default parameters and priors of class, and save to combine later
            if block == _likelihood:
                params_info = default_module_info.get(_params, {})
                full_info[block][module].update({_params:params_info})
                default_params_info[module] = params_info
                default_prior_info[module] = default_module_info.get(_prior, {})
    # Add priors info, after the necessary checks
    if _prior in input_info or any(default_prior_info.values()):
        full_info[_prior] = input_info.get(_prior, odict())
    for prior_info in default_prior_info.values():
        for name, prior in prior_info.iteritems():
            if full_info[_prior].get(name, prior) != prior:
                log.error("Two different priors cannot have the same name: '%s'.", name)
                raise HandledException
            full_info[_prior][name] = prior
    # Add parameters info, after the necessary updates and checks
    full_info[_params] = merge_params_info(input_info.get(_params, {}),
                                           defaults=default_params_info)
    # Rest of the options
    for k,v in input_info.iteritems():
        if k not in full_info:
            full_info[k] = v
    return full_info


def merge_params_info(params_info, defaults=None):
    """
    Merges input and default parameters info, after performing some consistency checks.
    """
    # First, merge defaults. Impose multiple defined (=shared) parameters have equal info
    defaults_merged = odict()
    for lik, params in defaults.iteritems():
        for p, info in params.iteritems():
            # if already there, check consistency
            if p in defaults_merged:
                log.debug("Parameter '%s' multiply defined.", p)
                if info != defaults_merged[p]:
                    log.error("Parameter '%s' multiply defined, but inconsistent info: "
                              "For likelihood '%s' is '%r', but for some other likelihood"
                              " it was '%r'. Check your defaults!",
                              p, lik, info, defaults_merged[p])
                    raise HandledException
            defaults_merged[p] = info
    # Combine with the input parameter info (make sure the theory parameters come first)
    params_info_copy = deepcopy(params_info)
    theory_params = params_info_copy.pop(_theory, None)
    info_updated = odict()
    if theory_params:
        info_updated.update({_theory: theory_params})
    info_updated.update(defaults_merged)
    info_updated.update(params_info)
    # Inherit labels (for sampled and derived) and min/max (just for derived params)
    getter = lambda info, key: getattr(info, "get", lambda x: None)(key)
    for p in defaults_merged:
        default_label = getter(defaults_merged[p], _p_label)
        if (default_label and
                (is_sampled_param(info_updated[p]) or is_derived_param(info_updated[p]))):
            info_updated[p][_p_label] = info_updated[p].get(_p_label) or default_label
        limits = ["min", "max"]
        default_limits = odict([[lim,getter(defaults_merged[p], lim)] for lim in limits])
        if default_limits.values() != [None, None] and is_derived_param(info_updated[p]):
            info_updated[p].update(default_limits)
    return info_updated
