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
from collections import Mapping, OrderedDict as odict
from copy import deepcopy
from importlib import import_module

# Local
from cobaya.conventions import package, _params, _p_label, _products_path
from cobaya.conventions import _prior, _theory, _likelihood, _sampler, _external
from cobaya.conventions import _output_prefix, _debug_file
from cobaya.tools import get_folder
from cobaya.yaml import yaml_load_file
from cobaya.log import HandledException
from cobaya.parametrization import is_sampled_param, is_derived_param

# Logger
import logging
log = logging.getLogger(__name__.split(".")[-1])


def load_input(input_file):
    """
    Loads general info, and splits it into the right parts.
    """
    file_name, extension = os.path.splitext(input_file)
    file_name = os.path.basename(file_name)
    if extension not in (".yaml",".yml"):
        log.error("Extension of input file '%s' not recognized.", input_file)
        raise HandledException
    info = yaml_load_file(input_file)
    # if output_prefix not defined, default to input_file name (sans ext.) as prefix;
    if _output_prefix not in info:
        info[_output_prefix] = file_name
    # warn if no output, since we are in shell-invocation mode.
    elif info[_output_prefix] is None:
        log.warning("WARNING: Output explicitly supressed with 'ouput_prefix: null'")
    # contained? Ensure that output is sent where it should
    if "CONTAINED" in os.environ:
        for out in [_output_prefix, _debug_file]:
            if info.get(out):
                if not info[out].startswith("/"):
                    info[out] = os.path.join(_products_path, info[out])
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
                modules[field] = []
            modules[field] += [a for a in (info.get(field) or [])
                               if a not in modules[field]]
    # pop empty blocks
    for k,v in list(modules.items()):
        if not v:
            modules.pop(k)
    return modules


def get_default_info(module, kind):
    path_to_defaults = os.path.join(get_folder(module, kind), module+".yaml")
    try:
        default_module_info = yaml_load_file(path_to_defaults)
    except IOError:
        # probably an external module
        default_module_info = {kind: {module: {}}}
        log.debug("Module %s:%s does not have a defaults file. "%(kind, module) +
                  "Maybe it is an external module.")
    try:
        default_module_info[kind][module]
    except KeyError:
        log.error("The defaults file for '%s' should be structured "
                  "as %s:%s:{[options]}.", module, kind, module)
        raise HandledException
    return default_module_info


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
            default_module_info = get_default_info(module, block)
            full_info[block][module].update(default_module_info[block][module] or {})
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
            options_not_recognized = (set(input_info[block][module])
                                      .difference(set([_external]))
                                      .difference(set(full_info[block][module])))
            if options_not_recognized:
                log.error("'%s' does not recognize some options: '%r'. "
                          "To see the allowed options, check out the documentation of "
                          "this module", module, tuple(options_not_recognized))
                raise HandledException
            full_info[block][module].update(input_info[block][module])
            # Store default parameters and priors of class, and save to combine later
            if block == _likelihood:
                params_info = default_module_info.get(_params, {})
                full_info[block][module].update({_params:list(params_info.keys())})
                default_params_info[module] = params_info
                default_prior_info[module] = default_module_info.get(_prior, {})
    # Add priors info, after the necessary checks
    if _prior in input_info or any(default_prior_info.values()):
        full_info[_prior] = input_info.get(_prior, odict())
    for prior_info in default_prior_info.values():
        for name, prior in prior_info.items():
            if full_info[_prior].get(name, prior) != prior:
                log.error("Two different priors cannot have the same name: '%s'.", name)
                raise HandledException
            full_info[_prior][name] = prior
    # Add parameters info, after the necessary updates and checks
    defaults_merged = merge_default_params_info(default_params_info)
    full_info[_params] = merge_params_info(defaults_merged, input_info.get(_params, {}))
    # Rest of the options
    for k,v in input_info.items():
        if k not in full_info:
            full_info[k] = v
    return full_info


def merge_default_params_info(defaults):
    """
    Merges default parameters info for all likelihoods.
    Checks that multiple defined (=shared) parameters have equal info.
    """
    defaults_merged = odict()
    for lik, params in defaults.items():
        for p, info in params.items():
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
    return defaults_merged


def merge_params_info(*params_infos):
    """
    Merges parameter infos, starting from the first one
    and updating with each additional one.
    Labels (for sampled and derived) and min/max
    (just for derived params) are inherited from defaults
    (but not if one of min/max is re-defined: in that case,
    to avoid surprises, the other one is set to None=+/-inf)
    """
    getter = lambda info, key: getattr(info, "get", lambda x: None)(key)
    previous_info = deepcopy(params_infos[0])
    for new_info in params_infos[1:]:
        current_info = deepcopy(previous_info)
        if not new_info:
            continue
        current_info.update(deepcopy(new_info))
        # inherit labels and bounds
        for p in previous_info:
            default_label = getter(previous_info[p], _p_label)
            if (default_label and (is_sampled_param(current_info[p]) or
                                   is_derived_param(current_info[p]))):
                current_info[p][_p_label] = (
                    new_info.get(p, {}).get(_p_label) or default_label)
            bounds = ["min", "max"]
            default_bounds = odict(
                [[bound, getter(previous_info[p], bound)] for bound in bounds])
            if (list(default_bounds.values()) != [None, None] and
                    is_derived_param(current_info.get(p))):
                if current_info.get(p) is None:
                    current_info[p] = {}
                new_bounds = {bound:new_info.get(p, {}).get(bound) for bound in bounds}
                if list(new_bounds.values()) == [None, None]:
                    for bound, value in default_bounds.items():
                        current_info[p][bound] = value
        previous_info = current_info
    return current_info


def recursive_update_yaml(d, u):
    """
    Recursive dictionary update, from `this stackoverflow question
    <https://stackoverflow.com/questions/3232943>`_.
    Modified for yaml input, where None and {} are almost equivalent
    """
    for k, v in u.items():
        v = v or {}
        d = d or {}
        if isinstance(v, Mapping):
            d[k] = recursive_update_yaml(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def merge_info(*infos):
    previous_info = deepcopy(infos[0])
    for new_info in infos[1:]:
        previous_params_info = deepcopy(previous_info.pop(_params, {}) or {})
        new_params_info = deepcopy(new_info).pop(_params, {}) or {}
        # NS: params have been popped, since they have their own merge function
        current_info = recursive_update_yaml(deepcopy(previous_info), new_info)
        current_info[_params] = merge_params_info(previous_params_info, new_params_info)
        previous_info = current_info
    return current_info
