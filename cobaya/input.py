"""
.. module:: input

:Synopsis: Input-related functions
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
from functools import reduce

# Global
import os
from collections import OrderedDict as odict
from copy import deepcopy
from importlib import import_module
import inspect
from six import string_types
from itertools import chain

# Local
from cobaya.conventions import _package, _products_path, _path_install, _resume
from cobaya.conventions import _output_prefix, _debug, _debug_file
from cobaya.conventions import _params, _prior, _theory, _likelihood, _sampler, _external
from cobaya.conventions import _p_label, _p_derived, _p_ref, _p_drop, _p_value, _p_renames
from cobaya.conventions import _p_proposal
from cobaya.tools import get_folder, recursive_update, recursive_odict_to_dict
from cobaya.yaml import yaml_load_file
from cobaya.log import HandledException
from cobaya.parameterization import expand_info_param
from cobaya.mpi import get_mpi_comm, am_single_or_primary_process

# Logger
import logging

log = logging.getLogger(__name__.split(".")[-1])


def load_input(input_file):
    """
    Loads general info, and splits it into the right parts.
    """
    file_name, extension = os.path.splitext(input_file)
    file_name = os.path.basename(file_name)
    if extension not in (".yaml", ".yml"):
        log.error("Extension of input file '%s' not recognized.", input_file)
        raise HandledException
    info = yaml_load_file(input_file) or {}
    # if output_prefix not defined, default to input_file name (sans ext.) as prefix;
    if _output_prefix not in info:
        info[_output_prefix] = file_name
    # warn if no output, since we are in shell-invocation mode.
    elif info[_output_prefix] is None:
        log.warning("WARNING: Output explicitly suppressed with 'output_prefix: null'")
    # contained? Ensure that output is sent where it should
    if "CONTAINED" in os.environ:
        for out in [_output_prefix, _debug_file]:
            if info.get(out):
                if not info[out].startswith("/"):
                    info[out] = os.path.join(_products_path, info[out])
    return info


# MPI wrapper for loading the input info
def load_input_MPI(input_file):
    if am_single_or_primary_process():
        info = load_input(input_file)
    else:
        info = None
    info = get_mpi_comm().bcast(info, root=0)
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
    for k, v in list(modules.items()):
        if not v:
            modules.pop(k)
    return modules


def get_default_info(module, kind):
    path_to_defaults = os.path.join(get_folder(module, kind), module + ".yaml")
    try:
        default_module_info = yaml_load_file(path_to_defaults)
    except IOError:
        # probably an external module
        default_module_info = {kind: {module: {}}}
        log.debug("Module %s:%s does not have a defaults file. " % (kind, module) +
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
                import_module(_package + "." + block, package=_package), "class_options", {}))
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
                                      .difference(set([_external, _p_renames]))
                                      .difference(set(full_info[block][module])))
            if options_not_recognized:
                if default_module_info[block][module]:
                    # Internal module
                    log.error("'%s' does not recognize some options: '%r'. "
                              "To see the allowed options, check out the documentation of"
                              " this module", module, tuple(options_not_recognized))
                    raise HandledException
                else:
                    # External module
                    log.error("External %s '%s' does not recognize some options: '%r'. "
                              "Check the documentation for 'external %s'.",
                              block, module, tuple(options_not_recognized), block)
                    raise HandledException
            full_info[block][module].update(input_info[block][module])
            # Store default parameters and priors of class, and save to combine later
            if block == _likelihood:
                params_info = default_module_info.get(_params, {})
                full_info[block][module].update({_params: list(params_info)})
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
    # Add aliases for theory params (after merging!)
    if _theory in full_info:
        renames = list(full_info[_theory].values())[0].get(_p_renames)
        str_to_list = lambda x: ([x] if isinstance(x, string_types) else x)
        renames_flat = [set([k] + str_to_list(v)) for k, v in renames.items()]
        for p in full_info.get(_params, {}):
            # Probably could be made faster by inverting the renames dicts *just once*
            renames_pairs = [a for a in renames_flat if p in a]
            if renames_pairs:
                this_renames = reduce(
                    lambda x, y: x.union(y), [a for a in renames_flat if p in a])
                full_info[_params][p][_p_renames] = list(
                    set(this_renames).union(set(
                        str_to_list(full_info[_params][p].get(_p_renames, []))))
                        .difference(set([p])))
    # Rest of the options
    for k, v in input_info.items():
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
    current_info = odict(
        [[p, expand_info_param(v)] for p, v in params_infos[0].items() or {}])
    for new_info in params_infos[1:]:
        if not new_info:
            continue
        for p, new_info_p in new_info.items():
            if p not in current_info:
                current_info[p] = odict()
            new_info_p = expand_info_param(new_info_p)
            current_info[p].update(deepcopy(new_info_p))
            # Account for incompatibilities: "prior" and ("value" or "derived"+bounds)
            incompatibilities = {_prior: [_p_value, _p_derived, "min", "max"],
                                 _p_value: [_prior, _p_ref, _p_proposal],
                                 _p_derived: [_prior, _p_drop, _p_ref, _p_proposal]}
            for f1, incomp in incompatibilities.items():
                if f1 in new_info_p:
                    for f2 in incomp:
                        current_info[p].pop(f2, None)
    # Re-sort, so that rightmost info takes precedence *also* in the sorting
    new_order = chain(*[list(params) for params in params_infos[::-1]])
    # The following removes duplicates maintaining order (keeps the first occurrence)
    new_order = list(odict.fromkeys(new_order))
    current_info = odict([[p, current_info[p]] for p in new_order])
    return current_info


def merge_info(*infos):
    """
    Merges information dictionaries. Rightmost arguments take precedence.
    """
    previous_info = deepcopy(infos[0])
    for new_info in infos[1:]:
        previous_params_info = deepcopy(previous_info.pop(_params, odict()) or odict())
        new_params_info = deepcopy(new_info).pop(_params, odict()) or odict()
        # NS: params have been popped, since they have their own merge function
        current_info = recursive_update(deepcopy(previous_info), new_info)
        current_info[_params] = merge_params_info(previous_params_info, new_params_info)
        previous_info = current_info
    return current_info


def is_equal_info(info1, info2, strict=True, print_not_log=False):
    """
    Compares two information dictionaries.

    Set ``strict=False`` (default: ``True``) to ignore options that would not affect
    the statistics of a posterior sample, including order of params/priors/likelihoods.
    """
    if print_not_log:
        myprint = print
    else:
        myprint = log.info
    myname = inspect.stack()[0][3]
    ignore = set([]) if strict else set(
        [_debug, _debug_file, _resume, _path_install])
    ignore_params = (set([]) if strict
                     else set([_p_label, _p_renames, _p_ref, _p_proposal, "min", "max"]))
    if set(info1).difference(ignore) != set(info2).difference(ignore):
        myprint(myname + ": different blocks or options: %r (old) vs %r (new)" % (
            set(info1).difference(ignore), set(info2).difference(ignore)))
        return False
    for block_name in info1:
        if block_name in ignore:
            continue
        block1, block2 = info1[block_name], info2[block_name]
        if not hasattr(block1, "keys"):
            if block1 != block2:
                myprint(myname + ": different option '%s'" % block_name)
                return False
        if block_name in [_sampler, _theory]:
            # Internal order does NOT matter
            if set(block1) != set(block2):
                myprint(myname + ": different [%s]" % block_name)
                return False
            # Anything to ignore?
            for k in block1:
                module_folder = get_folder(k, block_name, sep=".", absolute=False)
                try:
                    ignore_k = getattr(import_module(
                        module_folder, package=_package), "ignore_at_resume", {})
                except ImportError:
                    ignore_k = {}
                block1k, block2k = deepcopy(block1[k]), deepcopy(block2[k])
                if not strict:
                    for kignore in ignore_k:
                        try:
                            block1k.pop(kignore, None)
                            block2k.pop(kignore, None)
                        except:
                            pass
                if recursive_odict_to_dict(block1k) != recursive_odict_to_dict(block2k):
                    myprint(
                        myname + ": different content of [%s:%s]" % (block_name, k))
                    return False
        elif block_name in [_params, _likelihood, _prior]:
            # Internal order DOES matter, but just up to 1st level
            f = list if strict else set
            if f(block1) != f(block2):
                myprint(myname + ": different [%s] or different order of them: %r vs %r" % (
                    block_name, list(block1), list(block2)))
                return False
            for k in block1:
                block1k, block2k = deepcopy(block1[k]), deepcopy(block2[k])
                if block_name == _params:
                    # Unify notation
                    block1k = expand_info_param(block1k)
                    block2k = expand_info_param(block2k)
                    if not strict:
                        for kignore in ignore_params:
                            try:
                                block1k.pop(kignore, None)
                                block2k.pop(kignore, None)
                            except:
                                pass
                        # Fixed params, it doesn't matter if they are saved as derived
                        for b in [block1k, block2k]:
                            if _p_value in b:
                                b.pop(_p_derived, None)
                if (recursive_odict_to_dict(block1k) != recursive_odict_to_dict(block2k)):
                    myprint(myname + ": different content of [%s:%s]" % (
                        block_name, k))
                    return False
    return True
