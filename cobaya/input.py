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
import pkg_resources

# Local
from cobaya.conventions import _package, _products_path, _path_install, _resume, _force
from cobaya.conventions import _output_prefix, _debug, _debug_file, _external, _self_name
from cobaya.conventions import _params, _prior, kinds

from cobaya.conventions import _p_label, _p_derived, _p_ref, _p_drop, _p_value, _p_renames
from cobaya.conventions import _p_proposal, _input_params, _output_params, _module_path
from cobaya.conventions import _yaml_extensions
from cobaya.tools import get_class_module, recursive_update, recursive_odict_to_dict
from cobaya.tools import fuzzy_match, deepcopy_where_possible, get_class, get_kind
from cobaya.yaml import yaml_load_file, yaml_dump
from cobaya.log import LoggedError
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
    if extension not in _yaml_extensions:
        raise LoggedError(log, "Extension of input file '%s' not recognized.", input_file)
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


def get_used_modules(*infos):
    """Returns all requested modules as an odict ``{kind: set([modules])}``.
    Priors are not included."""
    modules = odict()
    for info in infos:
        for field in kinds:
            if field not in modules:
                modules[field] = []
            modules[field] += [a for a in (info.get(field) or [])
                               if a not in modules[field]]
    # pop empty blocks
    for k, v in list(modules.items()):
        if not v:
            modules.pop(k)
    return modules


def get_default_info(module_or_class, kind=None, fail_if_not_found=False,
                     return_yaml=False, yaml_expand_defaults=True, module_path=None):
    """
    Get default info for a module_or_class.
    """
    _kind = kind or get_kind(module_or_class)
    try:
        cls = get_class(module_or_class, _kind, None_if_not_found=not fail_if_not_found,
                        module_path=module_path)
        if cls:
            default_module_info = cls.get_defaults(return_yaml=return_yaml,
                                                   yaml_expand_defaults=yaml_expand_defaults)
        else:
            default_module_info = (
                lambda x: yaml_dump(x) if return_yaml else x)(
                {_kind: {module_or_class: {}}})
    except Exception as e:
        raise LoggedError(log, "Failed to get defaults for module or class '%s' [%s]",
                          ("%s:" % _kind if _kind else "") + module_or_class, e)

    # TODO: find batter place to put this definition where it doesn't give circular ref
    # TODO: and more generally should not be importing defaults more than once
    from cobaya.likelihood import Likelihood
    from cobaya.theory import Theory
    from cobaya.sampler import Sampler
    component_base_classes = {kinds.sampler: Sampler, kinds.theory: Theory,
                              kinds.likelihood: Likelihood}

    if cls:
        for kind_name, _cls in component_base_classes.items():
            if issubclass(cls, _cls):
                if kind and kind != kind_name:
                    raise LoggedError(log, "class %s of unexpected kind %s", cls.__name__,
                                      kind)
                _kind = kind_name
                break

    if not return_yaml:
        try:
            info = default_module_info[_kind]
            if _self_name not in info:
                info[module_or_class]
            else:
                info[module_or_class] = info.pop(_self_name)
        except KeyError:
            raise LoggedError(log,
                              "The defaults file for '%s' should be structured as "
                              "%s:%s:{[options]} or %s:%s:{[options]}.", module_or_class,
                              _kind, module_or_class, _kind, _self_name)
        info = deepcopy((cls or component_base_classes[_kind]).class_options)
        info.update(default_module_info[_kind][module_or_class])
        default_module_info[_kind][module_or_class] = info
    elif _self_name in default_module_info:
        # replace __self__ in yaml with explicit name
        if len(default_module_info.split(_self_name)) == 1:
            default_module_info = default_module_info.replace(_self_name, module_or_class)
        else:
            raise LoggedError(log, "%s has more than one __self__" % module_or_class)

    return default_module_info


def update_info(info):
    """
    Creates an updated info starting from the defaults for each module and updating it
    with the input info.
    """
    # Don't modify the original input!
    input_info = deepcopy_where_possible(info)
    # Creates an equivalent info using only the defaults
    updated_info = odict()
    default_params_info = odict()
    default_prior_info = odict()
    modules = get_used_modules(input_info)
    for block in modules:
        updated = odict()
        updated_info[block] = updated
        for module in modules[block]:
            # Preprocess "no options" and "external function" in input
            try:
                input_info[block][module] = input_info[block][module] or {}
            except TypeError:
                raise LoggedError(
                    log, "Your input info is not well formatted at the '%s' block. "
                         "It must be a dictionary {'%s':{options}, ...}. ", block, block)
            if not hasattr(input_info[block][module], "get"):
                input_info[block][module] = {_external: input_info[block][module]}

            module_path = input_info[block][module].get(_module_path, None)
            default_class_info = get_default_info(module, block, module_path=module_path)
            # TODO: check - get_default_info was ignoring this extra arg:
            #  input_info[block][module_or_class])
            updated_info[block][module] = default_class_info[block][module] or {}
            # Update default options with input info
            # Consistency is checked only up to first level! (i.e. subkeys may not match)
            ignore = {_external, _p_renames, _input_params, _output_params, _module_path}
            options_not_recognized = (set(input_info[block][module])
                                      .difference(ignore)
                                      .difference(set(updated_info[block][module])))
            if options_not_recognized:
                alternatives = odict()
                available = ({_external, _p_renames}.union(updated_info[block][module]))
                while options_not_recognized:
                    option = options_not_recognized.pop()
                    alternatives[option] = fuzzy_match(option, available, n=3)
                did_you_mean = ", ".join(
                    [("'%s' (did you mean %s?)" % (o, "|".join(["'%s'" % _ for _ in a]))
                      if a else "'%s'" % o)
                     for o, a in alternatives.items()])
                if default_class_info[block][module]:
                    # Internal module
                    raise LoggedError(
                        log, "'%s' does not recognize some options: %s. "
                             "To see the allowed options, check out the documentation of"
                             " this module.", module, did_you_mean)
                else:
                    # External module
                    raise LoggedError(
                        log, "External %s '%s' does not recognize some options: %s. "
                             "Check the documentation for 'external %s'.",
                        block, module, did_you_mean, block)
            updated_info[block][module].update(input_info[block][module])
            # Store default parameters and priors of class, and save to combine later
            if block == kinds.likelihood:
                params_info = default_class_info.get(_params, {})
                updated_info[block][module].update({_params: list(params_info or [])})
                default_params_info[module] = params_info
                default_prior_info[module] = default_class_info.get(_prior, {})
    # Add priors info, after the necessary checks
    if _prior in input_info or any(default_prior_info.values()):
        updated_info[_prior] = input_info.get(_prior, odict())
    for prior_info in default_prior_info.values():
        for name, prior in prior_info.items():
            if updated_info[_prior].get(name, prior) != prior:
                raise LoggedError(
                    log, "Two different priors cannot have the same name: '%s'.", name)
            updated_info[_prior][name] = prior
    # Add parameters info, after the necessary updates and checks
    defaults_merged = merge_default_params_info(default_params_info)
    updated_info[_params] = merge_params_info(
        defaults_merged, input_info.get(_params, {}))
    # Add aliases for theory params (after merging!)
    if kinds.theory in updated_info:
        renames = list(updated_info[kinds.theory].values())[0].get(_p_renames)
        str_to_list = lambda x: ([x] if isinstance(x, string_types) else x)
        renames_flat = [set([k] + str_to_list(v)) for k, v in (renames or {}).items()]
        for p in updated_info.get(_params, {}):
            # Probably could be made faster by inverting the renames dicts *just once*
            renames_pairs = [a for a in renames_flat if p in a]
            if renames_pairs:
                this_renames = reduce(
                    lambda x, y: x.union(y), [a for a in renames_flat if p in a])
                updated_info[_params][p][_p_renames] = list(
                    set(this_renames).union(set(
                        str_to_list(updated_info[_params][p].get(_p_renames, []))))
                        .difference({p}))
    # Rest of the options
    for k, v in input_info.items():
        if k not in updated_info:
            updated_info[k] = v
    return updated_info


def merge_default_params_info(defaults):
    """
    Merges default parameters info for all likelihoods.
    Checks that multiple defined (=shared) parameters have equal info.
    """
    defaults_merged = odict()
    for lik, params in defaults.items():
        for p, info in (params or {}).items():
            # if already there, check consistency
            if p in defaults_merged:
                log.debug("Parameter '%s' multiply defined.", p)
                if info != defaults_merged[p]:
                    raise LoggedError(
                        log, "Parameter '%s' multiply defined, but inconsistent info: "
                             "For likelihood '%s' is '%r', but for some other likelihood"
                             " it was '%r'. Check your defaults!",
                        p, lik, info, defaults_merged[p])
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
    assert len(infos)
    previous_info = deepcopy(infos[0])
    if len(infos) == 1:
        return previous_info
    for new_info in infos[1:]:
        previous_params_info = deepcopy(previous_info.pop(_params, odict()) or odict())
        new_params_info = deepcopy(new_info).pop(_params, odict()) or odict()
        # NS: params have been popped, since they have their own merge function
        current_info = recursive_update(deepcopy(previous_info), new_info)
        current_info[_params] = merge_params_info(previous_params_info, new_params_info)
        previous_info = current_info
    return current_info


def is_equal_info(info1, info2, strict=True, print_not_log=False, ignore_blocks=[]):
    """
    Compares two information dictionaries.

    Set ``strict=False`` (default: ``True``) to ignore options that would not affect
    the statistics of a posterior sample, including order of params/priors/likelihoods.
    """
    if print_not_log:
        myprint = print
        myprint_debug = lambda x: x
    else:
        myprint = log.info
        myprint_debug = log.debug
    myname = inspect.stack()[0][3]
    ignore = set([]) if strict else {_debug, _debug_file, _resume, _force, _path_install}
    ignore = ignore.union(set(ignore_blocks or []))
    ignore_params = (set([]) if strict else {_p_label, _p_renames, _p_ref, _p_proposal,
                                             "min", "max"})
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
        if block_name in [kinds.sampler, kinds.theory]:
            # Internal order does NOT matter
            if set(block1) != set(block2):
                myprint(myname + ": different [%s]" % block_name)
                return False
            # Anything to ignore?
            for k in block1:
                module_folder = get_class_module(k, block_name)
                try:
                    ignore_k = getattr(import_module(
                        module_folder, package=_package), "ignore_at_resume", {})
                except ImportError:
                    ignore_k = {}
                if block_name == kinds.theory:
                    ignore_k.update({_input_params: None, _output_params: None})
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
                    myprint_debug("%r (old) vs %r (new)" % (
                        recursive_odict_to_dict(block1k),
                        recursive_odict_to_dict(block2k)))
                    return False
        elif block_name in [_params, kinds.likelihood, _prior]:
            # Internal order DOES matter, but just up to 1st level
            f = list if strict else set
            if f(block1) != f(block2):
                myprint(
                    myname + ": different [%s] or different order of them: %r vs %r" % (
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
                            block1k.pop(kignore, None)
                            block2k.pop(kignore, None)
                        # Fixed params, it doesn't matter if they are saved as derived
                        for b in [block1k, block2k]:
                            if _p_value in b:
                                b.pop(_p_derived, None)
                if block_name == kinds.likelihood and not strict:
                    for kignore in [_input_params, _output_params]:
                        block1k.pop(kignore, None)
                        block2k.pop(kignore, None)
                if recursive_odict_to_dict(block1k) != recursive_odict_to_dict(block2k):
                    myprint(myname + ": different content of [%s:%s]" % (
                        block_name, k))
                    return False
    return True


class HasDefaults(object):

    @classmethod
    def get_qualified_names(cls):
        if cls.__module__ == '__main__':
            return [cls.__name__]
        parts = cls.__module__.split('.')
        if len(parts) > 1:
            # get shortest reference
            try:
                imported = import_module(".".join(parts[:-1]))
            except:
                pass
            else:
                if getattr(imported, cls.__name__, None) is cls:
                    parts = parts[:-1]
        if parts[-1] == cls.__name__:
            return ['.'.join(parts[i:]) for i in range(len(parts))]
        else:
            return ['.'.join(parts[i:]) + '.' + cls.__name__ for i in
                    range(len(parts) + 1)]

    @classmethod
    def get_qualified_class_name(cls):
        """
        Get the distinct shortest reference name for the class of the form
        module.ClassName or module.submodule.ClassName etc.
        For Cobaya modules the name is relative to subpackage for the relevant kind of
        class (e.g. Likelihood names are relative to cobaya.likelihoods).

        For external classes it loads the shortest fully qualified name of the form
        package.ClassName or package.module.ClassName or
        package.subpackage.module.ClassName, etc.
        """
        qualified_names = cls.get_qualified_names()
        if qualified_names[0].startswith('cobaya.'):
            return qualified_names[2]
        else:
            # external
            return qualified_names[0]

    @classmethod
    def get_class_path(cls):
        return os.path.abspath(os.path.dirname(inspect.getfile(cls)))

    @classmethod
    def get_root_file_name(cls):
        return os.path.join(cls.get_class_path(), cls.__name__)

    @classmethod
    def get_yaml_file(cls):
        filename = cls.get_root_file_name() + ".yaml"
        if os.path.exists(filename):
            return filename
        return None

    @classmethod
    def get_bibtex(cls):
        bib = cls.get_associated_file_content('.bibtex')
        if bib:
            return bib.decode('utf-8')
        for base in cls.__bases__:
            if issubclass(base, HasDefaults) and base is not HasDefaults:
                bib = base.get_bibtex()
                if bib:
                    return bib
        return None

    @classmethod
    def get_associated_file_content(cls, ext):
        # handle extracting package files when may be inside a zipped package so files
        # not accessible directly
        try:
            return pkg_resources.resource_string(cls.__module__, cls.__name__ + ext)
        except Exception as e:
            return None

    @classmethod
    def get_defaults(cls, return_yaml=False, yaml_expand_defaults=True):
        """
        Return defaults for this module_or_class, with syntax:

        .. code::
           [kind]
             [module_name]:
               option: value
               [...]

           params:
             [...]  # if required

           prior:
             [...]  # if required

        If keyword `return_yaml` is set to True, it returns literally that,
        whereas if False (default), it returns the corresponding Python dict.

        Note that in external modules installed as zip_safe=True packages files cannot be
        accessed directly.
        In this case using !default .yaml includes currently does not work.

        Also note that if you return a dictionary it may be modified (return a deep copy
        if you want to keep it).
        """
        yaml_text = cls.get_associated_file_content('.yaml')
        if not yaml_text:
            for base in cls.__bases__:
                if issubclass(base, HasDefaults) and base is not HasDefaults:
                    return base.get_defaults(return_yaml=return_yaml,
                                             yaml_expand_defaults=yaml_expand_defaults)
        if return_yaml:
            if yaml_expand_defaults:
                return yaml_dump(yaml_load_file(cls.get_yaml_file(), yaml_text))
            else:
                return yaml_text
        else:
            return yaml_load_file(cls.get_yaml_file(), yaml_text)
