"""
.. module:: input

:Synopsis: Input-related functions
:Author: Jesus Torrado

"""

# Global
import os
import inspect
import logging
from copy import deepcopy
from importlib import import_module
from itertools import chain
from functools import reduce
from typing import Mapping
from collections import defaultdict
import pkg_resources

# Local
from cobaya.conventions import _products_path, _packages_path, _resume, _force, _params, \
    partag, _external, _output_prefix, _debug, _debug_file, _auto_params, _prior, \
    kinds, _provides, _requires, _input_params, _output_params, _component_path, \
    _aliases, _yaml_extensions, reserved_attributes, empty_dict, _get_chi2_name, \
    _get_chi2_label, _test_run, _version
from cobaya.tools import recursive_update, str_to_list, get_base_classes, \
    fuzzy_match, deepcopy_where_possible, get_class, get_kind
from cobaya.yaml import yaml_load_file, yaml_dump
from cobaya.log import LoggedError
from cobaya.parameterization import expand_info_param
from cobaya.mpi import share_mpi, is_main_process

# Logger
log = logging.getLogger(__name__.split(".")[-1])


def load_input(input_file):
    """
    Loads general info, and splits it into the right parts.
    """
    file_name, extension = os.path.splitext(input_file)
    file_name = os.path.basename(file_name)
    if extension.lower() not in _yaml_extensions:
        raise LoggedError(log, "Extension of input file '%s' not recognized.", input_file)
    info = yaml_load_file(input_file) or {}
    # if output_prefix not defined, default to input_file name (sans ext.) as prefix;
    if _output_prefix not in info:
        info[_output_prefix] = file_name
    # warn if no output, since we are in shell-invocation mode.
    elif info[_output_prefix] is None:
        log.warning("WARNING: Output explicitly suppressed with '%s: null'",
                    _output_prefix)
    # contained? Ensure that output is sent where it should
    if "CONTAINED" in os.environ:
        for out in [_output_prefix, _debug_file]:
            if info.get(out):
                if not info[out].startswith("/"):
                    info[out] = os.path.join(_products_path, info[out])
    return info


# MPI wrapper for loading the input info
def load_input_MPI(input_file):
    return share_mpi(load_input(input_file) if is_main_process() else None)


def get_used_components(*infos):
    """Returns all requested components as an dict ``{kind: set([components])}``.
    Priors are not included."""
    components = defaultdict(list)
    for info in infos:
        for field in kinds:
            try:
                components[field] += [a for a in (info.get(field) or [])
                                      if a not in components[field]]
            except TypeError:
                raise LoggedError(
                    log, "Your input info is not well formatted at the '%s' block. "
                         "It must be a dictionary {'%s_i':{options}, ...}. ",
                    field, field)
    # return dictionary of non-empty blocks
    return {k: v for k, v in components.items() if v}


def get_default_info(component_or_class, kind=None, return_yaml=False,
                     yaml_expand_defaults=True, component_path=None,
                     input_options=empty_dict):
    """
    Get default info for a component_or_class.
    """
    _kind = kind
    try:
        if inspect.isclass(component_or_class):
            cls = component_or_class
        else:
            _kind = _kind or get_kind(component_or_class)
            cls = get_class(component_or_class, _kind, component_path=component_path)
        default_component_info = \
            cls.get_defaults(return_yaml=return_yaml,
                             yaml_expand_defaults=yaml_expand_defaults,
                             input_options=input_options)
    except Exception as e:
        raise LoggedError(log, "Failed to get defaults for component or class '%s' [%s]",
                          component_or_class, e)
    return default_component_info


def update_info(info):
    """
    Creates an updated info starting from the defaults for each component and updating it
    with the input info.
    """
    component_base_classes = get_base_classes()
    # Don't modify the original input, and convert all Mapping to consistent dict
    input_info = deepcopy_where_possible(info)
    # Creates an equivalent info using only the defaults
    updated_info = {}
    default_params_info = {}
    default_prior_info = {}
    components = get_used_components(input_info)
    from cobaya.component import CobayaComponent
    for block in components:
        updated = {}
        updated_info[block] = updated
        input_block = input_info[block]
        for component in components[block]:
            # Preprocess "no options" and "external function" in input
            try:
                input_block[component] = input_block[component] or {}
            except TypeError:
                raise LoggedError(
                    log, "Your input info is not well formatted at the '%s' block. "
                         "It must be a dictionary {'%s_i':{options}, ...}. ",
                    block, block)
            if isinstance(component, CobayaComponent) or \
                    isinstance(input_block[component], CobayaComponent):
                raise LoggedError(log, "Input for %s:%s should specify a class not "
                                       "an instance", block, component)
                # TODO: allow instance passing?
                #       could allow this, but would have to sort out deepcopy
                # if input_block[component]:
                #   raise LoggedError(log, "Instances should be passed a dictionary "
                #                           "entry of the form 'instance: None'")
                # change_key(input_block, component, component.get_name(),
                #           {_external: component})
                # updated[component.get_name()] = input_block[component.get_name()].copy()
                # continue
            if inspect.isclass(input_block[component]) or \
                    not isinstance(input_block[component], dict):
                input_block[component] = {_external: input_block[component]}
            ext = input_block[component].get(_external)
            if ext:
                if inspect.isclass(ext):
                    default_class_info = get_default_info(ext, block,
                                                          input_options=input_block[
                                                              component])
                else:
                    default_class_info = deepcopy_where_possible(
                        component_base_classes[block].get_defaults())
            else:
                component_path = input_block[component].get(_component_path, None)
                default_class_info = get_default_info(
                    component, block,
                    component_path=component_path, input_options=input_block[component])
            updated[component] = default_class_info or {}
            # Update default options with input info
            # Consistency is checked only up to first level! (i.e. subkeys may not match)
            ignore = {_external, _provides, _requires, partag.renames, _input_params,
                      _output_params, _component_path, _aliases}
            options_not_recognized = (set(input_block[component])
                                      .difference(ignore)
                                      .difference(set(updated[component])))
            if options_not_recognized:
                alternatives = {}
                available = (
                    {_external, _requires, partag.renames}.union(
                        updated_info[block][component]))
                while options_not_recognized:
                    option = options_not_recognized.pop()
                    alternatives[option] = fuzzy_match(option, available, n=3)
                did_you_mean = ", ".join(
                    [("'%s' (did you mean %s?)" % (o, "|".join(["'%s'" % _ for _ in a]))
                      if a else "'%s'" % o)
                     for o, a in alternatives.items()])
                raise LoggedError(
                    log, "%s '%s' does not recognize some options: %s. "
                         "Check the documentation for '%s'.",
                    block, component, did_you_mean, block)
            updated[component].update(input_block[component])
            # save params and priors of class to combine later
            default_params_info[component] = default_class_info.get(_params, {})
            default_prior_info[component] = default_class_info.get(_prior, {})
    # Add priors info, after the necessary checks
    if _prior in input_info or any(default_prior_info.values()):
        updated_info[_prior] = input_info.get(_prior, {})
    for prior_info in default_prior_info.values():
        for name, prior in prior_info.items():
            if updated_info[_prior].get(name, prior) != prior:
                raise LoggedError(
                    log, "Two different priors cannot have the same name: '%s'.", name)
            updated_info[_prior][name] = prior
    # Add parameters info, after the necessary updates and checks
    defaults_merged = merge_default_params_info(default_params_info)
    updated_info[_params] = merge_params_info([defaults_merged,
                                               input_info.get(_params, {})],
                                              default_derived=False)
    # Add aggregated chi2 params
    if kinds.likelihood in info:
        all_types = set(chain(
            *[str_to_list(like_info.get("type", []) or [])
              for like_info in updated_info[kinds.likelihood].values()]))
        for t in all_types:
            updated_info[_params][_get_chi2_name(t)] = {
                partag.latex: _get_chi2_label(t), partag.derived: True}
    # Add automatically-defined parameters
    if _auto_params in updated_info:
        make_auto_params(updated_info.pop(_auto_params), updated_info[_params])
    # Add aliases for theory params (after merging!)
    for kind in [k for k in [kinds.theory, kinds.likelihood] if k in updated_info]:
        for item in updated_info[kind].values():
            renames = item.get(partag.renames)
            if renames:
                if not isinstance(renames, Mapping):
                    raise LoggedError(log,
                                      "'renames' should be a dictionary of name mappings "
                                      "(or you meant to use 'aliases')")
                renames_flat = [set([k] + str_to_list(v)) for k, v in renames.items()]
                for p in updated_info[_params]:
                    # Probably could be made faster by inverting the renames dicts *once*
                    renames_pairs = [a for a in renames_flat if p in a]
                    if renames_pairs:
                        this_renames = reduce(
                            lambda x, y: x.union(y), [a for a in renames_flat if p in a])
                        updated_info[_params][p][partag.renames] = \
                            list(set(this_renames).union(set(str_to_list(
                                updated_info[_params][p].get(partag.renames, []))))
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
    defaults_merged = {}
    for lik, params in defaults.items():
        for p, info in (params or {}).items():
            # if already there, check consistency
            if p in defaults_merged:
                if info != defaults_merged[p]:
                    raise LoggedError(
                        log, "Parameter '%s' multiply defined, but inconsistent info: "
                             "For likelihood '%s' is '%r', but for some other likelihood"
                             " it was '%r'. Check your defaults!",
                        p, lik, info, defaults_merged[p])
                log.debug("Parameter '%s' is multiply defined but consistent.", p)
            defaults_merged[p] = info
    return defaults_merged


def merge_params_info(params_infos, default_derived=True):
    """
    Merges parameter infos, starting from the first one
    and updating with each additional one.
    Labels (for sampled and derived) and min/max
    (just for derived params) are inherited from defaults
    (but not if one of min/max is re-defined: in that case,
    to avoid surprises, the other one is set to None=+/-inf)
    """
    current_info = {p: expand_info_param(v, default_derived) for p, v in
                    params_infos[0].items() or {}}
    for new_info in params_infos[1:]:
        if not new_info:
            continue
        for p, new_info_p in new_info.items():
            if p not in current_info:
                current_info[p] = {}
            new_info_p = expand_info_param(new_info_p)
            current_info[p].update(deepcopy(new_info_p))
            # Account for incompatibilities: "prior" and ("value" or "derived"+bounds)
            incompatibilities = {_prior: [partag.value, partag.derived, "min", "max"],
                                 partag.value: [partag.prior, partag.ref,
                                                partag.proposal],
                                 partag.derived: [partag.prior, partag.drop, partag.ref,
                                                  partag.proposal]}
            for f1, incomp in incompatibilities.items():
                if f1 in new_info_p:
                    for f2 in incomp:
                        current_info[p].pop(f2, None)
    # Re-sort, so that rightmost info takes precedence *also* in the sorting
    new_order = chain(*[list(params) for params in params_infos[::-1]])
    # The following removes duplicates maintaining order (keeps the first occurrence)
    new_order = list(dict.fromkeys(new_order))
    current_info = {p: current_info[p] for p in new_order}
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
        previous_params_info = deepcopy(previous_info.pop(_params, {}) or {})
        new_params_info = deepcopy(new_info).pop(_params, {}) or {}
        # NS: params have been popped, since they have their own merge function
        current_info = recursive_update(deepcopy(previous_info), new_info)
        current_info[_params] = merge_params_info([previous_params_info, new_params_info])
        previous_info = current_info
    return current_info


def is_equal_info(info_old, info_new, strict=True, print_not_log=False, ignore_blocks=()):
    """
    Compares two information dictionaries, and old one versus a new one, and updates the
    new one for selected values of the old one.

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
    ignore = set() if strict else \
        {_debug, _debug_file, _resume, _force, _packages_path, _test_run, _version}
    ignore = ignore.union(set(ignore_blocks or []))
    if set(info for info in info_old if info_old[info] is not None).difference(ignore) \
            != set(info for info in info_new if info_new[info] is not None).difference(
        ignore):
        myprint(myname + ": different blocks or options: %r (old) vs %r (new)" % (
            set(info_old).difference(ignore), set(info_new).difference(ignore)))
        return False
    for block_name in info_old:
        if block_name in ignore or block_name not in info_new:
            continue
        block1 = deepcopy_where_possible(info_old[block_name])
        block2 = deepcopy_where_possible(info_new[block_name])
        # First, deal with root-level options (force, output, ...)
        if not isinstance(block1, dict):
            if block1 != block2:
                myprint(myname + ": different option '%s'" % block_name)
                return False
            continue
        # Now let's do components and params
        # 1. check order (it DOES matter, but just up to 1st level)
        f = list if strict else set
        if f(block1) != f(block2):
            myprint(
                myname + ": different [%s] or different order of them: %r vs %r" % (
                    block_name, list(block1), list(block2)))
            return False
        # 2. Gather general options to be ignored
        if not strict:
            ignore_k = set()
            if block_name in [kinds.theory, kinds.likelihood]:
                ignore_k = ignore_k.union({_input_params, _output_params})
            elif block_name == _params:
                for param in block1:
                    # Unify notation
                    block1[param] = expand_info_param(block1[param])
                    block2[param] = expand_info_param(block2[param])
                    ignore_k = ignore_k.union({partag.latex, partag.renames, partag.ref,
                                               partag.proposal, "min", "max"})
                    # Fixed params, it doesn't matter if they are saved as derived
                    if partag.value in block1[param]:
                        block1[param].pop(partag.derived, None)
                    if partag.value in block2[param]:
                        block2[param].pop(partag.derived, None)
                    # Renames: order does not matter
                    block1[param][partag.renames] = set(
                        block1[param].get(partag.renames, []))
                    block2[param][partag.renames] = set(
                        block2[param].get(partag.renames, []))
        # 3. Now check component/parameters one-by-one
        for k in block1:
            if not strict:
                # Add component-specific options to be ignored
                if block_name in kinds:
                    ignore_k_this = ignore_k.copy()
                    if _external not in block1[k]:
                        try:
                            component_path = block1[k].pop(_component_path, None) \
                                if isinstance(block1[k], dict) else None
                            cls = get_class(k, block_name, component_path=component_path)
                            ignore_k_this = ignore_k_this.union(
                                set(getattr(cls, "_at_resume_prefer_new", {})))
                        except ImportError:
                            pass
                    # Pop ignored and kept options
                    for j in list(ignore_k_this):
                        block1[k].pop(j, None)
                        block2[k].pop(j, None)
            if block1[k] != block2[k]:
                # For clarity, pop common stuff before printing
                to_pop = [j for j in block1[k] if (block1[k].get(j) == block2[k].get(j))]
                [(block1[k].pop(j, None), block2[k].pop(j, None)) for j in to_pop]
                myprint(
                    myname + ": different content of [%s:%s]" % (block_name, k) +
                    " -- (re-run with `debug: True` for more info)")
                myprint_debug("%r (old) vs %r (new)" % (block1[k], block2[k]))
                return False
    return True


def get_preferred_old_values(info_old):
    """
    Returns selected values in `info_old`, which are preferred at resuming.
    """
    keep_old = {}
    for block_name, block in info_old.items():
        if block_name not in kinds or not block:
            continue
        for k in block:
            try:
                component_path = block[k].pop(_component_path, None) \
                    if isinstance(block[k], dict) else None
                cls = get_class(k, block_name, component_path=component_path)
                prefer_old_k_this = getattr(cls, "_at_resume_prefer_old", {})
                if prefer_old_k_this:
                    if block_name not in keep_old:
                        keep_old[block_name] = {}
                    keep_old[block_name].update(
                        {k: {o: block[k][o] for o in prefer_old_k_this if o in block[k]}})
            except ImportError:
                pass
    return keep_old


class HasDefaults:
    """
    Base class for components that can read settings from a .yaml file.
    Class methods provide the methods needed to get the defaults information
    and associated data.

    """

    @classmethod
    def get_qualified_names(cls):
        if cls.__module__ == '__main__':
            return [cls.__name__]
        parts = cls.__module__.split('.')
        if len(parts) > 1:
            # get shortest reference
            try:
                imported = import_module(".".join(parts[:-1]))
            except ImportError:
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
        For Cobaya components the name is relative to subpackage for the relevant kind of
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
        """
        Get the file path for the class.
        """
        return os.path.abspath(os.path.dirname(inspect.getfile(cls)))

    @classmethod
    def get_root_file_name(cls):
        return os.path.join(cls.get_class_path(), cls.__name__)

    @classmethod
    def get_yaml_file(cls):
        """
        Gets the file name of the .yaml file for this component if it exists on file
        (otherwise None).
        """
        filename = cls.get_root_file_name() + ".yaml"
        if os.path.exists(filename):
            return filename
        return None

    @classmethod
    def get_bibtex(cls):
        """
        Get the content of .bibtex file for this component. If no specific bibtex
        from this class, it will return the result from an inherited class if that
        provides bibtex.
        """
        bib = cls.get_associated_file_content('.bibtex')
        if bib:
            try:
                return bib.decode("utf-8")
            except:
                return bib
        for base in cls.__bases__:
            if issubclass(base, HasDefaults) and base is not HasDefaults:
                bib = base.get_bibtex()
                if bib:
                    try:
                        return bib.decode("utf-8")
                    except:
                        return bib
        return None

    @classmethod
    def get_associated_file_content(cls, ext):
        # handle extracting package files when may be inside a zipped package so files
        # not accessible directly
        try:
            string = pkg_resources.resource_string(cls.__module__, cls.__name__ + ext)
            try:
                return string.decode("utf-8")
            except:
                return string
        except Exception:
            return None

    @classmethod
    def get_class_options(cls, input_options=empty_dict):
        """
        Returns dictionary of names and values for class variables that can also be
        input and output in yaml files, by default it takes all the
        (non-inherited and non-private)  attributes of the class excluding known
        specials.

        Could be overridden using input_options to dynamically generate defaults,
        e.g. a set of input parameters generated depending on the input_options.

        :param input_options: optional dictionary of input parameters
        :return:  dict of names and values
        """
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('_') and
                k not in reserved_attributes and not inspect.isroutine(v)
                and not isinstance(v, property)}

    @classmethod
    def get_defaults(cls, return_yaml=False, yaml_expand_defaults=True,
                     input_options=empty_dict):
        """
        Return defaults for this component_or_class, with syntax:

        .. code::

           option: value
           [...]

           params:
             [...]  # if required

           prior:
             [...]  # if required

        If keyword `return_yaml` is set to True, it returns literally that,
        whereas if False (default), it returns the corresponding Python dict.

        Note that in external components installed as zip_safe=True packages files cannot
        be accessed directly.
        In this case using !default .yaml includes currently does not work.

        Also note that if you return a dictionary it may be modified (return a deep copy
        if you want to keep it).

        if yaml_expand_defaults then !default: file includes will be expanded

        input_options may be a dictionary of input options, e.g. in case default params
        are dynamically dependent on an input variable
        """
        if 'class_options' in cls.__dict__:
            raise LoggedError(log, "class_options (in %s) should now be replaced by "
                                   "public attributes defined directly in the class" %
                              cls.get_qualified_class_name())
        yaml_text = cls.get_associated_file_content('.yaml')
        options = cls.get_class_options(input_options=input_options)
        if options and yaml_text:
            raise LoggedError(log,
                              "%s: any class can either have .yaml or class variables "
                              "but not both (type declarations without values are fine "
                              "also with yaml file). You have class attributes: %s",
                              cls.get_qualified_class_name(), list(options))
        if return_yaml and not yaml_expand_defaults:
            return yaml_text or ""
        this_defaults = yaml_load_file(cls.get_yaml_file(), yaml_text) \
            if yaml_text else deepcopy_where_possible(options)
        # start with this one to keep the order such that most recent class options
        # near the top. Update below to actually override parameters with these.
        defaults = this_defaults.copy()
        if not return_yaml:
            for base in cls.__bases__:
                if issubclass(base, HasDefaults) and base is not HasDefaults:
                    defaults.update(base.get_defaults(input_options=input_options))
        defaults.update(this_defaults)
        if return_yaml:
            return yaml_dump(defaults)
        else:
            return defaults


def make_auto_params(auto_params, params_info):
    def replace(item, tag):
        if isinstance(item, dict):
            for key, val in list(item.items()):
                item[key] = replace(val, tag)
        elif isinstance(item, str) and '%s' in item:
            item = item % tag
        return item

    for k, v in auto_params.items():
        if '%s' not in k:
            raise LoggedError(log, 'auto_param parameter names must have %s placeholder')
        replacements = v.pop('auto_range')
        if isinstance(replacements, str):
            replacements = eval(replacements)
        for value in replacements:
            params_info[k % value] = replace(deepcopy_where_possible(v), value)
