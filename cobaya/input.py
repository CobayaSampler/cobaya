"""
.. module:: input

:Synopsis: Input-related functions
:Author: Jesus Torrado and Antony Lewis

"""

import inspect
import os
import platform
from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from itertools import chain
from typing import TypeVar

import cobaya.typing
from cobaya import mpi
from cobaya.component import ComponentNotFoundError, get_component_class
from cobaya.conventions import (
    Extension,
    FileSuffix,
    get_chi2_label,
    get_chi2_name,
    kinds,
    packages_path_input,
    products_path,
    separator_files,
)
from cobaya.log import LoggedError, get_logger
from cobaya.parameterization import expand_info_param
from cobaya.tools import (
    deepcopy_where_possible,
    fuzzy_match,
    get_base_classes,
    recursive_update,
    str_to_list,
)
from cobaya.typing import (
    ExpandedParamsDict,
    InfoDict,
    InputDict,
    LikesDict,
    ModelDict,
    empty_dict,
)
from cobaya.yaml import yaml_load, yaml_load_file

# Logger
logger = get_logger(__name__)


def load_input_dict(info_or_yaml_or_file: InputDict | str | os.PathLike) -> InputDict:
    if isinstance(info_or_yaml_or_file, os.PathLike):
        return load_input_file(info_or_yaml_or_file)
    elif isinstance(info_or_yaml_or_file, str):
        if "\n" in info_or_yaml_or_file:
            return yaml_load(info_or_yaml_or_file)  # type: ignore
        else:
            return load_input_file(info_or_yaml_or_file)
    elif isinstance(info_or_yaml_or_file, (dict, Mapping)):
        return deepcopy_where_possible(info_or_yaml_or_file)
    else:
        raise ValueError(
            "The first argument must be a dictionary, file name or "
            "yaml string with the required input options."
        )


def load_input_file(
    input_file: str | os.PathLike, no_mpi: bool = False, help_commands: str | None = None
) -> InputDict:
    if no_mpi:
        mpi.set_mpi_disabled()
    input_file = str(input_file)
    stem, suffix = os.path.splitext(input_file)
    if os.path.basename(stem) in ("input", "updated"):
        raise ValueError(
            "'input' and 'updated' are reserved file names. Please, use a different one."
        )
    if suffix.lower() in Extension.yamls + (Extension.dill,):
        info = load_input_MPI(input_file)
        root, suffix = os.path.splitext(stem)
        if suffix == ".updated":
            # path may have been removed, so put in full path and name
            info["output"] = root
    else:
        # Passed an existing output_prefix?
        # First see if there is a binary info pickle
        updated_file = get_info_path(*split_prefix(input_file), ext=Extension.dill)
        if not os.path.exists(updated_file):
            # Try to find the corresponding *.updated.yaml
            updated_file = get_info_path(*split_prefix(input_file))
        try:
            info = load_input_MPI(updated_file)
        except OSError:
            err_msg = "Not a valid input file, or non-existent run to resume."
            if help_commands:
                err_msg += (
                    f" Maybe you mistyped one of the following commands: {help_commands}"
                )
            raise ValueError(err_msg)
        # We need to update the output_prefix to resume the run *where it is*
        info["output"] = input_file
        if "post" not in info:
            # If input given this way, we obviously want to resume!
            info["resume"] = True
    return info


def load_input(input_file: str) -> InputDict:
    """
    Loads general info, and splits it into the right parts.
    """
    file_name, extension = os.path.splitext(input_file)
    file_name = os.path.basename(file_name)
    info: InputDict
    if extension.lower() in Extension.yamls:
        info = yaml_load_file(input_file) or {}  # type: ignore
    elif extension == Extension.dill:
        info = load_info_dump(input_file) or {}
    else:
        raise LoggedError(
            logger, "Extension of input file '%s' not recognized.", input_file
        )
    # if output_prefix not defined, default to input_file name (sans ext.) as prefix;
    if "output" not in info:
        info["output"] = file_name
    # warn if no output, since we are in shell-invocation mode.
    elif info["output"] is None:
        logger.warning("WARNING: Output explicitly suppressed with '%s: null'", "output")
    # contained? Ensure that output is sent where it should
    if "CONTAINED" in os.environ:
        for out in ("output", "debug"):
            if isinstance(info.get(out), str):
                if not info[out].startswith("/"):
                    info[out] = os.path.join(products_path, info[out])
    return info


# separate MPI function, as sometimes just use load_input from root process only
@mpi.from_root
def load_input_MPI(input_file) -> InputDict:
    return load_input(input_file)


def load_info_overrides(*infos_or_yaml_or_files, **flags) -> InputDict:
    """
    Takes a number of input dictionaries (or paths to them), loads them and updates them,
    the latter ones taking precedence.

    If present, it updates the results with the kwargs if their value is not ``None``.

    Returns a deep copy of the resulting updated input dict (non-copyable object are
    retained).
    """
    info = load_input_dict(infos_or_yaml_or_files[0])  # makes deep copy if dict
    for another_info in infos_or_yaml_or_files[1:]:
        info = recursive_update(info, load_input_dict(another_info))
    for flag, value in flags.items():
        if value is not None:
            info[flag] = value
    if cobaya.typing.enforce_type_checking:
        cobaya.typing.validate_type(InputDict, info)
    return info


# load from dill pickle, including any lambda functions or external classes
def load_info_dump(input_file) -> InputDict:
    import dill

    with open(input_file, "rb") as f:
        return dill.load(f)


def split_prefix(prefix):
    """
    Splits an output prefix into folder and file name prefix.

    If on Windows, allows for unix-like input.
    """
    if platform.system() == "Windows":
        prefix = prefix.replace("/", os.sep)
    folder = os.path.dirname(prefix) or "."
    file_prefix = os.path.basename(prefix)
    if file_prefix == ".":
        file_prefix = ""
    return folder, file_prefix


def get_info_path(folder, prefix, infix=None, kind="updated", ext=Extension.yamls[0]):
    """
    Gets path to info files saved by Output.
    """
    if infix is None:
        infix = ""
    elif not infix.endswith("."):
        infix += "."
    info_file_prefix = os.path.join(folder, prefix + (separator_files if prefix else ""))
    try:
        suffix = {"input": FileSuffix.input, "updated": FileSuffix.updated}[kind.lower()]
    except KeyError:
        raise ValueError("`kind` must be `input|updated`")
    return info_file_prefix + infix + suffix + ext


def get_used_components(*infos, return_infos=False):
    """
    Returns all requested components as a dict ``{kind: set([components])}``.
    Priors are not included.

    The list of arguments may contain base strings, which are interpreted as component
    names and added to the returned dictionary under a ``None`` key. In this case, there
    is no guarantee that the same component will not be listed both under ``None`` and
    under its particular kind.

    If ``return_infos=True`` (default: ``False``), also returns a dictionary of inputs per
    component, updated in the order in which the info arguments are given.

    Components which are just renames of others (i.e. defined with `class_name`) return
    the original class' name.
    """
    # TODO: take inheritance into account
    comps: dict[str | None, list[str]] = defaultdict(list)
    comp_infos: dict[str, dict] = defaultdict(dict)
    for info in infos:
        if isinstance(info, str) and info not in comps[None]:
            comps[None] += [info]
            if return_infos and info not in comp_infos:
                comp_infos[info] = {}
            continue
        for kind in kinds:
            try:
                comps[kind] += [a for a in (info.get(kind) or []) if a not in comps[kind]]
            except TypeError:
                raise LoggedError(
                    logger,
                    (
                        "Your input info is not well formatted at the '%s' block. "
                        "It must be a dictionary {'%s_i':{options}, ...}. "
                    ),
                    kind,
                    kind,
                )
            if return_infos:
                for c in comps[kind]:
                    comp_infos[c].update(
                        (info[kind][c] or {}) if isinstance(info[kind], Mapping) else {}
                    )
    # return dictionary of non-empty blocks
    components = {k: v for k, v in comps.items() if v}
    return (components, dict(comp_infos)) if return_infos else components


def get_default_info(
    component_or_class,
    kind=None,
    return_yaml=False,
    yaml_expand_defaults=True,
    component_path=None,
    input_options=empty_dict,
    class_name=None,
    return_undefined_annotations=False,
):
    """
    Get default info for a component_or_class.
    """
    try:
        cls = get_component_class(
            component_or_class, kind, component_path, class_name, logger=logger
        )
        default_component_info = cls.get_defaults(
            return_yaml=return_yaml,
            yaml_expand_defaults=yaml_expand_defaults,
            input_options=input_options,
        )
        if not return_yaml:
            default_component_info = cls.get_modified_defaults(
                default_component_info, input_options=input_options
            )
    except ComponentNotFoundError:
        raise
    except Exception as e:
        raise LoggedError(
            logger,
            "Failed to get defaults for component or class '%s' [%s]",
            component_or_class,
            e,
        )
    if return_undefined_annotations:
        annotations = {
            k: v
            for k, v in cls.get_annotations().items()
            if k not in default_component_info
        }
        return default_component_info, annotations
    else:
        return default_component_info


def add_aggregated_chi2_params(param_info, all_types):
    for t in sorted(all_types):
        param_info[get_chi2_name(t)] = {"latex": get_chi2_label(t), "derived": True}


_Dict = TypeVar("_Dict", InputDict, ModelDict)


def update_info(info: _Dict, strict: bool = True, add_aggr_chi2: bool = True) -> _Dict:
    """
    Creates an updated info starting from the defaults for each component and updating it
    with the input info.
    """
    component_base_classes = get_base_classes()
    # Don't modify the original input, and convert all Mapping to consistent dict
    input_info = deepcopy_where_possible(info)
    # Creates an equivalent info using only the defaults
    updated_info: _Dict = {}  # type: ignore
    default_params_info = {}
    default_prior_info = {}
    from cobaya.component import CobayaComponent

    for block, block_info in get_used_components(input_info).items():
        updated: InfoDict = {}
        updated_info[block] = updated
        input_block = input_info[block]
        name: str
        for name in block_info:
            # Preprocess "no options" and "external function" in input
            try:
                input_block[name] = input_block[name] or {}
            except TypeError:
                raise LoggedError(
                    logger,
                    (
                        "Your input info is not well formatted at the '%s' block. "
                        "It must be a dictionary {'%s_i':{options}, ...}. "
                    ),
                    block,
                    block,
                )
            if isinstance(name, CobayaComponent) or isinstance(name, type):
                raise LoggedError(
                    logger,
                    (
                        "Instances and classes should be passed a "
                        "dictionary entry of the form 'name: instance'"
                    ),
                )
            if isinstance(input_block[name], CobayaComponent):
                logger.warning("Support for input instances is experimental")
            if isinstance(input_block[name], type) or not isinstance(
                input_block[name], dict
            ):
                input_block[name] = {"external": input_block[name]}

            annotations = {}
            if ext := input_block[name].get("external"):
                if isinstance(ext, type):
                    default_class_info, annotations = get_default_info(
                        ext,
                        block,
                        input_options=input_block[name],
                        return_undefined_annotations=True,
                    )
                else:
                    default_class_info = deepcopy_where_possible(
                        component_base_classes[block].get_defaults()
                    )
            else:
                component_path = input_block[name].get("python_path")
                try:
                    default_class_info, annotations = get_default_info(
                        name,
                        block,
                        class_name=input_block[name].get("class"),
                        component_path=component_path,
                        input_options=input_block[name],
                        return_undefined_annotations=True,
                    )
                except ComponentNotFoundError:
                    if strict:
                        raise
                    default_class_info, annotations = {}, {}
            updated[name] = default_class_info or {}
            # Update default options with input info
            # Consistency is checked only up to first level! (i.e. subkeys may not match)
            # Reserved attributes not necessarily already in default info:
            reserved = {
                "external",
                "class",
                "provides",
                "requires",
                "renames",
                "input_params",
                "output_params",
                "python_path",
                "aliases",
                "package_install",
            }

            if options_not_recognized := set(input_block[name]).difference(
                chain(reserved, updated[name], annotations)
            ):
                alternatives = {}
                available = {"external", "class", "requires", "renames"}.union(
                    updated_info[block][name]
                )
                while options_not_recognized:
                    option = options_not_recognized.pop()
                    alternatives[option] = fuzzy_match(option, available, n=3)
                did_you_mean = ", ".join(
                    [
                        (
                            "'{}' (did you mean {}?)".format(
                                o, "|".join(["'%s'" % _ for _ in a])
                            )
                            if a
                            else "'%s'" % o
                        )
                        for o, a in alternatives.items()
                    ]
                )
                raise LoggedError(
                    logger,
                    (
                        "%s '%s' does not recognize some options: %s. "
                        "Check the documentation for '%s'."
                    ),
                    block,
                    name,
                    did_you_mean,
                    block,
                )
            updated[name].update(input_block[name])
            # save params and priors of class to combine later
            default_params_info[name] = default_class_info.get("params", {})
            default_prior_info[name] = default_class_info.get("prior", {})
    # Add priors info, after the necessary checks
    if "prior" in input_info or any(default_prior_info.values()):
        updated_info["prior"] = input_info.get("prior", {})
    for prior_info in default_prior_info.values():
        for name, prior in prior_info.items():
            if updated_info["prior"].get(name, prior) != prior:
                raise LoggedError(
                    logger, "Two different priors cannot have the same name: '%s'.", name
                )
            updated_info["prior"][name] = prior
    # Add parameters info, after the necessary updates and checks
    defaults_merged = merge_default_params_info(default_params_info)
    param_info: ExpandedParamsDict = merge_params_info(
        [defaults_merged, input_info.get("params", {})], default_derived=False
    )
    updated_info["params"] = param_info  # type: ignore
    # Add aggregated chi2 params
    if info.get("likelihood") and add_aggr_chi2:
        all_types = set(
            chain(
                *[
                    str_to_list(like_info.get("type", []) or [])
                    for like_info in updated_info["likelihood"].values()
                    if like_info is not None
                ]
            )
        )
        add_aggregated_chi2_params(param_info, all_types)
    # Add automatically-defined parameters
    if "auto_params" in updated_info:
        make_auto_params(updated_info.pop("auto_params"), param_info)
    # Add aliases for theory params (after merging!)
    for name in ("theory", "likelihood"):
        if isinstance(updated_info.get(name), dict):
            for item in updated_info[name].values():
                if renames := item.get("renames"):
                    if not isinstance(renames, Mapping):
                        raise LoggedError(
                            logger,
                            (
                                "'renames' should be a dictionary of name mappings "
                                "(or you meant to use 'aliases')"
                            ),
                        )
                    renames_flat = [set([k] + str_to_list(v)) for k, v in renames.items()]
                    for p in param_info:
                        # Find all rename sets containing this parameter
                        if matching_rename_sets := [
                            rename_set for rename_set in renames_flat if p in rename_set
                        ]:
                            # Merge all rename sets containing this parameter
                            all_names = set().union(*matching_rename_sets)
                            # Add any existing renames and remove the parameter itself
                            existing_renames = set(
                                str_to_list(param_info[p].get("renames", []))
                            )
                            param_info[p]["renames"] = list(
                                all_names.union(existing_renames) - {p}
                            )
    # Rest of the options
    for k, v in input_info.items():
        if k not in updated_info:
            updated_info[k] = v
    return updated_info


def merge_default_params_info(defaults: LikesDict):
    """
    Merges default parameters info for all likelihoods.
    Checks that multiple defined (=shared) parameters have equal info.
    """
    defaults_merged: LikesDict = {}
    for lik, params in defaults.items():
        for p, info in (params or {}).items():
            # if already there, check consistency
            if p in defaults_merged:
                if info != defaults_merged[p]:
                    raise LoggedError(
                        logger,
                        (
                            "Parameter '%s' multiply defined, but inconsistent info:"
                            " For likelihood '%s' is '%r', but for some other "
                            "likelihood it was '%r'. Check your defaults!"
                        ),
                        p,
                        lik,
                        info,
                        defaults_merged[p],
                    )
                if mpi.is_main_process():
                    logger.debug("Parameter '%s' is multiply defined but consistent.", p)
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
    current_info = {
        p: expand_info_param(v, default_derived) for p, v in params_infos[0].items() or {}
    }
    for new_info in params_infos[1:]:
        if not new_info:
            continue
        for p, new_info_p in new_info.items():
            if p not in current_info:
                current_info[p] = {}
            new_info_p = expand_info_param(new_info_p)
            current_info[p].update(deepcopy(new_info_p))
            # Account for incompatibilities: "prior" and ("value" or "derived"+bounds)
            incompatibilities = {
                "prior": ["value", "derived", "min", "max"],
                "value": ["prior", "ref", "proposal"],
                "derived": ["prior", "drop", "ref", "proposal"],
            }
            for f1, incomp in incompatibilities.items():
                if f1 in new_info_p:
                    for f2 in incomp:
                        current_info[p].pop(f2, None)  # type: ignore
    # Re-sort, so that rightmost info takes precedence *also* in the sorting
    new_order_sorted = chain(*params_infos[::-1])
    # The following removes duplicates maintaining order (keeps the first occurrence)
    new_order = dict.fromkeys(new_order_sorted)
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
    current_info = None
    for new_info in infos[1:]:
        if isinstance(previous_info, str):
            raise LoggedError(logger, previous_info)
        previous_params_info = deepcopy(previous_info.pop("params", {}) or {})
        new_params_info = deepcopy(new_info).pop("params", {}) or {}
        # NS: params have been popped, since they have their own merge function
        current_info = recursive_update(previous_info, new_info)
        current_info["params"] = merge_params_info(
            [previous_params_info, new_params_info]
        )
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
    else:
        myprint = logger.info
    myname = inspect.stack()[0][3]
    ignorable = {
        "debug",
        "resume",
        "force",
        packages_path_input,
        "test",
        "version",
        "stop_at_error",
    }
    ignore = set() if strict else ignorable
    ignore = ignore.union(ignore_blocks or [])
    if {info for info in info_old if info_old[info] is not None} - ignore != {
        info for info in info_new if info_new[info] is not None
    } - ignore:
        myprint(
            myname
            + ": different blocks or options: {!r} (old) vs {!r} (new)".format(
                set(info_old).difference(ignore), set(info_new).difference(ignore)
            )
        )
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
                myname
                + ": different [{}] or different order of them: {!r} vs {!r}".format(
                    block_name, list(block1), list(block2)
                )
            )
            return False
        # 2. Gather general options to be ignored
        ignore_k = set()
        if not strict:
            if block_name in ["theory", "likelihood"]:
                ignore_k.update({"input_params", "output_params"})
            elif block_name == "params":
                for param in block1:
                    # Unify notation
                    block1[param] = expand_info_param(block1[param])
                    block2[param] = expand_info_param(block2[param])
                    ignore_k.update({"latex", "renames", "ref", "proposal", "min", "max"})
                    # Fixed params, it doesn't matter if they are saved as derived
                    if "value" in block1[param]:
                        block1[param].pop("derived", None)
                    if "value" in block2[param]:
                        block2[param].pop("derived", None)
                    # Renames: order does not matter
                    block1[param]["renames"] = set(block1[param].get("renames", []))
                    block2[param]["renames"] = set(block2[param].get("renames", []))
        # 3. Now check component/parameters one-by-one
        for k in block1:
            if not strict:
                # Add component-specific options to be ignored
                if block_name in kinds:
                    ignore_k_this = ignore_k.union({"python_path"})
                    if "external" not in block1[k]:
                        try:
                            component_path = (
                                block1[k].pop("python_path", None)
                                if isinstance(block1[k], dict)
                                else None
                            )
                            cls = get_component_class(
                                k,
                                kind=block_name,
                                component_path=component_path,
                                class_name=(block1[k] or {}).get("class"),
                                logger=logger,
                            )
                            ignore_k_this.update(
                                set(getattr(cls, "_at_resume_prefer_new", []))
                            )
                        except ImportError:
                            pass
                    # Pop ignored and kept options
                    for j in ignore_k_this:
                        block1[k].pop(j, None)
                        block2[k].pop(j, None)
            if not strict:
                # For Mapping values, homogenize to None empty lists, sets, maps, etc.
                # e.g. {value: {}} should be equal to {value: None}
                for value in [block1[k], block2[k]]:
                    if isinstance(value, MutableMapping):
                        for kk in value:
                            if hasattr(value[kk], "__len__") and len(value[kk]) == 0:
                                value[kk] = None
            if block1[k] != block2[k]:
                # For clarity, pop common stuff before printing
                to_pop = [j for j in block1[k] if block1[k].get(j) == block2[k].get(j)]
                for j in to_pop:
                    block1[k].pop(j, None)
                    block2[k].pop(j, None)
                myprint(
                    myname
                    + f": different content of [{block_name}:{k}]"
                    + " -- (re-run with `debug: True` for more info)"
                )
                if not print_not_log:
                    logger.debug(f"{block1[k]!r} (old) vs {block2[k]!r} (new)")
                return False
    return True


def get_preferred_old_values(info_old):
    """
    Returns selected values in `info_old`, which are preferred at resuming.
    """
    keep_old: InfoDict = {}
    for block_name, block in info_old.items():
        if block_name not in kinds or not block:
            continue
        for k in block:
            try:
                component_path = (
                    block[k].pop("python_path", None)
                    if isinstance(block[k], dict)
                    else None
                )
                cls = get_component_class(
                    k,
                    kind=block_name,
                    component_path=component_path,
                    class_name=(block[k] or {}).get("class"),
                    logger=logger,
                )
                prefer_old_k_this = getattr(cls, "_at_resume_prefer_old", [])
                if prefer_old_k_this:
                    if block_name not in keep_old:
                        keep_old[block_name] = {}
                    keep_old[block_name].update(
                        {k: {o: block[k][o] for o in prefer_old_k_this if o in block[k]}}
                    )
            except ImportError:
                pass
    return keep_old


def make_auto_params(auto_params, params_info):
    def replace(item, tag):
        if isinstance(item, dict):
            for key, val in list(item.items()):
                item[key] = replace(val, tag)
        elif isinstance(item, str) and "%s" in item:
            item = item % tag
        return item

    for k, v in auto_params.items():
        if "%s" not in k:
            raise LoggedError(
                logger, "auto_param parameter names must have '%s' placeholder"
            )
        replacements = v.pop("auto_range")
        if isinstance(replacements, str):
            replacements = eval(replacements)
        for value in replacements:
            params_info[k % value] = replace(deepcopy_where_possible(v), value)
