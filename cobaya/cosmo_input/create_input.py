from collections.abc import Mapping, MutableMapping
from copy import deepcopy

from cobaya.input import get_default_info, merge_info
from cobaya.parameterization import reduce_info_param
from cobaya.typing import InputDict

from . import input_database


def translate(p, info=None, dictionary=None):
    dictionary = dictionary or {}
    # Ignore if dropped
    if not (info if isinstance(info, Mapping) else {}).get("drop", False):
        p = dictionary.get(p, p)
    # Try to modify lambda parameters too!
    if isinstance(info, str):
        if info.startswith("lambda"):
            arguments = "".join(info.split(":")[0].split()[1:]).split(",")
            arguments_t = [translate(pi, dictionary=dictionary)[0] for pi in arguments]
            for pi, pit in zip(arguments, arguments_t):
                info = info.replace(pi, pit)
    if (
        isinstance(info, MutableMapping)
        and "derived" in info
        and isinstance(info["derived"], str)
    ):
        info["derived"] = translate(p, info["derived"], dictionary=dictionary)[1]
    elif (
        isinstance(info, MutableMapping)
        and "value" in info
        and isinstance(info["value"], str)
    ):
        info["value"] = translate(p, info["value"], dictionary=dictionary)[1]
    return p, info


def create_input(**kwargs) -> InputDict:
    get_comments = kwargs.pop("get_comments", False)
    preset = kwargs.pop("preset", None)
    if preset:
        fields = deepcopy(input_database.preset[preset])
        fields.update(kwargs)
        return create_input(get_comments=get_comments, **fields)
    # Need to copy to select theory.
    # Likelihoods always AT THE END!
    # (to check that sampled parameters are not redefined as derived)
    infos: dict = {}
    kwargs_params = [
        "primordial",
        "geometry",
        "hubble",
        "matter",
        "neutrinos",
        "dark_energy",
        "bbn",
        "reionization",
    ]
    kwargs_likes = ["like_cmb", "like_bao", "like_des", "like_sn", "like_H0"]
    for k in kwargs_params + kwargs_likes:
        if k not in kwargs:
            infos[k] = {}
        try:
            infos[k] = deepcopy(
                getattr(input_database, k)[kwargs.get(k, input_database.none)]
            )
        except KeyError:
            raise ValueError(
                "Unknown value '{}' for '{}'".format(
                    kwargs.get(k, input_database.none), k
                )
            )
    theory_requested = kwargs.get("theory")
    for i, (field, info) in enumerate(infos.items()):
        if not info:
            continue
        error_msg = info.pop(input_database.error_msg, None)
        try:
            info["theory"] = {theory_requested: info["theory"][theory_requested]}
        except KeyError:
            return (
                "There is no preset for\n'%s'" % (info.get("desc", field))
                + "with theory code '%s'." % theory_requested
                + (
                    "\n--> " + error_msg
                    if error_msg
                    else "\nThis does not mean that you cannot use this model with that "
                    "theory code; just that we have not implemented this yet."
                )
            )
        # Add non-translatable parameters (in info["theory"][classy|camb][params])
        if "params" not in info:
            info["params"] = {}
        info["params"].update((info["theory"][theory_requested] or {}).pop("params", {}))
        # Remove the *derived* parameters mentioned by the likelihoods that
        # are already *sampled* by some part of the model
        if field.startswith("like_") and "params" in info:
            remove_derived = []
            for p in info["params"]:
                if any(
                    (p in info_part["params"]) for info_part in list(infos.values())[:i]
                ):
                    remove_derived += [p]
            for p in remove_derived:
                info["params"].pop(p)
    # Prepare sampler info
    info_sampler = deepcopy(input_database.sampler.get(kwargs.get("sampler") or "", {}))
    if info_sampler:
        sampler_name = list(info_sampler["sampler"])[0]
        info_sampler["sampler"][sampler_name] = (
            info_sampler["sampler"][sampler_name] or {}
        )
        # Add recommended options for samplers
        for info in infos.values():
            this_info_sampler = info.pop("sampler", {})
            if sampler_name in this_info_sampler:
                info_sampler["sampler"][sampler_name].update(
                    this_info_sampler[sampler_name]
                )
    # Reorder, to have the *parameters* shown in the correct order, and *merge*
    all_infos = [info_sampler] + [
        infos[k] for k in kwargs_likes[::-1] + kwargs_params[::-1]
    ]
    comments = [info.pop("comment", None) for info in all_infos]
    comments = [c for c in comments if c]
    [info.pop("desc", None) for info in all_infos]
    merged = merge_info(*all_infos)
    # Simplify parameter infos
    for p, info in merged["params"].items():
        merged["params"][p] = reduce_info_param(info)
    # Translate from Planck param names
    planck_to_theo = get_default_info(theory_requested, "theory")["renames"]
    if kwargs.get("planck_names", False):
        merged["theory"][theory_requested] = merged["theory"][theory_requested] or {}
    else:
        merged_params_translated = dict(
            translate(p, info, planck_to_theo) for p, info in merged["params"].items()
        )
        merged["params"] = merged_params_translated
    if get_comments and comments:
        merged["comment"] = comments
    return merged  # type: ignore
