# Global
from copy import deepcopy
from typing import MutableMapping, Mapping

# Local
from cobaya.input import get_default_info, merge_info
from cobaya.conventions import kinds, partag, _params
from cobaya.parameterization import reduce_info_param
from . import input_database


def translate(p, info=None, dictionary=None):
    dictionary = dictionary or {}
    # Ignore if dropped
    if not (info if isinstance(info, Mapping) else {}).get(partag.drop, False):
        p = dictionary.get(p, p)
    # Try to modify lambda parameters too!
    if isinstance(info, str):
        if info.startswith("lambda"):
            arguments = info.split(":")[0].split()[1:]
            if not isinstance(arguments, str):
                arguments = "".join(arguments)
            arguments = arguments.split(",")
            arguments_t = [translate(pi, dictionary=dictionary)[0] for pi in arguments]
            for pi, pit in zip(arguments, arguments_t):
                info = info.replace(pi, pit)
    if ((isinstance(info, MutableMapping) and partag.derived in info and
         isinstance(info[partag.derived], str))):
        info[partag.derived] = translate(p, info[partag.derived],
                                         dictionary=dictionary)[1]
    elif (isinstance(info, MutableMapping) and partag.value in info and
          isinstance(info[partag.value], str)):
        info[partag.value] = translate(p, info[partag.value], dictionary=dictionary)[1]
    return p, info


def create_input(**kwargs):
    get_comments = kwargs.pop("get_comments", False)
    preset = kwargs.pop("preset", None)
    if preset:
        fields = deepcopy(input_database.preset[preset])
        fields.update(kwargs)
        return create_input(get_comments=get_comments, **fields)
    # Need to copy to select theory.
    # Likelihoods always AT THE END!
    # (to check that sampled parameters are not redefined as derived)
    infos = {}
    kwargs_params = ["primordial", "geometry", "hubble", "matter", "neutrinos",
                     "dark_energy", "bbn", "reionization"]
    kwargs_likes = ["like_cmb", "like_bao", "like_des", "like_sn", "like_H0"]
    for k in kwargs_params + kwargs_likes:
        if k not in kwargs:
            infos[k] = {}
        try:
            infos[k] = deepcopy(
                getattr(input_database, k)[kwargs.get(k, input_database._none)])
        except KeyError:
            raise ValueError("Unknown value '%s' for '%s'" %
                             (kwargs.get(k, input_database._none), k))
    theory_requested = kwargs.get(kinds.theory)
    for i, (field, info) in enumerate(infos.items()):
        if not info:
            continue
        error_msg = info.pop(input_database._error_msg, None)
        try:
            info[kinds.theory] = {theory_requested: info[kinds.theory][theory_requested]}
        except KeyError:
            return ("There is no preset for\n'%s'" % (
                info.get(input_database._desc, field)) +
                    "with theory code '%s'." % theory_requested +
                    ("\n--> " + error_msg if error_msg else
                     "\nThis does not mean that you cannot use this model with that "
                     "theory code; just that we have not implemented this yet."))
        # Add non-translatable parameters (in info["theory"][classy|camb][params])
        if _params not in info:
            info[_params] = {}
        info[_params].update(
            (info[kinds.theory][theory_requested] or {}).pop(_params, {}))
        # Remove the *derived* parameters mentioned by the likelihoods that
        # are already *sampled* by some part of the model
        if field.startswith("like_") and _params in info:
            remove_derived = []
            for p in info[_params]:
                if any((p in info_part[_params])
                        for info_part in list(infos.values())[:i]):
                    remove_derived += [p]
            for p in remove_derived:
                info[_params].pop(p)
    # Prepare sampler info
    info_sampler = deepcopy(input_database.sampler.get(kwargs.get(kinds.sampler), {}))
    if info_sampler:
        sampler_name = list(info_sampler[kinds.sampler])[0]
        info_sampler[kinds.sampler][sampler_name] = info_sampler[kinds.sampler][
                                                        sampler_name] or {}
        # Add recommended options for samplers
        for info in infos.values():
            this_info_sampler = info.pop(kinds.sampler, {})
            if sampler_name in this_info_sampler:
                info_sampler[kinds.sampler][sampler_name].update(
                    this_info_sampler[sampler_name])
    # Reorder, to have the *parameters* shown in the correct order, and *merge*
    all_infos = [info_sampler] + [infos[k]
                                  for k in kwargs_likes[::-1] + kwargs_params[::-1]]
    comments = [info.pop(input_database._comment, None) for info in all_infos]
    comments = [c for c in comments if c]
    [info.pop(input_database._desc, None) for info in all_infos]
    merged = merge_info(*all_infos)
    # Simplify parameter infos
    for p, info in merged[_params].items():
        merged[_params][p] = reduce_info_param(info)
    # Translate from Planck param names
    planck_to_theo = \
        get_default_info(theory_requested, kinds.theory)[partag.renames]
    if kwargs.get("planck_names", False):
        merged[kinds.theory][theory_requested] = merged[kinds.theory][
                                                     theory_requested] or {}
        merged[kinds.theory][theory_requested]["use_renames"] = True
    else:
        merged_params_translated = dict([
            translate(p, info, planck_to_theo)
            for p, info in merged[_params].items()])
        merged[_params] = merged_params_translated
    if get_comments and comments:
        merged[input_database._comment] = comments
    return merged
