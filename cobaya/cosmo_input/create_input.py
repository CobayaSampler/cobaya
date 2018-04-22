from copy import deepcopy
from collections import OrderedDict as odict
from six import string_types
import numpy as np

from cobaya.input import merge_info
from cobaya.conventions import _theory, _params, _p_drop, _p_alias
from cobaya.input import get_default_info
import input_database

planck_to_classy = get_default_info("classy", _theory)[_theory]["classy"]["planck_to_classy"]
planck_to_camb = get_default_info("camb", _theory)[_theory]["camb"]["planck_to_camb"]


def translate(p, info=None, dictionary=None, add_alias=False):
    dictionary = dictionary or {}
    # Ignore if dropped
    if not (info if hasattr(info, "keys") else {}).get(_p_drop, False):
        p_old = p
        p = dictionary.get(p, p)
        if hasattr(info, "keys") and p_old != p:
            pre_aliases = np.atleast_1d(info.get(_p_alias, [])).tolist()
            info.update({_p_alias: pre_aliases + [p_old]})
    # Try to modify lambda parameters too!
    if isinstance(info, string_types):
        if info.startswith("lambda"):
            arguments = info.split(":")[0].split()[1:]
            if not isinstance(arguments, string_types):
                arguments = "".join(arguments)
            arguments = arguments.split(",")
            arguments_t = [translate(pi, dictionary=dictionary)[0] for pi in arguments]
            for pi, pit in zip(arguments, arguments_t):
                info = info.replace(pi, pit)
    if hasattr(info, "keys") and "derived" in info:
        info["derived"] = translate(p, info["derived"], dictionary=dictionary)[1]
    return p, info


def create_input(**kwargs):
    preset = kwargs.get("preset", None)
    if preset:
        info = deepcopy(input_database.preset[preset])
        info.pop(input_database._desc, None)
        return create_input(**info)
    # Need to copy to select theory
    infos_model = [deepcopy(getattr(input_database, k)[kwargs.get(k, "null")]) for k in [
        "primordial", "geometry", "hubble", "baryons", "dark_matter", "dark_energy",
        "neutrinos", "bbn", "reionization", "cmb_lensing", "derived"]]
    for info in infos_model:
        info.pop(input_database._desc, None)
        try:
            theory = kwargs.get("theory")
            info[_theory] = {theory: info[_theory][theory]}
            # Translate parameter names
        except KeyError:
            return "Model not compatible with theory code '%s'"%theory
    infos_exp = [deepcopy(getattr(input_database, k)[kwargs[k]]) for k in ["cmb"]]
    for info in infos_exp:
        info.pop(input_database._desc, None)
    sampler = kwargs.get("sampler")
    info_sampler = deepcopy(getattr(input_database, "sampler")[sampler])
    info_sampler.pop(input_database._desc, None)
    merged = merge_info(*(infos_model+infos_exp+[info_sampler]))
    # Translate from Planck param names
    theo = list(merged[_theory].keys())[0]
    merged_params_translated = odict([
        translate(p, info, {"classy": planck_to_classy, "camb": planck_to_camb}[theo],
                  add_alias=True) for p,info in merged[_params].items()])
    merged[_params] = merged_params_translated
    return merged
