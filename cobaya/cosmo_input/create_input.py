from copy import deepcopy

from cobaya.input import merge_info
from cobaya.conventions import _theory
import input_database


def create_input(**kwargs):
    preset = kwargs.get("preset", None)
    if preset:
        info = deepcopy(input_database.preset[preset])
        info.pop(input_database._desc, None)
        return create_input(**info)
    # Need to copy to select theory
    infos_model = [deepcopy(getattr(input_database, k)[kwargs[k]]) for k in [
        "primordial", "geometry", "hubble", "baryons", "dark_matter", "dark_energy",
        "neutrinos", "bbn", "reionization", "cmb_lensing"]]
    for info in infos_model:
        info.pop(input_database._desc, None)
        try:
            theory = kwargs.get("theory")
            info[_theory] = {theory: info[_theory][theory]}
        except KeyError:
            return None
    infos_exp = [deepcopy(getattr(input_database, k)[kwargs[k]]) for k in ["cmb"]]
    for info in infos_exp:
        info.pop(input_database._desc, None)
    sampler = kwargs.get("sampler")
    info_sampler = deepcopy(getattr(input_database, "sampler")[sampler])
    info_sampler.pop(input_database._desc, None)
    return merge_info(*(infos_model+infos_exp+[info_sampler]))
