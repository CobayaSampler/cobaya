from copy import deepcopy

from cobaya.input import merge_info
from cobaya.conventions import _theory
import input_database


def create_input(
        theory=None, primordial=None, hubble=None, baryons=None, dark_matter=None, neutrinos=None, reionization=None,
        cmb=None, sampler=None, preset=None):
    if preset:
        info = deepcopy(input_database.preset[preset])
        info.pop(input_database._desc, None)
        return create_input(**info)
    # Need to copy to select theory
    infos_model = [deepcopy(info) for info in [
        input_database.primordial[primordial], input_database.hubble[hubble],
        input_database.baryons[baryons], input_database.dark_matter[dark_matter],
        input_database.neutrinos[neutrinos], input_database.reionization[reionization]]]
    for info in infos_model:
        info.pop(input_database._desc, None)
        try:
            info[_theory] = {theory: info[_theory][theory]}
        except KeyError:
            return None
    infos_exp = [deepcopy(info) for info in [input_database.cmb[cmb]]]
    for info in infos_exp:
        info.pop(input_database._desc, None)
    info_sampler = deepcopy(input_database.sampler[sampler])
    info_sampler.pop(input_database._desc, None)
    return merge_info(*(infos_model+infos_exp+[info_sampler]))
