from copy import deepcopy

from cobaya.input import merge_info
from cobaya.conventions import _theory
import input_database


def create_input(theory, primordial, reionization):
    # Need to copy to select theory
    infos = [deepcopy(info) for info in [input_database.primordial[primordial], input_database.reionization[reionization]]]
    for info in infos:
        info.pop(input_database._desc, None)
        info[_theory] = {theory: info[_theory][theory]}
    return merge_info(*infos)
