import os
from collections import OrderedDict
from cobaya.model import get_model
from cobaya.theory import Theory
from cobaya.tools import load_module
from .common import process_modules_path


# Test separating out the BBN consistency constraint into separate theory code,
# using CAMB's BBN interpolator class. Tests dependencies/multi-theory with one
# agnostic theory

class BBN(Theory):
    bbn = None

    def get_requirements(self):
        return {'ombh2', 'nnu'}

    def calculate(self, state, want_derived=True, **params_values_dict):
        yhe = self.bbn.Y_He(self.provider.get_param('ombh2'),
                            self.provider.get_param('nnu') - 3.046)
        state['derived'] = {'YHe': yhe}

    def get_can_provide_params(self):
        return ['YHe']


class BBN2(Theory):
    params = {'ombh2': None, 'nnu': None, 'YHe': {'derived': True}}
    bbn = None

    def calculate(self, state, want_derived=True, **params_values_dict):
        if want_derived:
            state['derived'] = {'YHe': self.bbn.Y_He(params_values_dict['ombh2'],
                                                     params_values_dict['nnu'] - 3.046)}


def cmb_likelihood(_derived={'check'},
                   _theory={'Hubble': {'z': [0.5]}, 'CAMBdata': None}):
    results = _theory.get_CAMBdata()
    _derived['check'] = results.Params.YHe
    return results.Params.YHe


camb_params = {
    "ombh2": 0.022274,
    "omch2": 0.11913,
    "cosmomc_theta": 0.01040867,
    "As": 0.2132755716e-8,
    "ns": 0.96597,
    "tau": 0.0639,
    "nnu": 3.046}

bbn_table = "PRIMAT_Yp_DH_Error.dat"
debug = True
info = {'likelihood': {'cmb': cmb_likelihood},
        'theory': OrderedDict({
            'camb': {"extra_args": {"lens_potential_accuracy": 1,
                                    "bbn_predictor": bbn_table},
                     "requires": ['YHe', 'ombh2'], "stop_at_error": True},
            'bbn': {'external': BBN, 'provides': ['YHe']}}),
        'params': camb_params,
        'debug': debug}

info2 = {'likelihood': {'cmb': {'external': cmb_likelihood}},
         'theory': OrderedDict({
             'camb': {"requires": ['YHe', 'ombh2']},
             'bbn': BBN2}),
         'params': camb_params, 'debug': debug}


def test_bbn_yhe(modules):
    modules = process_modules_path(modules)
    camb = load_module("camb", path=os.path.join(modules, "code", "CAMB"))
    from camb.bbn import BBN_table_interpolator
    BBN.bbn = BBN_table_interpolator(bbn_table)
    BBN2.bbn = BBN.bbn

    info['params']['check'] = {'derived': True}

    for inf in (info, info2):
        inf['modules'] = modules
        for order in [1, -1]:
            for explicit_derived in [None, None, {'derived': True}]:
                print(inf, order, explicit_derived)
                model = get_model(inf)
                loglike, derived = model.loglikes({})
                vals = set([BBN.bbn.Y_He(camb_params['ombh2'])] + derived)
                assert len(vals) == 1, \
                    "wrong Yhe value: %s" % vals
                inf['params']["YHe"] = explicit_derived
            inf['params'].pop('YHe')
            inf['theory'] = OrderedDict(
                (p, v) for p, v in reversed(list(inf['theory'].items())))
