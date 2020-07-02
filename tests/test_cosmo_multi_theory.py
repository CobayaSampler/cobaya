import os
import numpy as np
from typing import Any
from copy import deepcopy
from cobaya.model import get_model
from cobaya.theory import Theory
from cobaya.tools import load_module
from cobaya.likelihood import LikelihoodInterface, Likelihood
from cobaya.conventions import empty_dict
from .common import process_packages_path
from .conftest import install_test_wrapper
from .test_cosmo_camb import get_camb

# Test separating out the BBN consistency constraint into separate theory code,
# using CAMB's BBN interpolator class. Tests dependencies/multi-theory with one
# agnostic theory

class BBN(Theory):
    bbn: Any

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
    bbn: Any

    def calculate(self, state, want_derived=True, **params_values_dict):
        if want_derived:
            state['derived'] = {'YHe': self.bbn.Y_He(params_values_dict['ombh2'],
                                                     params_values_dict['nnu'] - 3.046)}


# noinspection PyDefaultArgument
def cmb_likelihood(_self):
    results = _self.provider.get_CAMBdata()
    derived = {'check': results.Params.YHe}
    return results.Params.YHe, derived

cmb_likelihood_info = {'external': cmb_likelihood, 'output_params': 'check',
                       'requires': {'Hubble': {'z': [0.5]}, 'CAMBdata': None}}

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
info = {'likelihood': {'cmb': cmb_likelihood_info},
        'theory': {
            'camb': {"extra_args": {"lens_potential_accuracy": 1},
                     "requires": ['YHe', 'ombh2']},
            'bbn': {'external': BBN, 'provides': ['YHe']}},
        'params': camb_params,
        'debug': debug, 'stop_at_error': True}

info2 = {'likelihood': {'cmb': cmb_likelihood_info},
         'theory': {'camb': {"requires": ['YHe', 'ombh2']}, 'bbn': BBN2},
         'params': camb_params, 'debug': debug}


def test_bbn_yhe(packages_path, skip_not_installed):
    packages_path = process_packages_path(packages_path)
    install_test_wrapper(skip_not_installed, get_camb, packages_path)
    from camb.bbn import BBN_table_interpolator
    BBN.bbn = BBN_table_interpolator(bbn_table)
    BBN2.bbn = BBN.bbn

    info['params']['check'] = {'derived': True}

    for inf in (info, info2):
        inf['packages_path'] = packages_path
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
            inf['theory'] = {p: v for p, v in reversed(list(inf['theory'].items()))}


# Not inherit from BBN to derive likelihoods that account for the theory error

class BBN_likelihood(BBN2, LikelihoodInterface):
    """
    Sample YHe and just calculate a direct theory likelihood
    """
    params = dict(zip(['ombh2', 'nnu', 'YHe'], [None] * 3))

    def calculate(self, state, want_derived=True, **params_values_dict):
        ombh2 = params_values_dict['ombh2']
        delta_neff = params_values_dict['nnu'] - 3.046
        yhemean = self.bbn.Y_He(ombh2, delta_neff)
        error = self.bbn.get('sig(Yp^BBN)', ombh2, delta_neff)
        state['logp'] = -(params_values_dict['YHe'] - yhemean) ** 2 / (2 * error ** 2)

    def get_can_provide_params(self):
        return {}


class BBN_with_theory_errors(BBN, LikelihoodInterface):
    """
    The BBN theory prediction has an error. So we can derive a likelihood for
    sampling over a separate error parameter to account for it if we approximate
    the distribution as Gaussian.
    """
    params = {'BBN_delta': {'prior': {'min': 0, 'max': 1},
                            'ref': dict(dist='norm', loc=0, scale=1)}}

    def calculate(self, state, want_derived=True, **params_values_dict):
        ombh2, nnu = self.provider.get_param(['ombh2', 'nnu'])
        delta_neff = nnu - 3.046
        yhemean = self.bbn.Y_He(ombh2, delta_neff)
        # get theory error (neglecting difference between Y_He and Yp^BBN)
        error = self.bbn.get('sig(Yp^BBN)', ombh2, delta_neff)
        yhe = yhemean + error * params_values_dict['BBN_delta']
        if want_derived:
            state['derived'] = {'YHe': yhe}
        # take the error to be Gaussian
        state['logp'] = -params_values_dict['BBN_delta'] ** 2 / 2


info_error = {'likelihood': dict([('cmb', cmb_likelihood_info),
                                  ('BBN', BBN_likelihood)]),
              'theory': {'camb': {"requires": ['YHe', 'ombh2']}},
              'params': dict(YHe={'prior': {'min': 0, 'max': 1}}, **camb_params),
              'debug': debug}

info_error2 = {'likelihood': {'cmb': cmb_likelihood_info,
                              'BBN': {'external': BBN_with_theory_errors,
                                      'provides': 'YHe'}},
               'theory': {'camb': {"requires": ['YHe', 'ombh2']}},
               'params': dict(BBN_delta={'prior': {'min': -5, 'max': 5}},
                              **camb_params),
               'debug': debug}


def test_bbn_likelihood(packages_path, skip_not_installed):
    packages_path = process_packages_path(packages_path)
    install_test_wrapper(skip_not_installed, get_camb, packages_path)
    from camb.bbn import BBN_table_interpolator
    BBN_likelihood.bbn = BBN_table_interpolator(bbn_table)
    info_error['packages_path'] = packages_path
    model = get_model(info_error)
    assert np.allclose(model.loglikes({'YHe': 0.246})[0], [0.246, -0.84340], rtol=1e-4), \
        "Failed BBN likelihood with %s" % info_error

    # second case, BBN likelihood has to be calculated before CAMB
    BBN_with_theory_errors.bbn = BBN_likelihood.bbn
    info_error2['packages_path'] = packages_path
    model = get_model(info_error2)
    assert np.allclose(model.loglikes({'BBN_delta': 1.0})[0], [0.24594834, -0.5],
                       rtol=1e-4)


class ExamplePrimordialPk(Theory):

    def initialize(self):
        # need to provide valid results at wide k range, any that might be used
        self.ks = np.logspace(-5.5, 2, 1000)

    def calculate(self, state, want_derived=True, **params_values_dict):
        pivot_scalar = 0.05
        pk = (self.ks / pivot_scalar) ** (
                params_values_dict['testns'] - 1) * params_values_dict['testAs']
        state['primordial_scalar_pk'] = {'kmin': self.ks[0], 'kmax': self.ks[-1],
                                         'Pk': pk, 'log_regular': True}

    def get_primordial_scalar_pk(self):
        return self._current_state['primordial_scalar_pk']

    def get_can_support_params(self):
        return ['testAs', 'testns']


testAs = 1.8e-9
testns = 0.8


class Pklike(Likelihood):
    def logp(self, **params_values):
        results = self.provider.get_CAMBdata()
        print(results.Params.scalar_power(1.1),
              testAs * (1.1 / 0.05) ** (testns - 1))
        assert np.allclose(results.Params.scalar_power(1.1),
                           testAs * (1.1 / 0.05) ** (testns - 1), rtol=1e-3, atol=1e-20)
        return 0

    def get_requirements(self):
        return {'Cl': {'tt': 1000}, 'CAMBdata': None}


info_pk = {'likelihood': {'cmb': Pklike},
           'theory': {'camb': {"external_primordial_pk": True},
                      'my_pk': ExamplePrimordialPk},
           'params': {
               "ombh2": 0.022274,
               "omch2": 0.11913,
               "cosmomc_theta": 0.01040867,
               "tau": 0.0639,
               "nnu": 3.046,
               'testAs': {'prior': {'min': 1e-9, 'max': 1e-8}},
               'testns': {'prior': {'min': 0.8, 'max': 1.2}}
           },
           'stop_at_error': True,
           'debug': debug}


def test_primordial_pk(packages_path, skip_not_installed):
    packages_path = process_packages_path(packages_path)
    info_pk['packages_path'] = packages_path
    model = install_test_wrapper(skip_not_installed, get_model, info_pk)
    model.loglikes({'testAs': testAs, 'testns': testns})


class BinnedPk(Theory):
    # example splined power spectrum exp(-2 tau)P(k) based on bin values.
    # Can pass dense sampling to CAMB to reproduce any function, here bins are directly
    # cubic spline values use by CAMB.
    # Note need to  have wide k bounds or latest CAMB (which takes value beyond start
    # and end bins to be equal to the end bins).
    nbins: int = 20
    k_min_bin: float = np.log10(0.001)
    k_max_bin: float = np.log10(0.35)
    scale: float = 1e-9
    bin_par = {'prior': {'min': 0, 'max': 100}}

    def initialize(self):
        self.ks = np.logspace(self.k_min_bin, self.k_max_bin, self.nbins)

    def get_requirements(self):
        return {'tau'}

    def calculate(self, state, want_derived=True, **params_values_dict):
        pk = np.zeros_like(self.ks)
        for b in range(self.nbins):
            pk[b] = params_values_dict['b%s' % (b + 1)]
        pk *= self.scale * np.exp(2 * self.provider.get_param('tau'))
        # should use log_regular: True for speed if the binning is log regular
        # here test the non-regular option for coverage
        state['primordial_scalar_pk'] = {'k': self.ks,
                                         'Pk': pk, 'log_regular': False}

    def get_primordial_scalar_pk(self):
        return self._current_state['primordial_scalar_pk']

    @classmethod
    def get_class_options(cls, input_options=empty_dict):
        # dynamically generate defaults for params based on nbins
        options = super().get_class_options().copy()
        nbins = input_options['nbins']
        bin_par = input_options.get('bin_par', cls.bin_par)
        params = {}
        for b in range(nbins):
            par = deepcopy(bin_par.copy())
            par['label'] = 'b_%s' % (b + 1)
            params['b%s' % (b + 1)] = par
        options['params'] = params
        return options


def test_pk_binning(packages_path, skip_not_installed):
    # reproduce power law by sending in spline point values
    # has to be fine sampling to get to 1e-3 precision in test.
    nbins = 40
    tau = 0.05
    k_min_bin = -5.5
    k_max_bin = 2

    info = {'packages_path': process_packages_path(packages_path),
            'likelihood': {'cmb': Pklike},
            'theory': {'camb': {"external_primordial_pk": True},
                       'my_pk': {"external": BinnedPk,
                                 'nbins': nbins, 'k_min_bin': k_min_bin,
                                 'k_max_bin': k_max_bin
                                 }},
            'params': {
                "ombh2": 0.022274,
                "omch2": 0.11913,
                "cosmomc_theta": 0.01040867,
                "tau": tau,
                "nnu": 3.046
            },
            'stop_at_error': True,
            'debug': debug}
    scale = 1e-9
    ks = np.logspace(k_min_bin, k_max_bin, nbins)

    def pk_test(k):
        return testAs * (k / 0.05) ** (testns - 1) / scale * np.exp(-2 * tau)

    pars = {'b%s' % (b + 1): pk_test(ks[b]) for b in range(nbins)}
    model = install_test_wrapper(skip_not_installed, get_model, info)
    model.loglikes(pars)
