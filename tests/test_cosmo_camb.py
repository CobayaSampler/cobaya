from __future__ import absolute_import
from .common import process_modules_path
import os
import numpy as np
from cobaya.model import get_model
from cobaya.tools import load_module

params = {'ombh2': 0.02242, 'omch2': 0.11933, 'H0': 67.66, 'tau': 0.0561,
          'mnu': 0.06, 'nnu': 3.046, 'num_massive_neutrinos': 1, 'ns': 0.9665,
          'YHe': 0.2454, 'As': 2e-9}


def get_camb(modules):
    return load_module("camb",
                       path=os.path.join(process_modules_path(modules), "code", "CAMB"))


def test_sources(modules):
    camb = get_camb(modules)
    from camb.sources import GaussianSourceWindow

    pars = camb.set_params(**params)
    pars.set_for_lmax(500)
    pars.SourceWindows = [
        GaussianSourceWindow(redshift=0.17, source_type='counts', bias=1.2, sigma=0.04)]
    results = camb.get_results(pars)
    dic = results.get_source_cls_dict()

    def test_likelihood(
            _theory={'source_Cl': {'sources': {'source1':
                                                   {'function': 'gaussian',
                                                    'source_type': 'counts', 'bias': 1.2,
                                                    'redshift': 0.17, 'sigma': 0.04}},
                                   'limber': True, 'lmax': 500}}):
        assert abs(_theory.get_source_Cl()[('source1', 'source1')][100] / dic['W1xW1'][
            100] - 1) < 0.001, \
            "CAMB gaussian source window results do not match"
        return 0

    info = {
        'params': params,
        'likelihood': {'test_likelihood': test_likelihood},
        'theory': {'camb': {'stop_at_error': True}}
    }

    model = get_model(info)
    model.loglike({})


def test_CAMBdata(modules):
    def test_likelihood(
            _theory={'CAMBdata': None, 'Pk_grid': dict(k_max=2, z=[0, 2])}):
        return _theory.get_CAMBdata().tau0

    info = {
        'params': params,
        'likelihood': {'test_likelihood': test_likelihood},
        'theory': {'camb': {'stop_at_error': True}},
        'modules': process_modules_path(modules)}

    model = get_model(info)
    assert np.isclose(model.loglike({})[0], 14165.63, rtol=1e-4), \
        "CAMBdata object result failed"


def test_CAMB_transfer(modules):
    camb = get_camb(modules)

    pars = camb.set_params(**params)
    pars.set_matter_power(redshifts=[0, 2], kmax=2)
    pars.WantCls = False
    results = camb.get_results(pars)
    k, z, PK1 = results.get_nonlinear_matter_power_spectrum(hubble_units=False)

    def test_likelihood(
            _theory={'Pk_grid': dict(k_max=2, z=[0, 2])}):
        k, z, PK = _theory.get_Pk_grid()
        assert np.isclose(PK[1, 30], 10294.3285)
        np.testing.assert_allclose(PK, PK1, rtol=1e-4)
        return 1

    info = {'params': params,
            'likelihood': {'test_likelihood': test_likelihood},
            'theory': {'camb': {'stop_at_error': True}},
            'modules': process_modules_path(modules)}

    model = get_model(info)
    model.loglike()
