from .common import process_packages_path
from .conftest import install_test_wrapper
import os
import pytest
import numpy as np
from cobaya.model import get_model
from cobaya.tools import load_module
from cobaya.install import NotInstalledError

params = {'ombh2': 0.02242, 'omch2': 0.11933, 'H0': 67.66, 'tau': 0.0561,
          'mnu': 0.06, 'nnu': 3.046, 'ns': 0.9665,
          'YHe': 0.2454, 'As': 2e-9}


def get_camb(packages_path):
    try:
        return load_module("camb", path=os.path.join(process_packages_path(packages_path),
                                                     "code", "CAMB"))
    except ModuleNotFoundError:
        raise NotInstalledError(None)


def _get_model(packages_path, likelihood_info, skip_not_installed):
    info = {
        'params': params,
        'likelihood': {'test_likelihood': likelihood_info},
        'theory': {'camb': {'stop_at_error': True,
                            'extra_args': {'num_massive_neutrinos': 1}}},
        'packages_path': process_packages_path(packages_path)}
    return install_test_wrapper(skip_not_installed, get_model, info)


def test_sources(packages_path, skip_not_installed):
    camb = install_test_wrapper(skip_not_installed, get_camb, packages_path)
    from camb.sources import GaussianSourceWindow

    pars = camb.set_params(**params)
    pars.set_for_lmax(500)
    pars.SourceWindows = [
        GaussianSourceWindow(redshift=0.17, source_type='counts', bias=1.2, sigma=0.04)]
    results = camb.get_results(pars)
    dic = results.get_source_cls_dict()

    # noinspection PyDefaultArgument
    def test_likelihood(_self):
        assert abs(_self.provider.get_source_Cl()[('source1', 'source1')][100] / dic['W1xW1'][
            100] - 1) < 0.001, \
            "CAMB gaussian source window results do not match"
        return 0
    test_likelihood_requires = {
        'source_Cl': {'sources': {'source1':
                                  {'function': 'gaussian',
                                   'source_type': 'counts',
                                   'bias': 1.2,
                                   'redshift': 0.17,
                                   'sigma': 0.04}},
                      'limber': True, 'lmax': 500}}

    model = _get_model(
        packages_path,
        {'external': test_likelihood, 'requires': test_likelihood_requires},
        skip_not_installed)
    model.loglike({})


def test_CAMBdata(packages_path, skip_not_installed):
    # noinspection PyDefaultArgument
    def test_likelihood(_self):
        return _self.provider.get_CAMBdata().tau0
    test_likelihood_requires = {'CAMBdata': None, 'Pk_grid': dict(k_max=2, z=[0, 2])}

    model = _get_model(
        packages_path,
        {'external': test_likelihood, 'requires': test_likelihood_requires},
        skip_not_installed)
    assert np.isclose(model.loglike({})[0], 14165.63, rtol=1e-4), \
        "CAMBdata object result failed"


def test_CAMB_transfer(packages_path, skip_not_installed):
    camb = install_test_wrapper(skip_not_installed, get_camb, packages_path)
    pars = camb.set_params(**params)
    pars.set_matter_power(redshifts=[0, 2], kmax=2)
    pars.WantCls = False
    results = camb.get_results(pars)
    k, z, PK1 = results.get_nonlinear_matter_power_spectrum(hubble_units=False)

    # noinspection PyDefaultArgument,PyUnresolvedReferences
    def test_likelihood(_self):
        _, _, PK = _self.provider.get_Pk_grid()
        assert np.isclose(PK[1, 30], 10294.3285)
        np.testing.assert_allclose(PK, PK1, rtol=1e-4)
        return 1
    test_likelihood_requires = {'Pk_grid': dict(k_max=2, z=[0, 2])}

    model = _get_model(
        packages_path,
        {'external': test_likelihood, 'requires': test_likelihood_requires},
        skip_not_installed)
    model.loglike()


def test_CAMB_sigma_R(packages_path, skip_not_installed):
    camb = install_test_wrapper(skip_not_installed, get_camb, packages_path)
    pars = camb.set_params(**params)
    redshifts = [0, 2, 5]
    pars.set_matter_power(redshifts=redshifts, kmax=2)
    pars.WantCls = False
    results = camb.get_results(pars)
    R = np.arange(1, 20, 1)
    sigma_R = results.get_sigmaR(R=R, hubble_units=False)[::-1, :]

    # noinspection PyDefaultArgument,PyUnresolvedReferences
    def test_likelihood(_self):
        r_out, z_out, sigma_R_out = _self.provider.get_sigma_R()
        assert np.allclose(z_out, redshifts)
        np.testing.assert_allclose(sigma_R, sigma_R_out, rtol=1e-3)
        return 1
    test_likelihood_requires = {'sigma_R': dict(z=[0, 2, 5], R=R),
                                'Pk_grid': {'k_max': 1, 'z': np.arange(0.2, 6, 1)}}

    model = _get_model(
        packages_path,
        {'external': test_likelihood, 'requires': test_likelihood_requires},
        skip_not_installed)
    model.loglike()
