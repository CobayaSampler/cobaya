from .conftest import install_test_wrapper
import numpy as np
from cobaya.model import get_model
from cobaya.log import LoggedError, NoLogging

import pytest
import logging

cosmology_params = {
    'ombh2': 0.022, 'omch2': 0.12,
    'H0': 68,
    'ns': 0.96,
    'mnu': 0.06, 'nnu': 3.046
}


def _get_model(params, skip_not_installed):
    info = {
        'params': params,
        'likelihood':
            {'one':
                 {'requires':
                      {"Pk_grid":
                           {'k_max': 10, 'z': 0},
                       "As": None,
                       "sigma8": None
                       }
                  }
             },
        'theory': {'camb': {'stop_at_error': True,
                            'extra_args': {'num_massive_neutrinos': 1,
                                           'halofit_version': 'mead'}}}}
    return install_test_wrapper(skip_not_installed, get_model, info)


def test_CAMB_sigma8_input(skip_not_installed):
    power_params_s8 = {
        'sigma8': 0.78,
        'omegam': None,
        's8': {'derived': 'lambda sigma8, omegam: sigma8*np.sqrt(omegam/0.3)'},
        'As': None,
    }
    model_s8 = _get_model(
        {**cosmology_params, **power_params_s8},
        skip_not_installed)
    model_s8.loglike({})

    k, z, pk_s8 = model_s8.provider.get_Pk_grid()
    As_from_sigma8 = model_s8.provider.get_param("As")
    sigma8 = model_s8.provider.get_param("sigma8")

    model_as = _get_model(
        {**cosmology_params, "As": As_from_sigma8},
        skip_not_installed)
    model_as.loglike({})

    k, z, pk_as = model_as.provider.get_Pk_grid()

    assert np.isclose(sigma8, model_as.provider.get_param("sigma8"))
    assert np.allclose(pk_s8, pk_as)


def test_CAMB_As_and_sigma8_input_error(skip_not_installed):
    power_params_s8 = {
        'sigma8': 0.78,
        'omegam': None,
        's8': {'derived': 'lambda sigma8, omegam: sigma8*np.sqrt(omegam/0.3)'},
        'As': 2.1e-9,
    }

    with pytest.raises(LoggedError), NoLogging(logging.ERROR):
        model_s8 = _get_model(
            {**cosmology_params, **power_params_s8},
            skip_not_installed)
        model_s8.loglike({})
