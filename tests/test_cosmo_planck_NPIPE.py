import numpy as np
from cobaya.model import get_model
from cobaya.install import install
from .common import process_packages_path

yaml = r"""
theory:
  camb:
    extra_args:
      lens_potential_accuracy: 1
likelihood:
  planck_2018_lowl.TT_native: null
  planck_2018_lowl.EE_native: null
  planck_NPIPE_highl_CamSpec.TTTEEE: null
  planckpr4lensing:
    package_install:
      github_repository: carronj/planck_PR4_lensing
      min_version: 1.0.2
params:
  tau: 0.05
  logA: 3.04920413
  As:
    value: 'lambda logA: 1e-10*np.exp(logA)'
  ns: 0.96399503
  theta_MC_100: 1.04240171
  thetastar:
    value: 'lambda theta_MC_100: 1.e-2*theta_MC_100'
    derived: false
  ombh2: 0.02235048
  omch2: 0.12121379
"""


def test_planck_NPIPE_install(packages_path):
    packages_path = process_packages_path(packages_path)
    from cobaya.yaml import yaml_load
    info = yaml_load(yaml)
    install(info, path=packages_path)

    info['packages_path'] = packages_path

    model = get_model(info)
    pars = (0.99818025, 10.35947284, 18.67072461, 7.54932654, 0.83715482,
            0.94987418, 1.23385364, 0.98781552, 1.013345)

    assert np.isclose(model.logposterior(pars).logpost, -5889.873, rtol=1e-4)
