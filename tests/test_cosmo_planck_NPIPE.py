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
  logA:
    prior:
      min: 1.61
      max: 3.91
    ref:
      dist: norm
      loc: 3.05
      scale: 0.001
    proposal: 0.001
    latex: \log(10^{10} A_\mathrm{s})
    drop: true
  As:
    value: 'lambda logA: 1e-10*np.exp(logA)'
    latex: A_\mathrm{s}
  ns:
    prior:
      min: 0.8
      max: 1.2
    ref:
      dist: norm
      loc: 0.965
      scale: 0.004
    proposal: 0.002
    latex: n_\mathrm{s}
  theta_MC_100:
    prior:
      min: 0.5
      max: 10
    ref:
      dist: norm
      loc: 1.04109
      scale: 0.0004
    proposal: 0.0002
    latex: 100\theta_\mathrm{MC}
    drop: true
    renames: theta
  cosmomc_theta:
    value: 'lambda theta_MC_100: 1.e-2*theta_MC_100'
    derived: false
  H0:
    latex: H_0
    min: 20
    max: 100
  ombh2:
    prior:
      min: 0.005
      max: 0.1
    ref:
      dist: norm
      loc: 0.0224
      scale: 0.0001
    proposal: 0.0001
  omch2:
    prior:
      min: 0.001
      max: 0.99
    ref:
      dist: norm
      loc: 0.12
      scale: 0.001
    proposal: 0.0005
"""


def test_planck_NPIPE(packages_path, skip_not_installed):
    packages_path = process_packages_path(packages_path)
    from cobaya.yaml import yaml_load
    info = yaml_load(yaml)
    install(info, path=packages_path)

    info['packages_path'] = packages_path

    model = get_model(info)
    pars = [3.04920413, 0.96399503, 1.04240171, 0.02235048, 0.12121379,
            0.99818025, 10.35947284, 18.67072461, 7.54932654, 0.83715482,
            0.94987418, 1.23385364, 0.98781552, 1.013345]

    print(model.logposterior(pars))
    assert np.isclose(model.logposterior(pars).logpost, -6584.6479, rtol=1e-4)
