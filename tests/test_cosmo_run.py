import os
import numpy as np
from getdist.mcsamples import MCSamplesFromCobaya, loadMCSamples
from cobaya.theory import Theory
from cobaya.run import run
from cobaya import mpi
from cobaya.conventions import _packages_path, InfoDict, _dill_extension
from .common import process_packages_path


def likelihood(_self):
    return -(_self.provider.get_param('sigma8') - 0.7) ** 2 / 0.1 ** 2


def likelihood2(sigma8, joint):
    return -(sigma8 - 0.75) ** 2 / 0.07 ** 2 + joint * 0


class ATheory(Theory):
    params = {'As': None, 'As100': {'derived': True}}

    def calculate(self, state, want_derived=True, **params_values_dict):
        state['derived']['As100'] = params_values_dict['As'] * 100


info: InfoDict = {"params": {
    "ombh2": 0.0245,
    "H0": 70,
    "ns": 0.965,
    "logA": {'prior': [1, 4], 'latex': r'\log(10^{10} A_\mathrm{s})', "proposal": 0.1},
    "As": {'value': lambda logA: 1e-10 * np.exp(logA), 'latex': r'A_\mathrm{s}'},
    "omegab": {'value': lambda ombh2, H0: ombh2 / (H0 / 100) ** 2,
               'latex': r'\Omega_\mathrm{b}'},
    "omegam": {'value': 0.3, 'latex': r'\Omega_\mathrm{m}', 'drop': True},
    "omch2": {'value': lambda omegam, omegab, mnu, H0: (omegam - omegab) * (
            H0 / 100) ** 2 - (mnu * (3.046 / 3) ** 0.75) / 94.0708,
              'latex': r'\Omega_\mathrm{c} h^2'},
    "tau": 0.05,
    "mnu": 0.0,
    "sigma8": None,
    "sigma82": {"derived": "lambda sigma8, omegam: sigma8 ** 2 * omegam"},
    "joint": {"derived": "lambda sigma8, omegam, As100: sigma8 ** 2 * omegam*As100"}},
    "likelihood": {
        "test_likelihood": {'external': likelihood,
                            'type': 'type1',
                            'requires': ['As', 'sigma8']}},
    'theory': {'test_theory': ATheory,
               'camb': {'stop_at_error': True,
                        'extra_args': {'num_massive_neutrinos': 1,
                                       'halofit_version': 'mead'}}},
    'sampler': {'mcmc': {'Rminus1_stop': 0.3}}
}


@mpi.sync_errors
def test_cosmo_run_resume_post(tmpdir, packages_path=None):
    # only vary As, so fast chain
    info['output'] = os.path.join(tmpdir, 'testchain')

    if packages_path:
        info[_packages_path] = process_packages_path(packages_path)
    run(info, force=True)
    # note that continuing from files leads to text-file precision at ead in, so a mix of
    # precision in the output Collection returned from run
    run(info, resume=True, override={'sampler': {'mcmc': {'Rminus1_stop': 0.2}}})
    updated_info, sampler = run(info['output'] + '.updated' + _dill_extension,
                                resume=True,
                                override={'sampler': {'mcmc': {'Rminus1_stop': 0.05}}})
    products = sampler.products()
    results = mpi.allgather(products["sample"])
    samp = MCSamplesFromCobaya(updated_info, results, ignore_rows=0.2)

    assert np.isclose(samp.mean('As100'), 100 * samp.mean('As'))
    assert abs(samp.mean('sigma8') - 0.69) < 0.02

    info_post = {'add': {'params': {'h': None},
                         "likelihood": {"test_likelihood2": likelihood2}},
                 'remove': {'likelihood': ["test_likelihood"]},
                 'suffix': 'testpost',
                 'skip': 0.2, 'thin': 4
                 }
    output_info, products = run(updated_info, override={'post': info_post}, force=True)
    results2 = mpi.allgather(products["sample"])
    samp2 = MCSamplesFromCobaya(output_info, results2)
    assert abs(samp2.mean('sigma8') - 0.75) < 0.02

    # from save info, no output
    info_post['output'] = None
    output_info, products = run({'output': info['output'], 'post': info_post}, force=True)
    results3 = mpi.allgather(products["sample"])
    samp3 = MCSamplesFromCobaya(output_info, results3)
    assert abs(samp3.mean('sigma8') - 0.75) < 0.02
    assert np.isclose(samp3.mean("joint"), samp2.mean("joint"))
    samps4 = loadMCSamples(info['output'] + '.post.testpost')
    assert np.isclose(samp3.mean("joint"), samps4.mean("joint"))

    # test recover original answer swapping likelihoods back
    info_revert = {'add': {'likelihood': info['likelihood']},
                   'remove': {'likelihood': ["test_likelihood2"]},
                   'suffix': 'revert',
                   'skip': 0, 'thin': 1,
                   'output': None
                   }
    output_info, products = run({'output': info['output'] + '.post.testpost',
                                 'post': info_revert}, force=True)
    results_revert = mpi.allgather(products["sample"])
    samp_revert = MCSamplesFromCobaya(output_info, results_revert)
    samp.weighted_thin(4)
    assert samp.numrows == samp_revert.numrows
    assert np.isclose(samp_revert.mean("sigma8"), samp.mean("sigma8"))

    # no remove
    info_post = {
        'add': {'params': {'h': None}, "likelihood": {"test_likelihood2": likelihood2}},
        'suffix': 'test2', 'skip': 0.2, 'thin': 4}
    output_info, products = run(updated_info, override={'post': info_post}, force=True)
    results2 = mpi.allgather(products["sample"])
    samp2 = MCSamplesFromCobaya(output_info, results2)
    assert "chi2__type1" in samp2.paramNames.list()
    # check what has been saved to disk is consistent
    samps4 = loadMCSamples(updated_info['output'] + '.post.test2')
    assert samp2.paramNames.list() == samps4.paramNames.list()
    assert np.isclose(samp2.mean("sigma8"), samps4.mean("sigma8"))
