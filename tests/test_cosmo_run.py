import logging
import os
import numpy as np
import pytest
from getdist.mcsamples import MCSamplesFromCobaya, loadMCSamples
from cobaya import mpi, run, Theory, InputDict, PostDict, LoggedError
from cobaya.conventions import Extension
from cobaya.component import ComponentNotFoundError
from cobaya.tools import deepcopy_where_possible
from cobaya.cosmo_input.convert_cosmomc import cosmomc_root_to_cobaya_info_dict
from cobaya.log import NoLogging
from .common import process_packages_path

pytestmark = pytest.mark.mpi


def likelihood(_self):
    return -(_self.provider.get_param('sigma8') - 0.7) ** 2 / 0.1 ** 2


def likelihood2(sigma8, joint):
    return -(sigma8 - 0.75) ** 2 / 0.07 ** 2 + joint * 0


class ATheory(Theory):
    params = {'As': None, 'As100': {'derived': True}}

    def calculate(self, state, want_derived=True, **params_values_dict):
        state['derived']['As100'] = params_values_dict['As'] * 100


class BTheory(Theory):
    params = {'As100': None, 'As1000': {'derived': True}}

    def calculate(self, state, want_derived=True, **params_values_dict):
        state['derived']['As1000'] = params_values_dict['As100'] * 10


class CTheory(Theory):
    params = {'AsX': None, 'As1000': {'derived': True}}


info: InputDict = {"params": {
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


def test_cosmo_run_not_found():
    with NoLogging(logging.ERROR):
        inf = deepcopy_where_possible(info)
        inf["likelihood"]["H0.perfect"] = None
        with pytest.raises(ComponentNotFoundError):
            run(inf)
        inf = deepcopy_where_possible(info)
        inf["likelihood"]["none"] = None
        with pytest.raises(ComponentNotFoundError):
            run(inf)
        inf = deepcopy_where_possible(info)
        inf["likelihood"]["pandas.plotting.PlotAccessor"] = None
        with pytest.raises(LoggedError) as e:
            run(inf)
        assert "Failed to get defaults for component" in str(e)


@mpi.sync_errors
def test_cosmo_run_resume_post(tmpdir, packages_path=None):
    # only vary As, so fast chain. Chain does not need to converge (tested elsewhere).
    info['output'] = os.path.join(tmpdir, 'testchain')
    if packages_path:
        info["packages_path"] = process_packages_path(packages_path)
    run(info, force=True)
    # note that continuing from files leads to text-file precision at read in, so a mix of
    # precision in the output SampleCollection returned from run
    run(info, resume=True, override={'sampler': {'mcmc': {'Rminus1_stop': 0.2}}})
    updated_info, sampler = run(info['output'] + '.updated' + Extension.dill,
                                resume=True,
                                override={'sampler': {'mcmc': {'Rminus1_stop': 0.05}}})
    results = mpi.allgather(sampler.products()["sample"])
    samp = MCSamplesFromCobaya(updated_info, results, ignore_rows=0.2)
    assert np.isclose(samp.mean('As100'), 100 * samp.mean('As'))

    # post-processing
    info_post: PostDict = {'add': {'params': {'h': None},
                                   "likelihood": {"test_likelihood2": likelihood2}},
                           'remove': {'likelihood': ["test_likelihood"]},
                           'suffix': 'testpost',
                           'skip': 0.2, 'thin': 4
                           }

    output_info, products = run(updated_info, override={'post': info_post}, force=True)
    results2 = mpi.allgather(products["sample"])
    samp2 = MCSamplesFromCobaya(output_info, results2)
    samp_test = samp.copy()
    samp_test.weighted_thin(4)
    sigma8 = samp_test.getParams().sigma8
    samp_test.reweightAddingLogLikes(-(sigma8 - 0.7) ** 2 / 0.1 ** 2
                                     + (sigma8 - 0.75) ** 2 / 0.07 ** 2)
    assert np.isclose(samp_test.mean('sigma8'), samp2.mean('sigma8'))

    # from getdist-format chain files
    root = os.path.join(tmpdir, 'getdist_format')
    if mpi.is_main_process():
        samp.saveChainsAsText(root)
    mpi.sync_processes()

    from_txt = dict(updated_info, output=root)
    post_from_text = dict(info_post, skip=0)  # getdist already skipped
    output_info, products = run(from_txt, override={'post': post_from_text}, force=True)
    samp_getdist = MCSamplesFromCobaya(output_info, mpi.allgather(products["sample"]))
    assert not products["stats"]["points_removed"]
    assert samp2.numrows == samp_getdist.numrows
    assert np.isclose(samp2.mean('sigma8'), samp_getdist.mean('sigma8'))

    # again with inferred-inputs for params
    info_conv = cosmomc_root_to_cobaya_info_dict(root)
    # have to manually add consistent likelihoods if re-computing
    info_conv['likelihood'] = info['likelihood']
    info_conv['theory'] = info['theory']
    post_from_text = dict(info_post, skip=0, suffix='getdist2')  # getdist already skipped
    output_info, products = run(info_conv, override={'post': post_from_text},
                                output=False)
    samp_getdist2 = MCSamplesFromCobaya(output_info, mpi.allgather(products["sample"]))
    assert np.isclose(samp2.mean('sigma8'), samp_getdist2.mean('sigma8'))

    # from save info, no output
    info_post['output'] = None
    output_info, products = run({'output': info['output'], 'post': info_post}, force=True)
    results3 = mpi.allgather(products["sample"])
    samp3 = MCSamplesFromCobaya(output_info, results3)
    assert np.isclose(samp3.mean("sigma8"), samp2.mean("sigma8"))
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

    samp_thin = MCSamplesFromCobaya(updated_info, results, ignore_rows=0.2)
    samp_thin.weighted_thin(4)
    assert samp_thin.numrows == samp_revert.numrows + products["stats"]["points_removed"]
    if not products["stats"]["points_removed"]:
        assert np.isclose(samp_revert.mean("sigma8"), samp_thin.mean("sigma8"))
    else:
        assert abs(samp_revert.mean("sigma8") - samp_thin.mean("sigma8")) < 0.01
    assert not products["stats"]["points_removed"]

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

    # adding new theory derived
    info_post['add']['theory'] = {'new_param_theory': BTheory}
    output_info, products = run(updated_info, override={'post': info_post}, output=False)
    results3 = mpi.allgather(products["sample"])
    samp3 = MCSamplesFromCobaya(output_info, results3)
    assert np.isclose(samp3.mean("sigma8"), samp2.mean("sigma8"))
    assert np.isclose(samp3.mean("As1000"), samp2.mean("As") * 1000)

    info_post['add']['theory'] = {'new_param_theory': CTheory}
    with pytest.raises(LoggedError) as e, NoLogging(logging.ERROR):
        run(updated_info, override={'post': info_post}, output=False)
    assert 'Parameter AsX no known value' in str(e)
