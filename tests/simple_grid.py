# sample grid parameter file
# build grid with "cobaya-grid-create grid_dir simple_grid.py"

import numpy as np
from cobaya import InputDict
from cobaya.grid_tools.batchjob import DataSet

# optional directory (or list) to look for pre-computed covmats with similar parameters
cov_dir = ""
skip = ['base_a_1_like1_like2']

default: InputDict = {
    'params': {'a_0': {'prior': {'min': -4, 'max': 2}, 'ref': -1, 'latex': '\\alpha_{0}'},
               'a_1': 0.1,
               'a_2': 1.1},
    'sampler': {'mcmc': {'max_samples': 500, 'burn_in': 100, 'covmat': 'auto'}},
}

defaults = [default]

# settings for the variation of each parameter that is varied in the grid
params = {'a_1': {'prior': {'min': -2, 'max': 2}, 'latex': '\\alpha_{1}'},
          'a_2': {'prior': {'min': -1, 'max': 3}, 'latex': '\\alpha_{2}'}}

# Additional (non-params) options to use when each parameter is varied
param_extra_opts = {'a_2': {
    'sampler': {'mcmc': {'max_samples': 100}}}}

# note that must use explicit "class" parameters,
# so when using two gaussian_mixture likelihoods at the same time the names are distinct
like1: InputDict = {
    'likelihood':
        {'mix1': {
            'class': 'gaussian_mixture',
            'means': [np.array([-1, 0, 1])],
            'covs': [np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])],
            'input_params_prefix': 'a'}}}

like2: InputDict = {
    'likelihood':
        {'mix2': {
            'class': 'gaussian_mixture',
            'means': [np.array([0])],
            'covs': [0.1],
            'input_params_prefix': 'b'}},
    'params': {'b_0': {'prior': {'min': -1, 'max': 1}, 'latex': '\\beta_{0}'}}
}

joint = DataSet(['like1', 'like2'], [like1, like2])

groups = {
    'main': {
        'params': [[], ['a_1'], ['a_2'], ['a_1', 'a_2']],
        'datasets': [('like1', like1), (['like1', 'like2'], [like1, like2])]
    }
}
