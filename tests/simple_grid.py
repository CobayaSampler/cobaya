# sample grid parameter file
# build grid with "cobaya-grid-create grid_dir simple_grid.py"

import numpy as np
from cobaya import InputDict

default: InputDict = {
    'params': {'a_0': {'prior': {'min': -1, 'max': 1}, 'latex': '\\alpha_{0}'},
               'a_1': 0.1,
               'a_2': 0.64},
    'sampler': {'mcmc': {'max_samples': 500, 'burn_in': 100, 'covmat': None}},
}

defaults = [default]

# settings for the variation of each parameter that is varied in the grid
params = {'a_1': {'prior': {'min': -1, 'max': 1}, 'latex': '\\alpha_{1}'},
          'a_2': {'prior': {'min': -1, 'max': 1}, 'latex': '\\alpha_{2}'}}

# Additional (non-params) options to use when each parameter is varied
params_extra_opts = {'a_2': {
    'sampler': {'mcmc': {'max_samples': 100}}}}

# note that must use explicit "class" parameters,
# so when using two gaussian_mixture likelihoods at the same time the names are distinct
like1: InputDict = {
    'likelihood':
        {'mix1': {
            'class': 'gaussian_mixture',
            'means': [np.array([-0.48591462, 0.10064559, 0.64406749])],
            'covs': [np.array([[0.00078333, 0.00033134, -0.0002923],
                               [0.00033134, 0.00218118, -0.00170728],
                               [-0.0002923, -0.00170728, 0.00676922]])],
            'input_params_prefix': 'a'}}}

like2: InputDict = {
    'likelihood':
        {'mix2': {
            'class': 'gaussian_mixture',
            'means': [np.array([0])],
            'covs': [0.01],
            'input_params_prefix': 'b'}},
    'params': {'b_0': {'prior': {'min': -1, 'max': 1}, 'latex': '\\beta_{0}'}}
}

groups = {
    'main': {
        'params': [[], ['a_1'], ['a_2'], ['a_1', 'a_2']],
        'datasets': [('like1', like1), (['like1', 'like2'], [like1, like2])]
    }
}
