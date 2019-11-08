def test_sources():
    import camb
    from camb.sources import GaussianSourceWindow

    params = {'ombh2': 0.02242, 'omch2': 0.11933, 'H0': 67.66, 'tau': 0.0561,
              'mnu': 0.06, 'nnu': 3.046, 'num_massive_neutrinos': 1, 'ns': 0.9665, 'YHe': 0.2454, 'As': 2e-9}

    pars = camb.set_params(**params)
    pars.set_for_lmax(500)
    pars.SourceWindows = [GaussianSourceWindow(redshift=0.17, source_type='counts', bias=1.2, sigma=0.04)]
    results = camb.get_results(pars)
    dic = results.get_source_cls_dict()

    def test_likelihood(
            _theory={'source_Cl': {'sources': {'source1':
                                                   {'function': 'gaussian', 'source_type': 'counts', 'bias': 1.2,
                                                    'redshift': 0.17, 'sigma': 0.04}},
                                   'limber': True, 'lmax': 500}}):
        assert abs(_theory.get_source_Cl()[('source1', 'source1')][100] / dic['W1xW1'][100] - 1) < 0.001, \
            "CAMB gaussian source window results do not match"
        return 0

    info = {
        'params': params,
        'likelihood': {'test_likelihood': test_likelihood},
        'theory': {'camb': {'stop_at_error': True}}}

    from cobaya.model import get_model
    model = get_model(info)
    model.loglike({})
