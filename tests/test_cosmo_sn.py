from test_cosmo_planck_2015 import params_lowTEB_highTTTEEE
from cosmo_common import body_of_test, baseline_cosmology
from cobaya.yaml import yaml_load


## IS THE VALUE FOR THE TOLERANCE GOOD???


# Pantheon (alpha and beta not used - no nuisance parameters), fast
def test_sn_pantheon_camb(modules):
    lik = "sn_pantheon"
    info_likelihood = {lik: None}
    info_theory = {"camb": None}
    body_of_test(modules, best_fit, info_likelihood, info_theory, chi2_sn_pantheon)


    # # JLA with alpha, beta parameters passed in, fairly fast (one matrix inversion)
    # like = SN_likelihood(r'C:\Work\Dist\git\cosmomcplanck\data\jla.dataset', marginalize=False)
    # zs = like.get_redshifts()
    # start = time.time()
    # chi2 = like.loglike(fit(zs), {'alpha': 0.1325237, 'beta': 2.959805}) * 2
    # print('Likelihood execution time:', time.time() - start)
    # print('JLA chi^2: %.2f, expected 716.23' % chi2)
    # assert np.isclose(chi2, 716.2296141)
    # print('')

    # # JLA marginalized over alpha, beta, e.g. for use in importance sampling with no nuisance parameters.
    # # Quite fast as inverses precomputed. Note normalization is not same as for alpha, beta varying.
    # like = SN_likelihood(r'C:\Work\Dist\git\cosmomcplanck\data\jla.dataset', marginalize=True)
    # zs = like.get_redshifts()
    # start = time.time()
    # chi2 = like.loglike(fit(zs)) * 2
    # print('Likelihood execution time:', time.time() - start)
    # print('JLA marged chi^2: %.2f, expected 720.00' % chi2)
    # assert np.isclose(chi2, 720.0035394)

    # # as above, but very slow (but lower memory) using non-precomputed inverses (and non-threaded in python)
    # like = SN_likelihood(r'C:\Work\Dist\git\cosmomcplanck\data\jla.dataset', precompute_covmats=False, marginalize=True)
    # zs = like.get_redshifts()
    # start = time.time()
    # chi2 = like.loglike(fit(zs)) * 2
    # print('Likelihood execution time:', time.time() - start)
    # print('JLA marged chi^2: %.2f, expected 720.00' % chi2)
    # assert np.isclose(chi2, 720.0035394)


# BEST FIT AND REFERENCE VALUES ##########################################################

best_fit = yaml_load(baseline_cosmology)
best_fit.update({k:v for k,v in params_lowTEB_highTTTEEE.items()
                 if k in baseline_cosmology})
best_fit.update({"alpha": 0.1325237, "beta": 2.959805})

chi2_sn_pantheon = {"sn_pantheon": 1054.557083, "tolerance": 0.1}
