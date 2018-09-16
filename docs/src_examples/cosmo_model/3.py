point = dict(zip(model.parameterization.sampled_params(),
                 model.prior.sample(ignore_external=True)[0]))

point.update({'omega_b': 0.0223, 'omega_cdm': 0.120, 'H0': 67.01712,
              'logA': 3.06, 'n_s': 0.966, 'tau_reio': 0.065})

logposterior = model.logposterior(point)
print('Full log-posterior:')
print('   logposterior: %g' % logposterior.logpost)
print('   logpriors: %r' % dict(zip(list(model.prior), logposterior.logpriors)))
print('   loglikelihoods: %r' % dict(zip(list(model.likelihood), logposterior.loglikes)))
print('   derived params: %r' % dict(zip(list(model.parameterization.derived_params()), logposterior.derived)))
