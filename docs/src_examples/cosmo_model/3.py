point = dict(zip(model.parameterization.sampled_params(),
                 model.prior.sample(ignore_external=True)[0]))

point.update({'omega_b': 0.0223, 'omega_cdm': 0.120, 'H0': 67.01712,
              'logA': 3.06, 'n_s': 0.966, 'tau_reio': 0.065})

logposterior = model.logposterior(point)
logposterior_dict = logposterior.as_dict(model)
print('Full log-posterior:')
print('   logposterior:', logposterior_dict["logpost"])
print('   logpriors:', logposterior_dict["logpriors"])
print('   loglikelihoods:', logposterior_dict["loglikes"])
print('   derived params:', logposterior_dict["derived"])
