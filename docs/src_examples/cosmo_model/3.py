point = dict(
    zip(
        model.parameterization.sampled_params(),
        model.prior.sample(ignore_external=True)[0],
    )
)

point.update(
    {
        "omega_b": 0.0223,
        "omega_cdm": 0.120,
        "H0": 67.01712,
        "logA": 3.06,
        "n_s": 0.966,
        "tau_reio": 0.065,
    }
)

logposterior = model.logposterior(point, as_dict=True)
print("Full log-posterior:")
print("   logposterior:", logposterior["logpost"])
print("   logpriors:", logposterior["logpriors"])
print("   loglikelihoods:", logposterior["loglikes"])
print("   derived params:", logposterior["derived"])
