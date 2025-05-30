from cobaya.model import get_model

model = get_model("sample_r_theta.yaml")

# Get one random point (select list element [0]).
# When sampling a random point, we need to ignore the Jacobian and
# the gaussian band, since Cobaya doesn't know how to sample from them
random_point = model.prior.sample(ignore_external=True)[0]

# Our random point is now an array. Turn it into a dictionary:
sampled_params_names = model.parameterization.sampled_params()
random_point_dict = dict(zip(sampled_params_names, random_point))
print("")
print("Our random point is:", random_point_dict)

# Let's print some probabilities for our random point
print("The log-priors are:", model.logpriors(random_point_dict, as_dict=True))
print(
    "The log-likelihoods and derived parameters are:",
    model.loglikes(random_point_dict, as_dict=True),
)
print("The log-posterior is:", model.logpost(random_point_dict))

print("")
print("You can also get all that information at once!")
posterior_dict = model.logposterior(random_point_dict, as_dict=True)
for k, v in posterior_dict.items():
    print(k, ":", v)

import matplotlib.pyplot as plt
import numpy as np

rs = np.linspace(0.75, 1.25, 200)
loglikes = [model.loglike({"r": r, "theta": np.pi / 4}, return_derived=False) for r in rs]
plt.figure()
plt.plot(rs, np.exp(loglikes))
plt.savefig("model_slice.png")

# Optional: define an output driver
from cobaya.output import get_output

out = get_output(prefix="chains/my_model", resume=False, force=True)

# Initialise and run the sampler (low number of samples, as an example)
info_sampler = {"mcmc": {"max_samples": 100}}
from cobaya.sampler import get_sampler

mcmc = get_sampler(info_sampler, model=model, output=out)
mcmc.run()

# Print results
print(mcmc.products()["sample"])
