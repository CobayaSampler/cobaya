# Optional: define an output driver
from cobaya.output import get_output
out = get_output(output_prefix="chains/my_model", resume=False, force=True)

# Initialise and run the sampler
info_sampler = {"mcmc": {"burn_in": 0, "max_samples": 1}}
from cobaya.sampler import get_sampler
mcmc = get_sampler(info_sampler, model=model, output=out,
                   packages_path=info["packages_path"])
mcmc.run()

# Print results
print(mcmc.products()["sample"]) 
