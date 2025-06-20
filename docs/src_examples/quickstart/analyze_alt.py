# Export the results to GetDist
from cobaya import load_samples

gd_sample = load_samples(info["output"], to_getdist=True)

# Analyze and plot
# [Exactly the same here...]
