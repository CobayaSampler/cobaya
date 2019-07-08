# Export the results to GetDist
from getdist.mcsamples import loadMCSamples
# Notice loadMCSamples requires a *full path*
import os

gd_sample = loadMCSamples(os.path.abspath(info["output"]))
# Analyze and plot
# [Exactly the same here...]
