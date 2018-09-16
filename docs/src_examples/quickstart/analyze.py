# Export the results to GetDist
from getdist.mcsamples import loadCobayaSamples
gd_sample = loadCobayaSamples(updated_info, products["sample"])
# Analyze and plot
print("Mean:")
print(gd_sample.getMeans()[:2])
print("Covariace matrix:")
print(gd_sample.getCovMat().matrix[:2,:2])
# %matplotlib inline  # uncomment if running from the Jupyter notebook
import getdist.plots as gdplt
gdplot = gdplt.getSubplotPlotter()
gdplot.triangle_plot(gd_sample, ["a", "b"], filled=True)
