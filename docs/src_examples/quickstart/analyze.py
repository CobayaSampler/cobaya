# Export the results to GetDist
from getdist.mcsamples import loadCobayaSamples
#print updated_info
#print products["sample"]
gd_sample = loadCobayaSamples(updated_info, products["sample"])
# Analyze and plot
import getdist.plots as gdplt
print("Mean:")
print(gd_sample.getMeans()[:2])
print("Covariace matrix:")
print(gd_sample.getCovMat().matrix[:2,:2])
# %matplotlib inline  # uncomment if running from the Jupyter notebook
import getdist.plots as gdplot
gdplot = gdplt.getSubplotPlotter()
gdplot.triangle_plot(gd_sample, ["a", "b"], filled=True)
