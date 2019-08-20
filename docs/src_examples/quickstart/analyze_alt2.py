# Just plotting (loading on-the-fly)
import getdist.plots as gdplt
import os

folder, name = os.path.split(os.path.abspath(info["output"]))
gdplot = gdplt.getSubplotPlotter(chain_dir=folder)
gdplot.triangle_plot(name, ["a", "b"], filled=True)
