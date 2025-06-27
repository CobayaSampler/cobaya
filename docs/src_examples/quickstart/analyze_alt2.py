# Just plotting (loading on-the-fly)
# Notice that GetDist requires a full path when loading samples
import os

import getdist.plots as gdplt

folder, name = os.path.split(os.path.abspath(info["output"]))
gdplot = gdplt.get_subplot_plotter(chain_dir=folder)
gdplot.triangle_plot(name, ["a", "b"], filled=True)
