cobaya, a Monte Carlo (cosmological) sampler
---------------------------------------------

:Author: `Jesus Torrado`_ and `Antony Lewis`_

:Info: See https://github.com/JesusTorrado/cobaya for the version-controlled source.

:License: [PROVISIONAL!] `LGPL <https://www.gnu.org/copyleft/lesser.html>`_
                 
cobaya is a generic *sampling infrastructure:* it allows you explore any posterior using a range of samplers (including the advanced MCMC sampler from CosmoMC_, and the advanced nested sampler PolyChord_). The results of the sampling can be analysed with GetDist_.

Its authors are `Jesus Torrado`_ and `Antony Lewis`_, and some ideas and modules (at least in the current version) have been adapted from `Monte Python`_, by `Julien Lesgourgues`_ and `Benjamin Audren`_ (in particular, the interface to the Planck 2015 likelihoods, without much modification, and pieces of the interface to the CLASS cosmological code).

It is equipped with a number of cosmological *theory codes* (CAMB_, CLASS_, and in the future Pico_) and *experimental likelihoods* (Planck 2015 for now only). The experimental likelihoods are agnostic to the theory codes with which the observables are computed, which facilitates comparisons between those codes.

cobaya has been conceived from the beginning to be highly and quickly extensible -- you can define your own priors and likelihoods easily, modify your theory code without fearing breaking the run down, or experiment with new sampling techniques. It can be called from a Python interpreter in an scripted way, and its input files can be automatically generated from Python dictionaries.

How to cite us
--------------

As of this version, there is no scientific publication associated directly to this software, so simply mention its Github repository (see above).

To appropriately cite the modules (samplers, theory codes, likelihoods), simply run the script `cobaya-citation` with your input file as an argument, and you will get *bibtex* references and a short suggested text snippet for each module mentioned in your input file. 

.. _`Jesus Torrado`: http://astronomy.sussex.ac.uk/~jt386
.. _`Antony Lewis`: http://cosmologist.info
.. _CosmoMC: http://cosmologist.info/cosmomc/
.. _`Monte Python`: http://baudren.github.io/montepython.html
.. _`Julien Lesgourgues`: https://www.particle-theory.rwth-aachen.de/cms/Particle-Theory/Das-Institut/Mitarbeiter-TTK/Professoren/~gufe/Lesgourgues-Julien/?lidx=1
.. _`Benjamin Audren`: http://baudren.github.io/
.. _Class: http://class-code.net/
.. _Camb: http://camb.info/
.. _Pico: http://cosmos.astro.illinois.edu/pico/
.. _GetDist: https://github.com/cmbant/getdist
.. _YAML: https://en.wikipedia.org/wiki/YAML
.. _PolyChord: http://ccpforge.cse.rl.ac.uk/gf/project/polychord
