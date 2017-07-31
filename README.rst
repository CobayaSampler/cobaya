cobaya, a code for Bayesian analysis in Cosmology
=================================================

:Author: `Jesus Torrado`_ and `Antony Lewis`_

:Source: `Source code at Github <https://github.com/JesusTorrado/cobaya>`_

:Documentation: `Documentation at Readthedocs <https://cobaya.readthedocs.org>`_
         
:License: `LGPL <https://www.gnu.org/licenses/lgpl-3.0.en.html>`_ for the code (except otherwise stated in a docstring) and `GFDL <https://www.gnu.org/licenses/fdl-1.3.en.html>`_ for the documentation (see `LICENSE.txt <https://github.com/JesusTorrado/cobaya/blob/master/LICENSE.txt>`_).
                 
**cobaya** (Guinea Pig, in Spanish) is a framework for sampling and statistical modelling: it allows you explore any prior or posterior using a range of Monte Carlo samplers (including the advanced MCMC sampler from CosmoMC_, and the advanced nested sampler PolyChord_). The results of the sampling can be analysed with GetDist_.

Its authors are `Jesus Torrado`_ and `Antony Lewis`_, and some ideas and modules (at least in the current version) have been adapted from `Monte Python`_, by `Julien Lesgourgues`_ and `Benjamin Audren`_ (in particular, the interface to the Planck 2015 likelihoods, without much modification, and pieces of the interface to the CLASS cosmological code).

Though **cobaya** is a general purpose code, it includes interfaces to cosmological *theory codes* (CAMB_ and CLASS_) and *experimental likelihoods* (Planck 2015 for now only). The interfaces to most cosmological likelihoods are agnostic as to which theory code is used to compute the observables, which facilitates comparison between those codes.

**cobaya** has been conceived from the beginning to be highly and quickly extensible -- without touching **cobaya**'s source, you can define your own priors and likelihoods, and modify your theory code. With a little more work, you can experiment with new sampling techniques. It supports MPI parallellisation, it can be run either from the shell or from a Python interpreter/notebook, and takes input either as YAML_ files or Python dictionaries.

How to cite us
--------------

As of this version, there is no scientific publication yet associated to this software, so simply mention its Github repository (see above).

To appropriately cite the modules (samplers, theory codes, likelihoods) that you have used, simply run the script `cobaya-citation` with your input file(s) as argument(s), and you will get *bibtex* references and a short suggested text snippet for each module mentioned in your input file. 

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
