*cobaya*, a code for Bayesian analysis in Cosmology
===================================================

:Author: `Jesus Torrado`_ and `Antony Lewis`_

:Source: `Source code at GitHub <https://github.com/JesusTorrado/cobaya>`_

:Documentation: `Documentation at Readthedocs <https://cobaya.readthedocs.org>`_

:Licensing: `LGPL <https://www.gnu.org/licenses/lgpl-3.0.en.html>`_ (source) and `GFDL <https://www.gnu.org/licenses/fdl-1.3.en.html>`_ (documentation) – exceptions: see `LICENCE.txt <https://github.com/JesusTorrado/cobaya/blob/master/LICENCE.txt>`_

**cobaya** (COde for BAYesian Analysis; Guinea Pig, in Spanish) is a framework for sampling and statistical modelling: it allows you to explore any prior or posterior using a range of Monte Carlo samplers (including the advanced MCMC sampler from CosmoMC_, and the advanced nested sampler PolyChord_). The results of the sampling can be analysed with GetDist_.

Its authors are `Jesus Torrado`_ and `Antony Lewis`_. Some ideas and modules have been adapted from `Monte Python`_, by `Julien Lesgourgues`_ and `Benjamin Audren`_ (in particular, the interface to the Planck 2015 likelihoods).

Though **cobaya** is a general purpose statistical framework, it includes interfaces to cosmological *theory codes* (CAMB_ and CLASS_) and *experimental likelihoods* (in this version, Planck 2015 and Bicep-Keck 2015). The interfaces to most cosmological likelihoods are agnostic as to which theory code is used to compute the observables, which facilitates comparison between those codes.

**cobaya** has an overhead ~0.5ms per posterior evaluation, which makes it suitable for most cosmological applications (CAMB_ and CLASS_ take seconds to run), but not necessarily for more general statistical applications, if the evaluation time per pdf involved is of that order or smaller.

**cobaya** has been conceived from the beginning to be highly and effortlessly extensible: without touching **cobaya**'s source code, you can define your own priors and likelihoods, use modified versions of a theory code (just pass **cobaya** their folder!), create new parameters as function of other parameters... You can also use **cobaya** simply as a wrapper for cosmological likelihoods to integrate them in your analysis pipeline.

In addition, it downloads and install requirements automatically (samplers, theory codes and likelihoods); it supports MPI parallelization and HPC containerization (Singularity and Docker+Shifter; the later is WIP); and it can be run either from the shell or from a Python interpreter/notebook, and takes input either as YAML_ files or Python dictionaries.


How to cite us
--------------

As of this version, there is no scientific publication yet associated to this software, so simply mention its GitHub repository (see above).

To appropriately cite the modules (samplers, theory codes, likelihoods) that you have used, simply run the script `cobaya-citation` with your input file(s) as argument(s), and you will get *bibtex* references and a short suggested text snippet for each module mentioned in your input file.


Acknowledgements
----------------

Thanks to `Guadalupe Cañas Herrera`_ for extensive and somewhat painful testing.


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
.. _`Guadalupe Cañas Herrera`: https://gcanasherrera.github.io/pages/about-me.html#about-me

===================

.. image:: ./img/logo_sussex.png
   :alt: University of Sussex
   :target: http://www.sussex.ac.uk/astronomy/
   :width: 150px
   :align: right

.. image:: ./img/logo_ERC.png
   :alt: European Research Council
   :target: http://erc.europa.eu/
   :width: 150px
   :align: right
