Example runs -- Planck 2015
===========================

We are assuming that you have installed the Planck 2015 likelihood, following the instructions in :doc:`likelihood_planck`, under the folder ``/path/to/cosmo/likelihoods/planck_2015``.

Also, we assume that you have installed one of the usual cosmological codes, :doc:`CAMB <theory_camb>` or :doc:`CLASS <theory_class>`, under their respectively named folder in ``/path/to/cosmo/``.

Now create a ``chains`` folder in that same place and copy the next input into a file names ``example_planck.yaml``:

.. code-block:: yaml
                
   theory:
     camb:

   # theory:
   #   classy:

   likelihood:
     planck_2015_lowTEB:
       speed: 1
     planck_2015_plikHM_TTTEEE:
       speed: 4

   params:
     # Likelihood parameters, e.g.:
     # A_planck: 1
     # Theory code parameters
     theory:
       ombh2:
         prior:
           min: 0.005
           max: 0.1
         ref:
           dist: norm
           loc: 0.0221
           scale: 0.0001
         proposal: 0.0001
         latex: \omega_\mathrm{b}
       omch2:
         prior:
           min: 0.001
           max: 0.99
         ref:
           dist: norm
           loc: 0.12
           scale: 0.001
         proposal: 0.0005
         latex: \omega_\mathrm{cdm}
       H0:
         prior:
           min: 50
           max: 90
         ref:
           dist: norm
           loc: 67
           scale: 2
         proposal: 1.2
         latex: H_0
       As:
         prior:
           min: 7.0e-10
           max: 1.0e-08
         ref:
           min: 2.217e-09
           max: 2.222e-09
         proposal: 2.0e-12
         latex: A_s
       ns:
         prior:
           min: 0.8
           max: 1.2
         ref:
           dist: norm
           loc: 0.96
           scale: 0.004
         proposal: 0.002
         latex: n_s
       tau:
         prior:
           min: 0.01
           max: 0.1
         ref:
           dist: norm
           loc: 0.09
           scale: 0.01
         proposal: 0.005
         latex: \tau_\mathrm{reio}

   sampler:
     mcmc:
       burn_in: 100
       max_samples: 10000
       drag_nfast_times: 3


This creates a folder named ``example_planck`` with a copy of the input file, an extended info file and a chain file ``1.txt`` that will get populated with samples after the burn-in phase is finished.
       
.. note::

   If for some reason (e.g. a failed attempt) the folder ``example_planck`` already exists. the run will fail, to prevent overwriting the old sample. Delete the folder and try again.

You can exchange the ``camb`` theory block for the corresponding ``classy`` theory block, and it would run in both cases.

.. note::
   
   You can also change the ``sampler`` block to use the PolyChord sampler, but in that case you would also need to:

   - use ``camb`` as a cosmological code, since it allows for a larger parameter space.
   - fix the value of the Planck parameters with a Gaussian prior, since PolyChord needs priors to be bounded. Simply add to the ``likelihood`` block of the ``params`` block:

     .. code-block:: yaml
                   
        A_planck: 1
        gal545_A_100: 7
        gal545_A_143: 9
        gal545_A_143_217: 21
        gal545_A_217: 80
        galf_EE_A_100: 0.06
        galf_EE_A_100_143: 0.05
        galf_EE_A_100_217: 0.11
        galf_EE_A_143: 0.1
        galf_EE_A_143_217: 0.24
        galf_EE_A_217: 0.72
        galf_TE_A_100: 0.14
        galf_TE_A_100_143: 0.12
        galf_TE_A_100_217: 0.3
        galf_TE_A_143: 0.24
        galf_TE_A_143_217: 0.6
        galf_TE_A_217: 1.8
        calib_100T: 0.999
        calib_217T: 0.995


After a couple of hours, you should be able to run ``GetDistGUI`` to generate some plots.


Citations made easy!
--------------------

If you would like to cite the results of this run in a paper, you would need citations for all the different parts of the process: this very sampling framework, the MCMC sampler used, the CAMB or CLASS cosmological code and the Planck 2015 likelihoods.

The ``bibtex`` for those citations, along with a short text snippet for each element, can be easily obtained and saved to some ``output_file.tex`` with

.. code-block:: bash

   $ cobaya-citation example_planck.yaml > output_file.tex
