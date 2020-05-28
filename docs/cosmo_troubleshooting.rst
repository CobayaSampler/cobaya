Troubleshooting in cosmological runs
====================================

This section will be progressively filled with the most common problems that our users encounter, so don't hesitate to open an issue/PR in `GitHub <https://github.com/CobayaSampler/cobaya/issues>`_ if you think there is something worth including here.


General troubleshooting advice
------------------------------

If you are getting an error whose cause is not immediately obvious, try evaluating your model at a point in parameter space where you expect it to work. To do that, either substitute your sampler for :doc:`the dummy sampler <sampler_evaluate>` ``evaluate``, or use the :doc:`model wrapper <cosmo_model>` instead of a sampler and call its ``logposterior`` method.

You can increase the level of verbosity running with ``debug: True`` (or adding the ``--debug`` flag to ``cobaya-run``). Cobaya will print what each part of the code is getting and producing, as well as some other intermediate info. You can pipe the debug output to a file with ``cobaya-run [input.yaml] --debug > file`` or setting ``debug_file: [filename]``.


MPI runs: getting one MCMC chain only
-------------------------------------

If your MPI run is only producing one chain (just ``[prefix].1.txt``, no higher numbers), your Python MPI wrapper is not working correctly. Please take a look at :ref:`this section <install_mpi>`.


.. _cosmo_polychord:

Using PolyChord in cosmological runs
------------------------------------

PolyChord explores the full posterior domain much more thoroughly than MCMC does, in order to get an accurate estimation of the model evidence. Due to that a couple of caveats are in order:

* The :doc:`Planck likelihood <likelihood_planck>` does not often deal gracefully with extreme values of the power spectrum: some times, when at the tails of the posterior, it may produce a segfault (the error message will contain something like ``Segmentation Fault`` or ``Signal 11``, and a references to a files such as ``clik/lkl.[some_text].so``). Not much can be done about this, except for reducing the prior boundaries for the cosmological parameters by hand.

* For most cases, CAMB and CLASS should produce very similar evidences for the same models, but only as long as the posterior does not extend towards regions which are unphysical for one of the Boltzmann codes and not the other one. You should not see any effect for most LCDM extensions, but keep an eye open if comparing CAMB and CLASS results for exotic models.

* [WIP: typical running time]


Sampling stuck or not saving any point
--------------------------------------

If your sampler appears to be making no progress, your likelihood or theory code may be failing silently, and thus assuming a *null* likelihood value (this is intended default behaviour, since cosmological theory codes and likelihoods tend to fail for extreme parameter values). If that is the case, you should see messages about errors being ignored when running with ``debug: True``. To stop when one of those errors occur, set the option ``stop_at_error: True`` for the relevant likelihood or theory code.


Low performance on a cluster
----------------------------

If you notice that Cobaya is performing unusually slow on your cluster *compared to your own computer*, run a small number of evaluations using the :doc:`sampler_evaluate` sampler with ``timing: True``: it will print the average computation time per evaluation and the total number of evaluations of each part of the code, so that you can check which one is the slowest. You can also combine ``timing: True`` with any other sampler, and it will print the timing info once every checkpoint, and once more at the end of the run.

- If CAMB or CLASS is the slowest part, they are probably not using OpenMP parallelisation. To check that this is the case, try running ``top`` on the node when Cobaya is running, and check that CPU usage goes above 100% regularly. If it does not, you need to allocate more cores per process, or, if it doesn't fix it, make sure that OpenMP is working correctly (ask your local IT support for help with this).

- If it is some other part of the code written in pure Python, ``numpy`` may not be taking advantage of parallelisation. To fix that, follow :ref:`these instructions <install_openblas>`.


Running out of memory
---------------------

If your job runs out of memory at **at initialisation** of the theory or likelihoods, you may need to allocate more memory for your job.

If, instead, your jobs runs out of memory **after a number of iterations**, there is probably a memory leak somewhere. Python rarely leaks memory, thanks to its *garbage collector*, so the culprit is probably some external C or Fortran code.

If you have modified C code, e.g. CLASS, make sure that every ``alloc`` is followed by the corresponding ``free``. Otherwise, a new array will be created at each iteration while the old one will not be deleted.

You can use e.g. `Valgrind <http://www.valgrind.org/>`_ to monitor memory usage.

.. note::

   In particular, for CLASS, check out :ref:`this warning <classy_install_warn>` concerning moving the CLASS folder after compilation.


Secondary MPI processes not dying
---------------------------------

We have noticed that hitting :kbd:`Control-c` **twice in a row** prevents the termination signal to be propagated among processes, letting some or all secondary ones running after the primary one is killed, so that they need to be killed manually. Please, be patient!

Secondary processes not dying is something that should not happen when running on a cluster. If this happens, please report to us via GitHub, including as much information about the run as possible.
