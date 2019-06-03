Troubleshooting in cosmological runs
====================================

This section will be progressively filled with the most common problems that our users encounter, so don't hesitate to open an issue/PR in GitHub if you think there is something worth including here.

Low performance on a cluster
----------------------------

If you notice that Cobaya is performing unusually slow on your cluster *compared to your own computer*, run a small number of evaluations using the :doc:`sampler_evaluate` sampler with ``timing: True``: it will print the average computation time of each part of the code, so that you can check which one is the slowest.

- If it is CAMB or CLASS, they are probably not using OpenMP parallelisation. To check that this is the case, try running ``top`` on the node when Cobaya is running, and check that CPU usage goes above 100% regularly. If it does not, you need to allocate more cores per process, or, if it doesn't fix it, make sure that OpenMP is working correctly (ask your local IT support for help with this).

- If it is some part of the code written in pure Python, ``numpy`` may not be taking advantage of parallelisation. To fix that, follow :ref:`this instructions <install_openblas>`.


Running out of memory (memory leak)
-----------------------------------

Python rarely runs out of memory, thanks to its *garbage collector*, so the culprit is probably some external C or Fortran code.

If you have modified CLASS or CAMB, make sure that every ``alloc`` is followed by the corresponding ``free`` in C, and every ``allocate`` is followed by a ``deallocate`` in Fortran. Otherwise, a new array will be created at each iteration while the old one will not be deleted.

You can use e.g. `Valgrind <http://www.valgrind.org/>`_ to monitor memory usage.

.. note::

   In particular, for CLASS, check out :ref:`this warning <classy_install_warn>` concerning moving the CLASS folder after compilation.
