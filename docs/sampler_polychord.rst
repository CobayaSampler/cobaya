``polychord`` sampler
=====================

.. |br| raw:: html

   <br />

.. note::
   **If you use this sampler, please cite it as:**
   |br|
   `W.J. Handley, M.P. Hobson, A.N. Lasenby, "PolyChord: nested sampling for cosmology"
   (arXiv:1502.01856) <https://arxiv.org/abs/1502.01856>`_
   |br|
   `W.J. Handley, M.P. Hobson, A.N. Lasenby, "PolyChord: next-generation nested sampling"
   (arXiv:1506.00171) <https://arxiv.org/abs/1506.00171>`_

``PolyChord`` is an advanced
`nested sampler <https://projecteuclid.org/euclid.ba/1340370944>`_,
that uses slice sampling to sample within the
nested isolikelihoods contours. The use of slice sampling instead of rejection sampling
makes ``PolyChord`` specially suited for high-dimensional parameter spaces, and allows for
exploiting speed hierarchies in the parameter space. Also, ``PolyChord`` can explore
multi-modal distributions efficiently.

``PolyChord`` is an *external* sampler, not installed by default (just a wrapper for it).
You need to install it yourself following the installation instructions :ref:`below <pc_installation>`.

Usage
-----

To use ``PolyChord``, you just need to mention it in the ``sampler`` block:

.. code-block:: yaml

   sampler:
     polychord:
       # polychord options ...


Just copy the options that you wish to modify from the defaults file below:

.. literalinclude:: ../cobaya/samplers/polychord/polychord.yaml
   :language: yaml

.. warning::

   If you want to sample with ``PolyChord``, your priors need to be bounded. This is
   because ``PolyChord`` samples uniformly from a bounded *hypercube*, defined by a
   non-trivial transformation for general unbounded priors.

   The option ``confidence_for_unbounded`` will automatically bind the priors at 5-sigma
   c.l., but this may cause problems with likelihood modes at the edge of the prior.
   In those cases, check for stability with respect to increasing that parameter.
   Of course, if ``confidence_for_unbounded`` is much smaller than unity,
   the resulting evidence may be biased towards a lower value.

The main output is the Monte Carlo sample of sequentially discarded *live points*, saved
in the standard sample format together with the ``input.yaml`` and ``full.yaml``
files (see :doc:`output`). The raw ``PolyChord`` products are saved in a
subfolder of the output folder
(determined by the option ``base_dir`` – default: ``[your_output_prefix]_polychord_raw``). Since PolyChord is a nested sampler integrator, the log-evidence and its standard deviation are also returned.

If the posterior was found to be **multi-modal**, the output will include separate samples and evidences for each of the modes.


.. _polychord_bayes_ratios:

Computing Bayes ratios
----------------------

If you are using only internal priors, your full prior is correctly normalized, so you can directly use the output evidence of PolyChord, e.g. in a Bayes ratio.

If you are using external priors (as described :ref:`here <prior_external>`), the full prior is not normalized by default, so the resulting evidence, :math:`\mathcal{Z}_\mathrm{u}`, needs to be divided by the prior volume. To compute the prior volume :math:`\mathcal{Z}_\pi`, substitute your likelihoods for :doc:`the unit likelihood <likelihood_one>` ``one``. The normalized likelihood and its propagated error are in this case:

.. math::

   \log\mathcal{Z} &= \log\mathcal{Z}_\mathrm{u} - \log\mathcal{Z}_\pi\\
   \sigma(\log\mathcal{Z}) &= \sigma(\log\mathcal{Z}_\mathrm{u}) + \sigma(\log\mathcal{Z}_\pi)

.. warning::

   If any of the priors specified in the ``prior`` block or any of the likelihoods has large *unphysical* regions, i.e. regions of null prior or likelihood, you may want to increase the ``nprior`` parameter of PolyChord to a higher multiple of ``nlive`` (default ``10nlive``), depending on the complexity of the unphysical region.

   **Why?** The unphysical fraction of the parameter space, which is automatically subtracted to the raw PolyChord result, is guessed from a prior sample of size ``nprior``, so the higher that sample is, the smaller the bias introduced by its misestimation.

   Increasing ``nprior`` will make your run slower to initialize, but will not significantly affect the total duration.

   Notice that this behaviour differs from stock versions of popular nested samplers (MultiNest and PolyChord), which simply ignore unphysical points; but we have chosen to take them into account to keep the value of the prior density meaningful: otherwise by simply ignoring unphysical points, doubling the prior size (so halving its density) beyond physical limits would have no effect on the evidence.


Taking advantage of a speed hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cobaya's PolyChord wrapper *automatically* sorts parameters optimally, and chooses the number of repeats per likelihood according to the value of the ``oversampling_power`` property. You can also specify the blocking and oversampling by hand using the ``blocking`` option. For more thorough documentation of these options, check :ref:`the corresponding section in the MCMC sampler page<mcmc_speed_hierarchy>` (just ignore the references to ``drag``, which do not apply here).


.. _polychord_callback:

Callback functions
^^^^^^^^^^^^^^^^^^

A callback function can be specified through the ``callback_function`` option. It must be a function of a single argument, which at runtime is the current instance of the ``polychord`` sampler. You can access its attributes and methods inside your function, including the collection of ``live`` and ``dead`` points, the current calculation of the evidence, and the ``model`` (of which ``prior`` and ``likelihood`` are attributes). For example, the following callback function would print the number of points added to the chain since the last callback, the current evidence estimate and the maximum likelihood point:

.. code:: python

    def callback(sampler):
        print("There are %d dead points. The last %d were added since the last callback."%(
            len(sampler.dead), len(sampler.dead) - sampler.last_point_callback))
        print("Current logZ = %g +/- %g"%(sampler.logZ, sampler.logZstd))
        # Maximum likelihood: since we sample over the posterior, it may be "dead"!
        min_chi2_dead = sampler.dead[sampler.dead["chi2"].values.argmin()]
        # At the end of the run, the list of live points is empty
        try:
            min_chi2_live = sampler.live[sampler.live["chi2"].values.argmin()]
            min_chi2_point = (min_chi2_live if min_chi2_live["chi2"] < min_chi2_dead["chi2"]
                              else min_chi2_dead)
        except:
            min_chi2_point = min_chi2_dead
        print("The maximum likelihood (min chi^2) point reached is\n%r"%min_chi2_point)

The frequency of calls of the callback function is given by the ``compression_factor`` (see contents of ``polychord.yaml`` above).

.. note::

   Errors produced inside the callback function will be reported, but they will not stop PolyChord.


Troubleshooting
---------------

If you are getting an error whose cause is not immediately obvious, try substituting ``polychord`` by :doc:`the dummy sampler <sampler_evaluate>` ``evaluate``.

If still in doubt, run with debug output and check what the prior and likelihood are getting and producing: either set ``debug: True`` in the input file and set ``debug_file`` to some file name, or add the ``--debug`` flag to ``cobaya-run`` and pipe the output to a file with ``cobaya-run [input.yaml] --debug > file``.

If everything seems to be working fine, but PolyChord is taking too long to converge, reduce the number of live points ``nlive`` to e.g. 10 per dimension, and the ``precision_criterion`` to 0.1 or so, and check that you get reasonable (but low-precision) sample and evidences.

See also :ref:`cosmo_polychord`.


.. _pc_installation:

Installation
------------

Simply run ``cobaya-install polychord --packages-path [/path/to/packages]`` (or, instead of ``polychord`` after ``cobaya-install``, mention an input file that uses ``polychord``).

If PolyChord has been installed this way, it is not necessary to specify a ``path`` option for it.

.. note::

   To run PolyChord with MPI (highly recommended!) you need to make sure that MPI+Python is working in your system, see :ref:`install_mpi`.

   In addition, you need a MPI-wrapped Fortran compiler. You should have an MPI implementation installed if you followed  :ref:`the instructions to install mpi4py <install_mpi>`. On top of that, you need the Fortran compiler (we recommend the GNU one) and the *development* package of MPI. Use your system's package manager to install them (``sudo apt install gfortran libopenmpi-dev`` in Ubuntu/Debian systems), or contact your local IT service. If everything is correctly installed, you should be able to type ``mpif90`` in the shell and not get a ``Command not found`` error.


.. note::

   **Polychord for Mac users:**

   To have PolyChord work, install ``gcc`` (maybe using Homebrew), and check that ``gcc-[X]`` works in the terminal, where ``[X]`` is the version that you have just installed.

   Now install PolyChord, either by hand or using **cobaya**'s automatic installer, go to the newly created ``PolyChord`` folder, and compile it with

   .. code:: bash

      $ make pypolychord MPI= CC=gcc-[X] CXX=g++-[X]
      $ python setup.py build

   If you want to use PolyChord with MPI on a Mac, you need to have compiled OpenMPI with the same ``gcc`` version with which you are compiling PolyChord. To do that, prepend OpenMPI's ``make`` command with ``CC=gcc-[X]``, where ``[X]`` is your gcc version. Then follow the instructions above to compile PolyChord, but with ``MPI=1`` instead when you do ``make pypolychord``.

   *Thanks to Guadalupe Cañas Herrera for these instructions!*

   **Polychord for Windows users:**

   Polychord currently does not support Windows. You'd have to run it in linux virtual machine or using a Docker container.


Manual installation of PolyChord
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to install PolyChord manually, assuming you want to install it at ``/path/to/polychord``, simply do

.. code:: bash

   $ cd /path/to/polychord
   $ git clone https://github.com/PolyChord/PolyChordLite.git
   $ cd PolyChordLite
   $ make pypolychord MPI=1
   $ python setup.py build

After this, mention the path in your input file as

.. code-block:: yaml

   sampler:
     polychord:
       path: /path/to/polychord/PolyChordLite


PolyChord class
---------------

.. automodule:: samplers.polychord.polychord
   :noindex:

.. autoclass:: samplers.polychord.polychord
   :members:

