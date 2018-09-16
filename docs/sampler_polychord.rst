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
You need to install it yourself following the :doc:`general instructions for installing external modules <installation_cosmo>`, or the manual installation instructions :ref:`below <pc_installation>`.

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
(determined by the option ``base_dir`` – default: ``raw_polychord_output``). Since PolyChord is a nester sampler integrator, the log-evidence and its standard deviation are also returned.

If the posterior was found to be **multi-modal**, the output will include separate samples and evidences for each of the modes.


.. _polychord_bayes_ratios:

Computing Bayes ratios
----------------------

If you are using only internal priors, your full prior is correctly normalised, so you can directly use the output evidence of PolyChord, e.g. in a Bayes ratio.

If you are using external priors (as described :ref:`here <prior_external>`), the full prior is not normalised by default, so the resulting evidence, :math:`\mathcal{Z}_\mathrm{u}`, needs to be divided by the prior volume. To compute the prior volume :math:`\mathcal{Z}_\pi`, substitute your likelihoods for :doc:`the unit likelihood <likelihood_one>` ``one``. The normalised likelihood and its propagated error are in this case:

.. math::

   \log\mathcal{Z} &= \log\mathcal{Z}_\mathrm{u} - \log\mathcal{Z}_\pi\\
   \sigma(\log\mathcal{Z}) &= \sigma(\log\mathcal{Z}_\mathrm{u}) + \sigma(\log\mathcal{Z}_\pi)

If your prior is **constant** in the region of interest, in order for PolyChord not to get stuck, you have to add a small noise term to the likelihood ``one`` (see :doc:`its documentation <likelihood_one>`). The noise amplitude must be smaller by a couple of orders of magnitude than the inverse of a rough estimation of the expected prior volume, e.g. if your prior volume is expected to be :math:`\mathcal{O}(10^3)`, make the noise :math:`\mathcal{O}(10^{-5})`. PolyChord will take a while to converge (even if the prior evaluation is fast), and you may get some ``Non deterministic loglikelihood`` warnings coming from PolyChord, but don't worry about them.

As an example, if we want to compute the area of the **constant** triangle :math:`y > x` in a square of side 10 (expected area :math:`\sim 100`), we would use the following input file:

.. code:: yaml

   params:
     x:
       prior:
         min:  0
         max: 10
     y:
       prior:
         min:  0
         max: 10
   prior:
     triangle: "lambda x,y: np.log(y>x)"
   likelihood:
     one:
       noise: 1e-4
   sampler:
     polychord:


Troubleshooting
---------------

If you are getting an error whose cause is not immediately obvious, try substituting ``polychord`` by :doc:`the dummy sampler <sampler_evaluate>` ``evaluate``.

If still in doubt, run with debug output and check what the prior and likelihood are getting and producing: either set ``debug: True`` in the input file and set ``debug_file`` to some file name, or add the ``--debug`` flag to ``cobaya-run`` and pipe the output to a file with ``cobaya-run [input.yaml] --debug >file``.

If PolyChord gets stuck in ``started sampling``, it probably means that your posterior is flat; if that was intentional, check the :ref:`polychord_bayes_ratios` section, where it is discussed how to deal with those cases.

If everything seems to be working fine, but PolyChord is taking too long to converge, reduce the number of live points ``nlive`` to e.g. 10 per dimension, and the ``precision_criterion`` to 0.1 or so, and check that you get reasonable (but low-precision) sample and evidences.


.. _pc_installation:

Installation
------------

At the moment, **cobaya** uses its own PolyChord distribution, hosted `here <https://github.com/CobayaSampler/PolyChord>`_, which is an exact clone of the original PolyChord, hosted at `CCPForge <https://ccpforge.cse.rl.ac.uk/gf/project/polychord/>`_, with an alternative Python interface.

The easiest way to install it is through ``cobaya-install [input.yaml] --modules [/path/to/modules]``, where the ``input.yaml`` contains a mention to ``polychord`` inside the ``sampler`` block, and the installation path of the modules is indicated:

.. code:: yaml

    # Contents of input.yaml
    modules: /path/to/modules
    sampler:
      polychord:
        # [Polychord options]
    # ...

If it has been installed this way, it is not necessary to specify a ``path`` for it, as long as the modules folder has been indicated.

.. note::

   To run PolyChord with MPI (highly recommended!) you need to make sure that MPI+Python is working in your system, see :ref:`install_mpi`.

   In addition, you need a MPI-wrapped Fortran compiler. You should have an MPI implementation installed if you followed  :ref:`the instructions to install mpi4py <install_mpi>`. On top of that, you need the Fortran compiler (we recommend the GNU one) and the *development* package of MPI. Use your system's package manager to install them (``sudo apt install gfortran libopenmpi-dev`` in Ubunto/Debian systems), or contact your local IT service. If everything is correctly installed, you should be able to type ``mpif90`` in the shell and not get a ``Command not found`` error.


.. note::

   **Polychord for Mac users:**

   To have PolyChord work, install ``gcc`` (maybe using Homebrew), and check that ``gcc-[X]`` works in the terminal, where ``[X]`` is the version that you have just installed.

   Now install PolyChord, either by hand or using **cobaya**'s automatic installer, go to the newly created ``PolyChord`` folder, and compile it with

   .. code:: bash

      $ make veryclean
      $ make PyPolyChord MPI= CC=gcc-[X] CXX=g++-[X]
      $ python[Y] setup.py install

   where ``[X]`` is your ``gcc`` version and ``[Y]`` is your python version: ``2`` or ``3``. Add a ``--user`` flag to the Python command if you get an error message showing ``[Errno 13] Permission denied`` anywhere.

   If you want to use PolyChord with MPI on a Mac, you need to have compiled OpenMPI with the same ``gcc`` version with which you are compiling PolyChord. To do that, prepend OpenMPI's ``make`` command with ``CC=gcc-[X]``, where ``[X]`` is your gcc version. Then follow the instructions above to compile PolyChord, but with ``MPI=1`` instead when you do ``make PyPolyChord``.

   *Thanks to Guadalupe Cañas Herrera for these instructions!*


Manual installation of PolyChord
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to install PolyChord manually, assuming you want to install it at ``/path/to/polychord``, simply do

.. code:: bash

   $ cd /path/to/polychord
   $ git clone https://github.com/CobayaSampler/PolyChord.git
   $ cd PolyChord
   $ make PyPolyChord MPI=1

After this, mention the path in your input file as

.. code-block:: yaml

   sampler:
     polychord:
       path: /path/to/polychord/PolyChord


PolyChord class
---------------

.. automodule:: samplers.polychord.polychord
   :noindex:
   
.. autoclass:: samplers.polychord.polychord
   :members:

