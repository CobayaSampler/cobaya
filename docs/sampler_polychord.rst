Samplers – PolyChord
====================

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
`nested sampler <http://projecteuclid.org/euclid.ba/1340370944>`_,
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

.. note::

   The speed hierarchy exploitation is disabled for now, sorry!

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

If you are using only internal 1-dimensional priors, your full prior is correctly normalised, so the output evidence of PolyChord should be directly the marginal likelihood and you can use it in a Bayes ratio.

If you are using custom/external priors, as described :ref:`here <prior_external>`, the full prior is not normalised by default. Thus, to get the marginal likelihood you need to subtract the prior volume. To compute it, substitute your likelihoods for :doc:`the unit likelihood <likelihood_one>` ``one``. If your prior is flat, add a small noise term to ``one`` (see its documentation) and, unless your prior has *islands*, set PolyChord's ``do_clustering: False`` and ``precision_criterion: 0.01`` (or smaller, but not much) to converge faster. You may get some ``Non deterministic loglikelihood`` warnings coming from PolyChord, but don't worry about them.


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

The easiest way to install it is through ``cobaya-install [input.yaml]``, where the ``input.yaml`` contains a mention to ``polychord`` inside the ``sampler`` block:

.. code:: yaml

   # Contents of input.yaml
   # [...]
   sampler:
     polychord:
       # [...]
   # [...]


If it has been installed this way, it is not necessary to specify a ``path`` for it, as long as the modules folder has been indicated.

.. note::

   To run PolyChord with MPI (highly recommended!) you need to make sure that MPI+Python is working in your system, see :ref:`install_mpi`.

   In addition, you need a MPI-wrapped Fortran compiler. You should have an MPI implementation installed if you followed  :ref:`the instructions to install mpi4py <install_mpi>`. On top of that, you need the Fortran compiler (we recommend the GNU one) and the *development* package of MPI. Use your system's package manager to install them (``sudo apt install gfortran libopenmpi-dev`` in Ubunto/Debian systems), or contact your local IT service. If everything is correctly installed, you should be able to type ``mpif90`` in the shell and not get a ``Command not found`` error.


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

