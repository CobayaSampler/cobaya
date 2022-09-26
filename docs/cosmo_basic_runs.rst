Basic cosmology runs
====================

Sampling from a cosmological posterior works the same way as the examples at the beginning of the documentation, except that one usually needs to add a theory code, and possibly some of the cosmological likelihoods presented later.

You can sample or track any parameter that is understood by the theory code in use (or any dynamical redefinition of those). You **do not need to modify Cobaya's source** to use new parameters that you have created by :ref:`modifying CLASS <classy_modify>` or :ref:`modifying CAMB <camb_modify>`, or to :doc:`create a new cosmological likelihood <cosmo_external_likelihood>` and track its parameters.

Creating *from scratch* the input for a realistic cosmological case is quite a bit of work. But to make it simpler, we have created an automatic **input generator**, that you can run from the shell as:

.. code:: bash

   $ cobaya-cosmo-generator

.. image:: img/cosmo_input_generator.png
   :align: center

.. note::

   If ``PySide2`` is not installed, this will fail. To fix it:

   .. code:: bash

      $ python -m pip install pyqt5 pyside2


   Anaconda users should instead do:

   .. code:: bash

      $ conda install -c conda-forge pyside2

   Installing PySide2 via pip, and sometime Anaconda, is often problematic.
   The most reliable solution seems to be to make a clean conda-forge environment and
   use that, e.g. install Anaconda or Miniconda and use the environment created with

  .. code:: bash

      $ conda create -n py39forge -c conda-forge python=3.9 scipy pandas matplotlib PyYAML PySide2

Start by choosing a preset, maybe modify some aspects using the options provided, and copy or save the generated input to a file, either in ``yaml`` form or as a python dictionary.

The parameter combinations and options included in the input generator are in general well-tested, but they are only suggestions: **you can add by hand any parameter that your theory code or likelihood can understand, or modify any setting**.

You can add an ``output`` prefix if you wish (otherwise, the name of the input file without extension is used). If it contains a folder, e.g. ``chains/[whatever]``, that folder will be created if it does not exist.

In general, you do not need to mention the installation path used by ``cobaya-install`` (see :doc:`installation_cosmo`): it will be selected automatically. If that does not work, add ``packages_path: '/path/to/packages'`` in the input file, or ``-p /path/to/packages`` as a ``cobaya-run`` argument.

.. Notice the checkbox **"Keep common parameter names"**: if checked, instead of the parameter names used by CAMB or CLASS (different from each other), the input will use a common parameter names set, understandable by both. If you are using this, you can exchange both theory codes safely (just don't forget to add the ``extra_args`` generated separately for each theory code.


As an example, here is the input for Planck 2015 base :math:`\Lambda\mathrm{CDM}`, both for CLASS and CAMB:

.. container:: cosmo_example

   .. container:: switch

      Click to toggle CAMB/CLASS

   .. container:: default

      .. literalinclude:: ./src_examples/cosmo_basic/basic_camb.yaml
         :language: yaml
         :caption: **CAMB parameter names:**

   .. container:: alt

      .. literalinclude:: ./src_examples/cosmo_basic/basic_classy.yaml
         :language: yaml
         :caption: **CLASS parameter names:**

.. note::

   Note that Planck likelihood parameters (or *nuisance parameters*) do not appear in the input: they are included automatically at run time. The same goes for all *internal* likelihoods (i.e. those listed below in the table of contents).

   You can still add them to the input, if you want to redefine any of their properties (its prior, label, etc.). See :ref:`prior_inheritance`.


Save the input generated to a file and run it with ``cobaya-run [your_input_file_name.yaml]``. This will create output files as explained :ref:`here <output_shell>`, and, after some time, you should be able to run ``getdist-gui`` to generate some plots.

.. note::

   You may want to start with a *test run*, adding ``--test`` to ``cobaya-run`` (run without MPI). It will initialise all components (cosmological theory code and likelihoods, and the sampler) and exit.

Typical running times for MCMC when using computationally heavy likelihoods (e.g. those involving :math:`C_\ell`, or non-linear :math:`P(k,z)` for several redshifts) are ~10 hours running 4 MPI processes with 4 OpenMP threads per process, provided that the initial covariance matrix is a good approximation to the one of the real posterior (Cobaya tries to select it automatically from a database; check the ``[mcmc]`` output towards the top to see if it succeeded), or a few hours on top of that if the initial covariance matrix is not a good approximation.

It is much harder to provide typical PolyChord running times. We recommend starting with a low number of live points and a low convergence tolerance, and build up from there towards PolyChord's default settings (or higher, if needed).

If you would like to find the MAP (maximum-a-posteriori) or best fit (maximum of the likelihood within prior ranges, but ignoring prior density), you can swap the sampler (``mcmc``, ``polychord``, etc) by ``minimize``, as described in :doc:`sampler_minimize`. As a shortcut, to run a minimizer process for the MAP without modifying your input file, you can simply do

.. code:: bash

   cobaya-run [your_input_file_name.yaml] --minimize


.. _cosmo_post:

Post-processing cosmological samples
------------------------------------

Let's suppose that we want to importance-reweight a Planck sample, in particular the one we just generated with the input above, with some late time LSS data from BAO. To do that, we ``add`` the new BAO likelihoods. We would also like to increase the theory code's precision with some extra arguments: we will need to re-``add`` it, and set the new precision parameter under ``extra_args`` (the old ``extra_args`` will be inherited, unless specifically redefined).
For his example let's say we do not need to recompute the CMB likelihoods, so power spectra do not need to be recomputed, but we do want to add a new derived parameter.

Assuming we saved the sample at ``chains/planck``, we need to define the following input file, which we can run with ``$ cobaya-run``:

.. code:: yaml

   # Path the original sample
   output: chains/planck

   # Post-processing information
   post:
     suffix: BAO  # the new sample will be called "chains\planck_post_des*"
     # If we want to skip the first third of the chain as burn in
     skip: 0.3
     # Now let's add the DES likelihood,
     # increase the precision (remember to repeat the extra_args)
     # and add the new derived parameter
     add:
       likelihood:
         sixdf_2011_bao:
         sdss_dr7_mgs:
         sdss_dr12_consensus_bao:
       theory:
         # Use *only* the theory corresponding to the original sample
         classy:
           extra_args:
             # New precision parameter
             # [option]: [value]
         camb:
           extra_args:
             # New precision parameter
             # [option]: [value]
       params:
         # h = H0/100. (nothing to add: CLASS/CAMB knows it)
         h:
         # A dynamic derived parameter (change omegam to Omega_m for classy)
         # note that sigma8 itself is not recomputed unless we add+remove it
         S8:
           derived: 'lambda sigma8, omegam: sigma8*(omegam/0.3)**0.5'
           latex: \sigma_8 (\Omega_\mathrm{m}/0.3)^{0.5}


.. _compare_cosmomc:

Comparison with CosmoMC/GetDist conventions
-------------------------------------------

In CosmoMC, uniform priors are defined with unit density, whereas in Cobaya their density is the inverse of their range, so that they integrate to 1. Because of this, the value of CosmoMC posteriors is different from Cobaya's. In fact, CosmoMC (and GetDist) call its posterior *log-likelihood*, and it consists of the sum of the individual data log-likelihoods and the non-flat log-priors (which also do not necessarily have the same normalisation as in Cobaya). So the comparison of posterior values is non-trivial. But values of particular likelihoods (``chi2__[likelihood_name]`` in Cobaya) should be almost exactly equal in Cobaya and CosmoMC at equal cosmological parameter values.

Regarding minimizer runs, Cobaya produces both a ``[prefix].minimum.txt`` file following the same conventions as the output chains, and also a legacy ``[prefix].minimum`` file (no ``.txt`` extension) similar to CosmoMC's for GetDist compatibility, following the conventions described above.


.. _citations:

Getting help and bibliography for a component
---------------------------------------------

If you want to get the available options with their default values for a given component, use

.. code-block:: bash

   $ cobaya-doc [component_name]

The output will be YAML-compatible by default, and Python-compatible if passed a ``-p`` / ``--python`` flag.

Call ``$ cobaya-doc`` with no arguments to get a list of all available components of all kinds.

If you would like to cite the results of a run in a paper, you would need citations for all the different parts of the process. In the example above that would be this very sampling framework, the MCMC sampler, the CAMB or CLASS cosmological code and the Planck 2018 likelihoods.

The ``bibtex`` for those citations, along with a short text snippet for each element, can be easily obtained and saved to some ``output_file.tex`` with

.. code-block:: bash

   $ cobaya-bib [your_input_file_name.yaml] > output_file.tex

You can pass multiple input files this way, or even a (list of) component name(s).

You can also do this interactively, by passing your input info, as a python dictionary, to the function :func:`~bib.get_bib_info`:

.. code-block:: python

   from cobaya.bib import get_bib_info
   get_bib_info(info)

.. note::

   Both defaults and bibliography are available in the **GUI** (menu ``Show defaults and bibliography for a component ...``).

   Bibliography for *preset* input files is displayed in the ``bibliography`` tab.
