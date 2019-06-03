Basic cosmology runs
====================

Sampling from a cosmological posterior works exactly the same as the examples at the beginning of the documentation, now using a theory code and some likelihoods from the ones presented later.

You can sample over any parameter that is understood by the theory code that you are using (or any dynamical dynamical redefinition of them), and same goes for tracking a derived parameter. Than includes new parameters that you have created by modifying the theory code :doc:`modify_theory` or :doc:`creating a new likelihood <cosmo_external_likelihood>`, without modifying **cobaya**'s source code.

Creating the input info for a realistic cosmological case from scratch would take a little work, so, to make it simpler, we have created an automatic *input generator*, that you can run from the console as:

.. code:: bash

   $ cobaya-cosmo-generator

.. image:: img/cosmo_input_generator.png
   :align: center

.. note::

   If ``PySide`` is not installed, this will fail. To fix it:

   .. code:: bash

      $ pip3 install pyside2


   **Anaconda** users should instead do:

   .. code:: bash

      $ conda install -c conda-forge pyside2

   .. warning::

      In Python 2 (soon to be discontinued!) try **one** of the following:

      .. code:: bash

         $ sudo apt install python-pyside

      .. code:: bash

         $ pip install PySide2  # add --user if it fails

Start by choosing a preset, maybe modify some aspects using the options provided, and copy or save the generated input to a file, either in ``yaml`` form or as a python dictionary.

The parameter combinations and options included in the input generator are in general well-tested, but they are only suggestions: **you can add by hand any parameter that your theory code or likelihood can understand, or modify any setting**.

Don't forget to add your installation path for the cosmological modules as ``modules: '/path/to/modules'``, and an ``output`` prefix if you wish.

.. note::

   Notice the checkbox **"Keep common parameter names"**: if checked, instead of the parameter names used by CAMB or CLASS (different from each other), the input will use a common parameter names set, understandable by both. If you are using this, you can exchange both theory codes safely (just don't forget to add the ``extra_args`` generated separately for each theory code.


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


Save the input generated to a file and run it with ``cobaya-run [your_input_file_name.yaml]``. This will create output files as explained :ref:`here <output_shell>`, and, after a couple of hours, you should be able to run ``GetDistGUI`` to generate some plots.


.. _citations:

Citations made easy!
--------------------

If you would like to cite the results of this run in a paper, you would need citations for all the different parts of the process: this very sampling framework, the MCMC sampler used, the CAMB or CLASS cosmological code and the Planck 2015 likelihoods.

The ``bibtex`` for those citations, along with a short text snippet for each element, can be easily obtained and saved to some ``output_file.tex`` with

.. code-block:: bash

   $ cobaya-citation [your_input_file_name.yaml] > output_file.tex

You can pass multiple input files this way.

You can also do this interactively, by passing your input info, as a python dictionary, to the function :func:`~citation.citation`:

.. code-block:: python

   from cobaya.citation import citation
   citation(info)
