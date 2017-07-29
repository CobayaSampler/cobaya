Output and analysis
===================

.. _out_files:

Output files
------------

cobaya generates most commonly the following output files:

- One ``.yaml`` file with the same content as the input file, whose name ends in ``input.yaml``.
- Another ``.yaml`` file containing the input information plus the default values used by each module, ending in ``full.yaml``.
- One or more sample files, ending with ``[number].txt`` and containing one sample per line, with values separated by spaces. The first line specifies the columns.

.. note:: 

   Some samplers produce additional output, e.g. :doc:`PolyChord <sampler_polychord>`. In those cases, the resulting output is translated into cobaya's output format, but the sampler's native output is kept too, usually under subfolder within the output folder indicated with ``output_prefix`` (see the documentation for the particular sampler that you are using).


.. _out_example:
  
An example output and its analysis
----------------------------------

When invoking ``cobaya-run`` with the example input in the :ref:`previous section <in_example>`, a folder named ``example_gaussian`` will be created, and inside it you will find the three files described above:

.. code-block:: bash

   /path/to/cosmo/chains
                  └── example_gaussian
                      ├── input.yaml
                      ├── full.yaml
                      └── 1.txt

If you inspect the ``.txt`` file, the first lines will look like

.. code::

   # weight mlogpost mock_a mock_b logprior chi2 chi2__gaussian 
   4 3.5530711 0.58861423 -0.011199195 2.8622009 1.3817405   1.3817405
   7 2.897216  0.38683271  0.21944827  2.8622009 0.070030315 0.070030315
   [...]

You can use `GetDist <http://getdist.readthedocs.io/en/latest/index.html>`_ to analyse the results of this sample: get marginalised statistics, convergence diagnostics and some plots. We recommend (and for now **only** support) the `graphical user interface <http://getdist.readthedocs.io/en/latest/gui.html>`_. Simply run ``GetDistGUI.py`` from the chains folder, and add the ``example_gaussian`` chain folder: click the green *plus* button, in the pop-up window select the ``example_gaussian`` (just select it, don't go into it) and click on *choose*). After that, you can get some result statistics from the *Data* menu, or generate some plots like this one (just mark the the options in the red boxes and hit *Make plot*):

.. image:: img_output_getdistgui.png


.. _out_example_scripted:
   
Scripted analysis
-----------------

You can also do your analysis in a Python terminal or notebook, using either your own tools or the methods in GetDist.

For example, assuming that you have just run the example in :ref:`in_example_script`, to generate a similar plot to the one above and some statistics, simply do:

.. code:: python

   %matplotlib inline
   import getdist as gd
   import getdist.plots as gdplt
   gdsamples = collection.as_getdist_mcsamples()
   gdplot = gdplt.getSubplotPlotter()
   gdplot.triangle_plot(gdsamples, ["mock_a", "mock_b"], filled=True)
   print "Covariace matrix:\n", gdsamples.getCovMat().matrix


.. _output_prefix:
   
Specifying the output folder and/or prefix
------------------------------------------

You can tell cobaya to write the samples into a particular folder, or to name the output files in a certain way. To do that, use the option ``output_prefix`` at the top level of the input file (i.e. not inside any block):

- ``output_prefix: something``: the output will be written into the current folder, and all output files names will start with ``something``.
- ``output_prefix: somefolder/something``: similar to the last case, but writes into the folder ``somefolder``, which is created at that point and must not exist before.
- ``output_prefix: null``: will produce no output files whatsoever.

.. note::

   **When calling from the command line**, if ``output_prefix`` has not been specified, it
   defaults to the fisrt case, using as a prefix the name of the input file sans the ``yaml`` extension.

   Instead, **when calling from a Python interpreter**, if ``output_prefix`` has not been specified, it is understood as ``output_prefix: null``.


In all cases, the output folder is based on the invocation folder if cobaya is called from the command line, or the *current working directory* (i.e. the output of ``import os; os.getcwd()``) if invoked within a Python script or a Jupyter notebook.

Whatever the requiered output, the (potential) output files must not already exist, unless you are resuming a previous sampler (see :ref:`input_cont`).

.. note::

   When the output is written into a certain folder different from the invocation one, the value of ``output_prefix`` in the output ``.yaml`` file(s) is updated such that it drops the mention to that folder. This is done for consistency when resuming a sample (see :ref:`input_cont`), when the automatically generated ``.yaml`` file is used as the input.

