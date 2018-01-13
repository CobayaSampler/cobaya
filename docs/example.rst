Quickstart example
==================

Let us present here a simple example, without explaining much about it — each of its aspects will be broken down in the following sections.

We will run it first from the **shell**, and later from a **Python interpreter** (or a Jupyter notebook).


From the shell
--------------

The input of **cobaya** usually looks like this:

.. literalinclude:: ./src_examples/quickstart/gaussian.yaml
   :language: yaml

You can see the following *blocks* up there:
              
- A ``likelihood`` block, listing the likelihood pdf's to be explored, here a gaussian with the mean and covariance stated.
- A ``params`` block, stating the parameters that are going to be explored (or derived), their ``prior``, the the Latex label that will be used in the plots, the `ref`erence starting point for the chains (optional), and the initial spread of the MCMC covariance matrix ``proposal``.
- A ``sampler`` block stating that we will use the ``mcmc`` sampler to explore the prior+likelihood described above, stating the maximum number of samples used, how many initial samples to ignore, and that we will sequentially refine our initiall guess for a covariance matrix.
- An ``output_prefix``, indicating where the products will be written and a prefix for their name.

To run this example, save the text above in a file called ``gaussian.yaml`` in a folder of your choice, and do

.. code:: bash

   $ cobaya-run gaussian.yaml

After a few seconts, a folder named ``chains`` will be created, and inside it you will find three files:

.. code-block:: bash

   chains
   ├── gaussian__input.yaml
   ├── gaussian__full.yaml
   └── gaussian_1.txt

The first file reproduces the same information as the input file given, here ``gaussian.yaml``. The second containts the ``full`` information needed to reproduce the sample, similar to the input one, but populated with the default options for the sampler, likelihood, etc. that you have used.

The third file, ending in ``.txt``, contains the MCMC sample, and its first lines should look like

.. code::

   # weight minuslogpost a b derived__derived_a derived__derived_b minuslogprior chi2 chi2__gaussian
   14 3.7928475 -0.041594155 0.41613663 -0.7639878 1.2835171 2.8622009 1.8612931 1.8612931
   3 4.2508064 -0.30329058 0.076149634 -1.5915446 0.78357974 2.8622009 2.777211 2.777211
   [...]

You can use `GetDist <http://getdist.readthedocs.io/en/latest/index.html>`_ to analyse the results of this sample: get marginalized statistics, convergence diagnostics and some plots. We recommend using the `graphical user interface <http://getdist.readthedocs.io/en/latest/gui.html>`_. Simply run ``GetDistGUI.py`` from anywhere, press the green ``+`` button, navigate in the pop-up window into the folder containing the chains (here ``chains``) and click ``choose``. Now you can get some result statistics from the ``Data`` menu, or generate some plots like this one (just mark the the options in the red boxes and hit ``Make plot``):

.. image:: img_output_getdistgui.png

.. note::

   For a detailed user manual and many more examples, check out the `GetDist documentation <http://getdist.readthedocs.io/en/latest/index.html>`_!


From a Python interpreter
-------------------------

You use **cobaya** interactively within a Python interpreter or a Jupyter notebook. This will allow you to create input and process products *programatically*, making it easier to streamline a complicated analyses.

The actual input information of **cobaya** are Python *dictionaries* (a ``yaml`` file is just a representation of it). We can easily define the same information above as a dictionary:

.. literalinclude:: ./src_examples/quickstart/create_info.py
   :language: python


- NEEDS TESTING FOR CONSISTENCY!!!

.. code:: python

    from cobaya.input import load_input
    info = load_input("gaussian.yaml")
    info.pop("output_prefix")  # suppresses external output

  
- comment of lack of output_prefix

# run the sampler
from cobaya.run import run
updated_info, products = run(info)


.. note::

   Notice that the parameters are defined here using an `OrderedDict <https://docs.python.org/2/library/collections.html#ordereddict-examples-and-recipes>`_, instead of a normal dictionary. This is optional (a normal dictionary can be used), but recommended: it keeps the order consistent between input and output. Same goes for the likelihoods, when there is more than one.


You can also do your analysis in a Python terminal or notebook, using either your own tools or the methods in GetDist.

For example, assuming that you have just run the example in :ref:`in_example_script`, to generate a similar plot to the one above and some statistics, simply do:

.. code:: python

   %matplotlib inline
   import getdist as gd
   import getdist.plots as gdplt
   gdsamples = products["sample"].as_getdist_mcsamples()
   gdplot = gdplt.getSubplotPlotter()
   gdplot.triangle_plot(gdsamples, ["mock_a", "mock_b"], filled=True)
   print "Covariace matrix:\n", gdsamples.getCovMat().matrix[:2,:2]

COMMENT ON LOADING CHAINS FROM HDD IF WRITTEN THERE!!!


