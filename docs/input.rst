Input and invocation
====================

The input of **cobaya** consists of a collection of Python dictionaries specifying the different parts of the code to use (likelihoods, theory codes and samplers) and which parameters to sample and how. This information is often provided as a text file in the `YAML <https://en.wikipedia.org/wiki/YAML>`_ language, which is so simple that you don't even have to learn it.

.. _in_example:

An example input file
---------------------

.. code:: yaml


That input structure above contains three *blocks*:

- ``likelihood``: Specifies a list of likelihoods to sample, with their corresponding options -- here a Gaussian likelihood with the mean and covariance specified under it.
- ``params``: lists the parameters to be explored, and for each its prior and a LaTeX label to be used when producing plots. The parameters ``mock_a`` and ``mock_b`` will be sampled (they have a prior specified under them), whereas ``mock_derived_a`` and ``mock_derived_b`` are derived quantities computed by the likelihood (they don't have a prior).
- ``sampler``: specifies which sampler to use, here the internal MCMC sampler, and some options for it: the number of burn-in samples and the total number of samples to be drawn.

The *top-level option* ``output_prefix: chains/gaussian`` indicates that the resulting output files will be placed in a folder called ``chains`` (which will be created if necessary), and the corresponding output files will start with ``gaussian``.

You can save that piece of text as is into a file, e.g. ``example_gaussian.yaml``, and run **cobaya** as

.. code:: bash

   $ cobaya-run example_gaussian.yaml

It will be finished in a few seconds. The result of that sample is discussed in the :ref:`next section <out_example>`.

For a more accurate description of the blocks and their usage continue reading below -- :ref:`input_blocks`.


.. _input_blocks:

Basic input structure
---------------------

There are 5 different input blocks (two of them optional), which can be specified in any order:

- ``likelihood``: contains the likelihoods that are going to be explored, and their respective options -- see :doc:`likelihoods`

- ``params``: contains a list of parameters to be *fixed*, *sampled* or *derived*, their priors, LaTeX labels, etc. -- see :doc:`params_prior`.

- ``prior``: (optional) contains additional priors to be imposed, whenever they are complex or non-separable -- see :ref:`prior_external`.

- ``sampler``: contains the sampler as a single entry, and its options -- see :doc:`sampler`.

- ``theory`` (optional): has only one entry, which specifies the theory code with which to compute the observables used by the likelihoods, and options for it. Also, if a ``theory`` is specified, the ``params`` block may contain a ``theory`` sub-block containing the parameters belonging to the theory code -- see :doc:`theory` and also :doc:`examples_planck` for a usage example.

The modules specified above (i.e. likelihoods, samplers, theories...) can have any number of options, but you don't need to specify all of them every time you use them: if an option is not specified, its **default** value is used. The default values for each module are described in their respective section of the documentation, and in a ``defaults.yaml`` file in the folder of **cobaya** where that module is defined, e.g. ``cobaya/cobaya/likelihoods/gaussian/defaults.yaml`` for the defaults of the ``gaussian`` likelihood.

In addition, there are some *top level* options (i.e. defined outside any block):

+ ``output_prefix``: determines where the output files are written and/or a prefix for their names -- see :ref:`output_prefix`.
+ ``path_to_modules``: path where the external modules have been automatically installed -- see :doc:`installation_ext`.
+ ``debug``: sets the verbosity level of the output. By default (undefined or ``False``), it produces a rather informative output, reporting on initialization, overall progress and results. If ``True``, it produces a very verbose output (a few lines per sample) that can be used for debugging. You can also set it directly to a particular `integer level of the Python logger <https://docs.python.org/2/library/logging.html#logging-levels>`_, e.g. 40 to produce error output only.
+ ``debug_file``: a file name, with a relative or absolute path if desired, to which to send all logged output. When used, only basic progress info is printed on-screen, and the full debug output (if ``debug: True``) will be sent to this file instead

Some common YAML *gotchas*
--------------------------

+ **use colons(+space), not equal signs!** Values are assigned with a ``:``, not a ``=``; e.g. the following input would produce an error:

  .. code:: yaml

     sampler:
       mcmc:
         burn_in = 10   # ERROR: should be 'burn_in: 10'
         max_tries:100  # ERROR: should have a space: 'max_tries: 100'

+ **missing colons!** Each module or parameter definition, even if it is a bare *mention* and does not have options, must end in a colon (which is actually equivalent to writing a null value ``null`` after the colon); e.g. the following input would produce an error:

  .. code:: yaml

     sampler:
       mcmc  # ERROR: no colon!

+ **indentation!** Block indentation must be *coherent*, i.e. everything within the same block must be the same number of spaces to the right; e.g. the following input would produce two errors

  .. code:: yaml

     sampler:
       mcmc:
         burn_in: 10
          max_samples: 100  # ERROR: misaligned!

     params:
       mock_a:
         prior:
           min: 0
           max: 1
          latex: \alpha  # ERROR: misaligned!

  Above, ``max_samples`` should be aligned to ``burn_in``, because both belong into ``mcmc``. In the same way, ``latex`` should be aligned to ``prior``, since both belong into the definition of the parameter ``mock_a``.

.. note::

   For the YAML *connoisseur*, notice that the YAML parser used here has been modified to simplify the input/output notation: it now retains the ordering of parameters and likelihoods (loads mappings as `OrderedDict <https://docs.python.org/2/library/collections.html#ordereddict-examples-and-recipes>`_) and prints arrays as lists.


.. _in_example_script:

Scripted input -- Python dictionaries
-------------------------------------

You can invoke **cobaya** directly from a Python interpreter or the Jupyter notebook. If you have saved the example above in a file named ``example_gaussian.yaml`` in Python's working directory:

.. code:: python

    from cobaya.run import run
    from cobaya.input import load_input
    input_file = "example_gaussian.yaml"
    info = load_input(input_file)
    info.pop("output_prefix", None)  # suppresses external output
    updated_info, products = run(info)

But, actually, the YAML file is simply parsed as a Python dictionary, so you could as well have defined it by hand:

.. code:: python

    from collections import OrderedDict as odict
    from cobaya.run import run
    info = {"params": odict([
               ("mock_a", {"prior": {"min": -0.5, "max": 3}, "latex": r"\alpha"}),
               ("mock_b", {"prior": {"min": -1,   "max": 4}, "latex": r"\beta",
                           "ref":0.5, "proposal":0.5}),
               ("mock_derived_a", {"latex": r"\alpha^\prime"}),
               ("mock_derived_b", {"latex": r"\beta^\prime"})]),
            "likelihood": {"gaussian": {
               "mean": [0.2, 0],
               "cov": [[0.1, 0.05],
                       [0.05,0.2]]}},
            "sampler": {"mcmc": {"burn_in": 100, "max_samples": 1000}}}
    # run the sampler
    updated_info, products = run(info)

The analysis of this sample in an scripted way is discussed in :ref:`out_example_scripted`.

.. note::

   Notice that the parameters are defined here using an `OrderedDict <https://docs.python.org/2/library/collections.html#ordereddict-examples-and-recipes>`_, instead of a normal dictionary. This is optional (a normal dictionary can be used), but recommended: it keeps the order consistent between input and output. Same goes for the likelihoods, when there is more than one.


.. _input_cont:

Continuing a sample
-------------------

.. todo::

   Sample continuation is not implemented yet.

